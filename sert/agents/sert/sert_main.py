from math import pi
from functools import wraps

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from helpers.network_utils import DenseBlock, SpatialSoftmax3D, Conv3DBlock, Conv3DUpsampleBlock

def exists(x): return x is not None
def default(v, d): return v if exists(v) else d

def cache_fn(f):
    cache = None
    def cached_fn(*args, _cache=True, **kwargs):
        nonlocal cache
        if not _cache: return f(*args, **kwargs)
        if cache is not None: return cache
        cache = f(*args, **kwargs); return cache
    return cached_fn

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None
    def forward(self, x, **kwargs):
        x = self.norm(x)
        if exists(self.norm_context):
            c = kwargs['context']; kwargs.update(context=self.norm_context(c))
        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, g = x.chunk(2, dim=-1); return x * F.gelu(g)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim*mult*2), GEGLU(), nn.Linear(dim*mult, dim))
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.to_q = nn.Linear(query_dim, inner, bias=False)
        self.to_kv = nn.Linear(context_dim, inner * 2, bias=False)
        self.to_out = nn.Linear(inner, query_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, context=None, mask=None):
        h = self.heads
        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg = -torch.finfo(sim.dtype).max
            mask = mask[:, None, :].repeat(h, 1, 1)
            sim.masked_fill_(~mask, max_neg)
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class PerceiverVoxelLangEncoder(nn.Module):
    def __init__(self,
                 depth,
                 iterations,
                 voxel_size,
                 initial_dim,
                 low_dim_size,
                 layer=0,
                 num_rotation_classes=72,
                 num_grip_classes=2,
                 num_collision_classes=2,
                 input_axis=3,
                 num_latents=512,
                 im_channels=64,
                 latent_dim=512,
                 cross_heads=1,
                 latent_heads=8,
                 cross_dim_head=64,
                 latent_dim_head=64,
                 activation='relu',
                 weight_tie_layers=False,
                 pos_encoding_with_lang=True,
                 input_dropout=0.1,
                 attn_dropout=0.1,
                 decoder_dropout=0.0,
                 lang_fusion_type='seq',
                 voxel_patch_size=9,
                 voxel_patch_stride=8,
                 no_skip_connection=False,
                 no_perceiver=False,
                 no_language=False,
                 final_dim=64,
                 safety_dim=2,
                 chunk_len=0,
                 use_rl_heads: bool = False,
                 risk_len: int = 0):
        super().__init__()
        self.depth = depth
        self.layer = layer
        self.init_dim = int(initial_dim)
        self.iterations = iterations
        self.input_axis = input_axis
        self.voxel_size = voxel_size
        self.low_dim_size = low_dim_size
        self.im_channels = im_channels
        self.pos_encoding_with_lang = pos_encoding_with_lang
        self.lang_fusion_type = lang_fusion_type
        self.voxel_patch_size = voxel_patch_size
        self.voxel_patch_stride = voxel_patch_stride
        self.num_rotation_classes = num_rotation_classes
        self.num_grip_classes = num_grip_classes
        self.num_collision_classes = num_collision_classes
        self.final_dim = final_dim
        self.input_dropout = input_dropout
        self.attn_dropout = attn_dropout
        self.decoder_dropout = decoder_dropout
        self.no_skip_connection = no_skip_connection
        self.no_perceiver = no_perceiver
        self.no_language = no_language
        self.safety_dim = safety_dim
        self.chunk_len = int(chunk_len)
        self.use_rl_heads = bool(use_rl_heads)
        self.risk_len = int(risk_len)

        spatial_size = voxel_size // self.voxel_patch_stride

        # --- Language & visual token dims before sequence cross-attn ---
        lang_feat_dim, lang_emb_dim, lang_max_seq_len = 1024, 512, 77

        if self.lang_fusion_type == 'concat':
            # visual tokens만으로 self.im_channels*4 (ss0/global/ss1/global 같은 요약 포함 설계와 호환)
            self.input_dim_before_seq = self.im_channels * 4
            self.lang_preprocess = nn.Linear(lang_feat_dim, self.im_channels)
            self.seq_proj = None
        else:
            # seq 융합: (patchify + safety + (optional) proprio)의 채널을 선형 사상해 맞춘다
            # patchify: C, safety: C, proprio(optional): C  => 2C or 3C
            x_ch = self.im_channels + self.im_channels + (self.im_channels if self.low_dim_size > 0 else 0)
            self.input_dim_before_seq = self.im_channels * 2  # 시퀀스 토큰 폭(언어 임베딩 투영 폭과 동일하게 유지)
            self.lang_preprocess = nn.Linear(lang_emb_dim, self.input_dim_before_seq)
            self.seq_proj = nn.Linear(x_ch, self.input_dim_before_seq)

        if self.pos_encoding_with_lang:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, lang_max_seq_len + spatial_size ** 3, self.input_dim_before_seq)
            )
        else:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, spatial_size, spatial_size, spatial_size, self.input_dim_before_seq)
            )

        # --- 3D conv stem ---
        self.input_preprocess = Conv3DBlock(
            self.init_dim, self.im_channels, kernel_sizes=1, strides=1, norm=None, activation=activation
        )
        self.patchify = Conv3DBlock(
            self.input_preprocess.out_channels, self.im_channels,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation
        )

        if self.low_dim_size > 0:
            self.proprio_preprocess = DenseBlock(self.low_dim_size, self.im_channels, norm=None, activation=activation)
        else:
            self.proprio_preprocess = None

        self.safety_preprocess = DenseBlock(self.safety_dim, self.im_channels, norm=None, activation=activation)

        self.local_maxp = nn.MaxPool3d(3, 2, padding=1)
        self.global_maxp = nn.AdaptiveMaxPool3d(1)

        self.ss0 = SpatialSoftmax3D(self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels)
        flat_size = self.im_channels * 4

        # --- Perceiver ---
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim,
                    Attention(latent_dim, self.input_dim_before_seq,
                              heads=cross_heads, dim_head=cross_dim_head, dropout=input_dropout),
                    context_dim=self.input_dim_before_seq),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head, dropout=attn_dropout))
        get_latent_ff   = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}
        for _ in range(depth):
            self.layers.append(nn.ModuleList([get_latent_attn(**cache_args), get_latent_ff(**cache_args)]))

        # decoder side
        self.decoder_cross_attn = PreNorm(
            self.input_dim_before_seq,
            Attention(self.input_dim_before_seq, latent_dim, heads=cross_heads, dim_head=cross_dim_head, dropout=decoder_dropout),
            context_dim=latent_dim
        )
        self.up0 = Conv3DUpsampleBlock(
            self.input_dim_before_seq, self.final_dim,
            kernel_sizes=self.voxel_patch_size, strides=self.voxel_patch_stride,
            norm=None, activation=activation
        )
        self.ss1 = SpatialSoftmax3D(spatial_size, spatial_size, spatial_size, self.input_dim_before_seq)
        flat_size += self.input_dim_before_seq * 4

        # ✅ 최종 합성 입력 채널을 플래그에 맞춰 한 번만 정의
        if self.no_skip_connection:
            final_in_ch = self.final_dim          # u = self.final(u0)
        elif self.no_perceiver:
            final_in_ch = self.im_channels        # u = self.final(d0)
        else:
            final_in_ch = self.im_channels + self.final_dim  # u = self.final(cat([d0, u0], dim=1))

        self.final = Conv3DBlock(
            final_in_ch, self.im_channels, kernel_sizes=3, strides=1, norm=None, activation=activation
        )
        # ✅ final의 출력 채널(self.im_channels)에 맞게 trans_decoder 입력 채널 설정
        self.trans_decoder = Conv3DBlock(self.im_channels, 1, kernel_sizes=3, strides=1, norm=None, activation=None)

        # heads (rot / grip / collision / safety)
        self.ss_final = None
        self.dense0 = None
        self.dense1 = None
        self.rot_grip_collision_ff = None
        self.safety_ff = None
        if self.num_rotation_classes > 0:
            self.ss_final = SpatialSoftmax3D(self.voxel_size, self.voxel_size, self.voxel_size, self.im_channels)
            flat_size += self.im_channels * 4
            self.dense0 = DenseBlock(flat_size, 256, None, activation)
            self.dense1 = DenseBlock(256, self.final_dim, None, activation)
            self.rot_grip_collision_ff = DenseBlock(
                self.final_dim, self.num_rotation_classes * 3 + self.num_grip_classes + self.num_collision_classes, None, None
            )
            self.safety_ff = DenseBlock(self.final_dim, 1, None, None)

        # optional chunk head (unused by caller now, kept for extension)
        self.chunk_ff = None
        if self.chunk_len > 0:
            self.chunk_ff = nn.Sequential(
                nn.Linear(self.final_dim, self.final_dim),
                nn.GELU(),
                nn.Linear(self.final_dim, self.chunk_len * 8)
            )
        self.value_reward_head = None
        self.value_cost_head = None
        self.risk_ff = None
        if self.use_rl_heads:
            self.value_reward_head = DenseBlock(self.final_dim, 1, None, None)
            self.value_cost_head = DenseBlock(self.final_dim, 1, None, None)
            if self.risk_len > 0:
                self.risk_ff = nn.Sequential(
                    nn.Linear(self.final_dim, self.final_dim),
                    nn.GELU(),
                    nn.Linear(self.final_dim, self.risk_len)
                )

    def forward(self, ins, proprio, safety_now, lang_goal_emb, lang_token_embs, prev_layer_voxel_grid, bounds, prev_layer_bounds, mask=None):
        # stem
        d0 = self.input_preprocess(ins)  # (B, C, D, H, W) with C = im_channels
        feats = [self.ss0(d0.contiguous()), self.global_maxp(d0).view(ins.shape[0], -1)]

        x = self.patchify(d0)            # (B, C, d, h, w), C = im_channels
        b, c, d, h, w = x.shape

        # proprio → channels
        if self.low_dim_size > 0 and proprio is not None:
            p = self.proprio_preprocess(proprio)             # (B, C)
            p = p[:, :, None, None, None].repeat(1, 1, d, h, w)
            x = torch.cat([x, p], dim=1)                    # C += im_channels

        # safety → channels
        s = self.safety_preprocess(safety_now)               # (B, C)
        s = s[:, :, None, None, None].repeat(1, 1, d, h, w)
        x = torch.cat([x, s], dim=1)                        # C += im_channels

        # to sequence
        x = rearrange(x, 'b c d h w -> b d h w c')
        qshape = x.shape
        x = rearrange(x, 'b d h w c -> b (d h w) c')         # (B, N, Csum)

        # language fusion (seq)
        if self.lang_fusion_type == 'seq':
            x = self.seq_proj(x)                             # (B, N, input_dim_before_seq)
            l = self.lang_preprocess(lang_token_embs)        # (B, T, input_dim_before_seq)
            x = torch.cat((l, x), dim=1)                     # (B, T+N, input_dim_before_seq)

        # positional encoding
        if self.pos_encoding_with_lang:
            x = x + self.pos_encoding

        # perceiver core
        latents = repeat(self.latents, 'n d -> b n d', b=b)
        ca, ff = self.cross_attend_blocks
        for _ in range(self.iterations):
            latents = ca(latents, context=x, mask=mask) + latents
            latents = ff(latents) + latents
            for sa, sff in self.layers:
                latents = sa(latents) + latents
                latents = sff(latents) + latents

        # decoder cross-attn → back to 3D grid tokens (remove language tokens)
        dec = self.decoder_cross_attn(x, context=latents)     # (B, T+N, Cseq)
        if self.lang_fusion_type == 'seq':
            dec = dec[:, l.shape[1]:]                        # keep only N
        dec = dec.view(b, *qshape[1:-1], dec.shape[-1])      # (B, d, h, w, Cseq)
        dec = rearrange(dec, 'b d h w c -> b c d h w')       # (B, Cseq, d, h, w)

        # upsample and skip
        feats.extend([self.ss1(dec.contiguous()), self.global_maxp(dec).view(b, -1)])
        u0 = self.up0(dec)                                   # (B, final_dim, D, H, W)

        if self.no_skip_connection:
            u = self.final(u0)
        elif self.no_perceiver:
            u = self.final(d0)
        else:
            u = self.final(torch.cat([d0, u0], dim=1))       # (B, im_channels, D, H, W)

        # translation heatmap
        trans = self.trans_decoder(u)                        # (B, 1, D, H, W)

        # heads
        rot_and_grip_out, collision_out, safety_out, chunk_out = None, None, None, None
        value_r, value_c, risk_out = None, None, None
        if self.num_rotation_classes > 0:
            feats.extend([self.ss_final(u.contiguous()), self.global_maxp(u).view(b, -1)])
            d0f = self.dense0(torch.cat(feats, dim=1))
            d1 = self.dense1(d0f)
            rgc = self.rot_grip_collision_ff(d1)
            rot_and_grip_out = rgc[:, :-self.num_collision_classes]
            collision_out = rgc[:, -self.num_collision_classes:]
            safety_out = self.safety_ff(d1)

            if self.chunk_ff is not None:
                chunk_out = self.chunk_ff(d1).view(b, self.chunk_len, 8)

            if self.use_rl_heads:
                value_r = self.value_reward_head(d1)
                value_c = self.value_cost_head(d1)
                if self.risk_ff is not None:
                    risk_out = self.risk_ff(d1)

        if self.use_rl_heads:
            return trans, rot_and_grip_out, collision_out, safety_out, chunk_out, value_r, value_c, risk_out

        return trans, rot_and_grip_out, collision_out, safety_out, chunk_out
