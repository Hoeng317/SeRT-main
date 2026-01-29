import copy
import logging
import os
from typing import List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from yarr.agents.agent import Agent, ActResult, ScalarSummary, HistogramSummary, ImageSummary, Summary

from helpers import utils
from helpers.utils import visualise_voxel
from voxel.voxel_grid import VoxelGrid
from voxel.augmentation import apply_se3_augmentation
from helpers.clip.core.clip import build_model, load_clip

import transformers
from helpers.optim.lamb import Lamb
from torch.nn.parallel import DistributedDataParallel as DDP

NAME = 'QAttentionAgent'


class QFunction(nn.Module):
    def __init__(
        self,
        perceiver_encoder: nn.Module,
        voxelizer: VoxelGrid,
        bounds_offset: float,
        rotation_resolution: float,
        device,
        training: bool,
        use_ddp: bool = True,
        find_unused_parameters: bool = False,
    ):
        super().__init__()
        self._rotation_resolution = rotation_resolution
        self._voxelizer = voxelizer
        self._bounds_offset = bounds_offset
        self._qnet = perceiver_encoder.to(device)
        if use_ddp and training and isinstance(device, torch.device) and device.type == 'cuda' and device.index is not None:
            self._qnet = DDP(
                self._qnet,
                device_ids=[device.index],
                output_device=device.index,
                find_unused_parameters=find_unused_parameters,
            )

    @staticmethod
    def _argmax_3d(t: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = t.shape
        flat = t.view(B, C, -1)
        idxs = flat.argmax(-1)
        z = (idxs // (H * W))
        y = (idxs // W) % H
        x = idxs % W
        return torch.cat([z, y, x], dim=1).to(torch.long)

    def choose_highest_action(self, q_trans, q_rot_grip, q_collision):
        coords = self._argmax_3d(q_trans)
        rot_and_grip_indicies, ignore_collision = None, None
        if q_rot_grip is not None:
            nrot = int(360 // self._rotation_resolution)
            q_rot = torch.stack(torch.split(q_rot_grip[:, :-2], nrot, dim=1), dim=1)
            rot_and_grip_indicies = torch.cat(
                [
                    q_rot[:, 0:1].argmax(-1),
                    q_rot[:, 1:2].argmax(-1),
                    q_rot[:, 2:3].argmax(-1),
                    q_rot_grip[:, -2:].argmax(-1, keepdim=True),
                ],
                dim=-1,
            )
            if q_collision is not None:
                ignore_collision = q_collision.argmax(-1, keepdim=True).to(torch.long)
        return coords, rot_and_grip_indicies, ignore_collision

    def forward(
        self,
        rgb_pcd,
        proprio,
        pcd,
        lang_goal_emb,
        lang_token_embs,
        safety_now,
        bounds=None,
        prev_bounds=None,
        prev_layer_voxel_grid=None,
        return_extra: bool = False,
    ):
        b = rgb_pcd[0][0].shape[0]
        pcd_flat = torch.cat([p.permute(0, 2, 3, 1).reshape(b, -1, 3) for p in pcd], 1)
        rgb = [rp[0] for rp in rgb_pcd]
        feat_size = rgb[0].shape[1]
        flat_imag_features = torch.cat([p.permute(0, 2, 3, 1).reshape(b, -1, feat_size) for p in rgb], 1)
        voxel_grid = self._voxelizer.coords_to_bounding_voxel_grid(pcd_flat, coord_features=flat_imag_features, coord_bounds=bounds)
        voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach()
        if bounds.shape[0] != b:
            bounds = bounds.repeat(b, 1)
        qnet_out = self._qnet(
            voxel_grid,
            proprio,
            safety_now,
            lang_goal_emb,
            lang_token_embs,
            prev_layer_voxel_grid,
            bounds,
            prev_bounds,
        )
        if isinstance(qnet_out, (list, tuple)) and len(qnet_out) == 8:
            q_trans, q_rot_and_grip, q_ignore_collisions, q_safety, chunk_actions, value_r, value_c, risk_pred = qnet_out
        else:
            q_trans, q_rot_and_grip, q_ignore_collisions, q_safety, chunk_actions = qnet_out
            value_r, value_c, risk_pred = None, None, None

        if return_extra:
            return q_trans, q_rot_and_grip, q_ignore_collisions, q_safety, chunk_actions, value_r, value_c, risk_pred, voxel_grid
        return q_trans, q_rot_and_grip, q_ignore_collisions, q_safety, chunk_actions, voxel_grid


class QAttentionPerActBCAgent(Agent):
    def __init__(
        self,
        layer: int,
        coordinate_bounds: list,
        perceiver_encoder: nn.Module,
        camera_names: list,
        batch_size: int,
        voxel_size: int,
        bounds_offset: float,
        voxel_feature_size: int,
        image_crop_size: int,
        num_rotation_classes: int,
        rotation_resolution: float,
        chunk_len: int = 0,
        chunk_loss_weight: float = 0.0,
        chunk_action_steps: int = 0,
        safety_tau: float = 0.5,
        lr: float = 1e-4,
        lr_scheduler: bool = False,
        training_iterations: int = 100000,
        num_warmup_steps: int = 20000,
        trans_loss_weight: float = 1.0,
        rot_loss_weight: float = 1.0,
        grip_loss_weight: float = 1.0,
        collision_loss_weight: float = 1.0,
        include_low_dim_state: bool = False,
        image_resolution: list = None,
        lambda_weight_l2: float = 0.0,
        transform_augmentation: bool = True,
        transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
        transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
        transform_augmentation_rot_resolution: int = 5,
        optimizer_type: str = 'adam',
        num_devices: int = 1,
    ):
        self._layer = layer
        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._voxel_feature_size = voxel_feature_size
        self._bounds_offset = bounds_offset
        self._image_crop_size = image_crop_size
        self._lr = lr
        self._lr_scheduler = lr_scheduler
        self._training_iterations = training_iterations
        self._num_warmup_steps = num_warmup_steps
        self._trans_loss_weight = trans_loss_weight
        self._rot_loss_weight = rot_loss_weight
        self._grip_loss_weight = grip_loss_weight
        self._collision_loss_weight = collision_loss_weight
        self._include_low_dim_state = include_low_dim_state
        self._image_resolution = image_resolution or [128, 128]
        self._voxel_size = voxel_size
        self._camera_names = camera_names
        self._num_cameras = len(camera_names)
        self._batch_size = batch_size
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = torch.from_numpy(np.array(transform_augmentation_xyz))
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type
        self._num_devices = num_devices
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._chunk_len = max(0, int(chunk_len))
        self._chunk_loss_weight = float(chunk_loss_weight)
        self._chunk_action_steps = int(chunk_action_steps)
        self._safety_tau_on = float(safety_tau)
        if self._chunk_len > 0:
            if self._chunk_action_steps <= 0 or self._chunk_action_steps >= self._chunk_len:
                self._chunk_action_steps = max(1, self._chunk_len - 1)
        self._chunk_cycle_step = 0
        self._last_action_coords = None

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self._name = NAME + '_layer' + str(self._layer)
        self._last_p_unsafe_mean = None
        self._last_act_p_unsafe = None
        self._last_scalar_logs = {}

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device or torch.device('cpu')
        self._chunk_cycle_step = 0
        self._last_action_coords = None

        self._voxelizer = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=self._device,
            batch_size=self._batch_size if training else 1,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod(self._image_resolution) * self._num_cameras,
        )

        self._q = QFunction(
            self._perceiver_encoder,
            self._voxelizer,
            self._bounds_offset,
            self._rotation_resolution,
            self._device,
            training,
        ).to(self._device).train(training)

        grid_for_crop = torch.arange(0, self._image_crop_size, device=self._device).unsqueeze(0).repeat(self._image_crop_size, 1).unsqueeze(-1)
        self._grid_for_crop = torch.cat([grid_for_crop.transpose(1, 0), grid_for_crop], dim=2).unsqueeze(0)
        self._coordinate_bounds = torch.tensor(self._coordinate_bounds, device=self._device).unsqueeze(0)

        if self._training:
            if self._optimizer_type == 'lamb':
                self._optimizer = Lamb(self._q.parameters(), lr=self._lr, weight_decay=self._lambda_weight_l2, betas=(0.9, 0.999), adam=False)
            elif self._optimizer_type == 'adam':
                self._optimizer = torch.optim.Adam(self._q.parameters(), lr=self._lr, weight_decay=self._lambda_weight_l2)
            else:
                raise Exception('Unknown optimizer type')

            if self._lr_scheduler:
                self._scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
                    self._optimizer,
                    num_warmup_steps=self._num_warmup_steps,
                    num_training_steps=self._training_iterations,
                    num_cycles=max(1, self._training_iterations // 10000),
                )

            B = self._batch_size
            V = self._voxel_size
            R = self._num_rotation_classes
            self._action_trans_one_hot_zeros = torch.zeros((B, 1, V, V, V), dtype=int, device=self._device)
            self._action_rot_x_one_hot_zeros = torch.zeros((B, R), dtype=int, device=self._device)
            self._action_rot_y_one_hot_zeros = torch.zeros((B, R), dtype=int, device=self._device)
            self._action_rot_z_one_hot_zeros = torch.zeros((B, R), dtype=int, device=self._device)
            self._action_grip_one_hot_zeros = torch.zeros((B, 2), dtype=int, device=self._device)
            self._action_ignore_collisions_one_hot_zeros = torch.zeros((B, 2), dtype=int, device=self._device)

            logging.info('# Q Params: %d' % sum(p.numel() for name, p in self._q.named_parameters() if p.requires_grad and 'clip' not in name))
        else:
            for p in self._q.parameters():
                p.requires_grad = False
            model, _ = load_clip("RN50", jit=False)
            self._clip_rn50 = build_model(model.state_dict()).float().to(self._device)
            self._clip_rn50.eval()
            del model
            self._voxelizer.to(self._device)
            self._q.to(self._device)

    def _preprocess_inputs(self, replay_sample):
        obs, pcds = [], []
        self._crop_summary = []
        for n in self._camera_names:
            rgb = replay_sample[f'{n}_rgb']
            pcd = replay_sample[f'{n}_point_cloud']
            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _act_preprocess_inputs(self, observation):
        obs, pcds = [], []
        for n in self._camera_names:
            rgb = observation[f'{n}_rgb']
            pcd = observation[f'{n}_point_cloud']
            obs.append([rgb, pcd])
            pcds.append(pcd)
        return obs, pcds

    def _celoss(self, pred, labels_onehot):
        return self._cross_entropy_loss(pred, labels_onehot.argmax(-1))

    def _softmax_q_trans(self, q):
        B, C, D, H, W = q.shape
        return F.softmax(q.view(B, -1), dim=1).view(B, C, D, H, W)

    def _softmax_q_rot_grip(self, q_rot_grip):
        if q_rot_grip is None:
            return None
        R = self._num_rotation_classes
        qx = q_rot_grip[:, 0 * R : 1 * R]
        qy = q_rot_grip[:, 1 * R : 2 * R]
        qz = q_rot_grip[:, 2 * R : 3 * R]
        qg = q_rot_grip[:, 3 * R :]
        return torch.cat([F.softmax(qx, dim=1), F.softmax(qy, dim=1), F.softmax(qz, dim=1), F.softmax(qg, dim=1)], dim=1)

    def _softmax_ignore_collision(self, q_collision):
        return None if q_collision is None else F.softmax(q_collision, dim=1)

    @staticmethod
    def _to_device_list(maybe_list: Union[List[torch.Tensor], torch.Tensor, None], device):
        if maybe_list is None:
            return None
        if isinstance(maybe_list, list):
            return [t.to(device) for t in maybe_list]
        return maybe_list.to(device)

    def update(self, step: int, replay_sample: dict) -> dict:
        device = self._device

        action_trans = replay_sample['trans_action_indicies'][:, self._layer * 3 : self._layer * 3 + 3].int().to(device)
        action_rot_grip = replay_sample['rot_grip_action_indicies'].int().to(device)
        action_gripper_pose = replay_sample['gripper_pose'].to(device)
        action_ignore_collisions = replay_sample['ignore_collisions'].int().to(device)
        lang_goal_emb = replay_sample['lang_goal_emb'].float().to(device)
        lang_token_embs = replay_sample['lang_token_embs'].float().to(device)
        prev_layer_voxel_grid = self._to_device_list(replay_sample.get('prev_layer_voxel_grid', None), device)
        prev_layer_bounds = self._to_device_list(replay_sample.get('prev_layer_bounds', None), device)
        safety_now = replay_sample['safety_now'].float().to(device)
        safety_future_label = replay_sample['safety_future_label'].float().to(device)
        future_actions = replay_sample.get('future_actions', None)
        if isinstance(future_actions, torch.Tensor):
            future_actions = future_actions.float().to(device)
        elif future_actions is not None:
            future_actions = torch.as_tensor(future_actions, dtype=torch.float32, device=device)

        bounds = self._coordinate_bounds.to(device)
        if self._layer > 0:
            cp = replay_sample[f'attention_coordinate_layer_{self._layer - 1}'].to(device)
            bounds = torch.cat([cp - self._bounds_offset, cp + self._bounds_offset], dim=1)

        proprio = replay_sample['low_dim_state'].to(device) if self._include_low_dim_state else None
        obs, pcd = self._preprocess_inputs(replay_sample)
        pcd = [p.to(device) for p in pcd]
        obs = [[o[0].to(device), o[1].to(device)] for o in obs]
        bs = pcd[0].shape[0]

        if self._transform_augmentation:
            action_trans, action_rot_grip, pcd = apply_se3_augmentation(
                pcd,
                action_gripper_pose,
                action_trans,
                action_rot_grip,
                bounds,
                self._layer,
                self._transform_augmentation_xyz,
                self._transform_augmentation_rpy,
                self._transform_augmentation_rot_resolution,
                self._voxel_size,
                self._rotation_resolution,
                device,
            )

        q_trans, q_rot_grip, q_collision, q_safety, chunk_pred, voxel_grid = self._q(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            safety_now,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
        )

        coords, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans, q_rot_grip, q_collision)

        action_trans_one_hot = self._action_trans_one_hot_zeros.clone()
        for b in range(bs):
            gt = action_trans[b, :].int()
            action_trans_one_hot[b, :, gt[0], gt[1], gt[2]] = 1

        q_trans_flat = q_trans.view(bs, -1)
        trans_one_hot_flat = action_trans_one_hot.view(bs, -1)
        q_trans_loss = self._celoss(q_trans_flat, trans_one_hot_flat)

        q_rot_loss = 0.0
        q_grip_loss = 0.0
        q_collision_loss = 0.0

        with_rot_and_grip = rot_and_grip_indicies is not None
        if with_rot_and_grip:
            act_rx = self._action_rot_x_one_hot_zeros.clone()
            act_ry = self._action_rot_y_one_hot_zeros.clone()
            act_rz = self._action_rot_z_one_hot_zeros.clone()
            act_g = self._action_grip_one_hot_zeros.clone()
            act_ic = self._action_ignore_collisions_one_hot_zeros.clone()

            for b in range(bs):
                gt = action_rot_grip[b, :].int()
                act_rx[b, gt[0]] = 1
                act_ry[b, gt[1]] = 1
                act_rz[b, gt[2]] = 1
                act_g[b, gt[3]] = 1
                gt_ic = action_ignore_collisions[b, :].int()
                act_ic[b, gt_ic[0]] = 1

            R = self._num_rotation_classes
            qx = q_rot_grip[:, 0 * R : 1 * R]
            qy = q_rot_grip[:, 1 * R : 2 * R]
            qz = q_rot_grip[:, 2 * R : 3 * R]
            qg = q_rot_grip[:, 3 * R :]

            q_rot_loss += self._celoss(qx, act_rx)
            q_rot_loss += self._celoss(qy, act_ry)
            q_rot_loss += self._celoss(qz, act_rz)
            q_grip_loss += self._celoss(qg, act_g)

            if q_collision is not None:
                q_collision_loss += self._celoss(q_collision, act_ic)

        safety_loss = 0.0
        if q_safety is not None:
            logits = q_safety.view(-1)
            targets = safety_future_label.view(-1)
            safety_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
            with torch.no_grad():
                self._last_p_unsafe_mean = torch.sigmoid(logits).mean().item()

        chunk_loss = 0.0
        if self._chunk_len > 0 and chunk_pred is not None and future_actions is not None:
            chunk_loss = F.smooth_l1_loss(chunk_pred, future_actions, reduction='mean')

        combined_losses = (
            q_trans_loss * self._trans_loss_weight
            + q_rot_loss * self._rot_loss_weight
            + q_grip_loss * self._grip_loss_weight
            + q_collision_loss * self._collision_loss_weight
            + safety_loss
            + chunk_loss * self._chunk_loss_weight
        )
        total_loss = combined_losses.mean()

        self._optimizer.zero_grad()
        total_loss.backward()
        self._optimizer.step()

        self._summaries = {
            'losses/total_loss': total_loss,
            'losses/trans_loss': q_trans_loss.mean(),
            'losses/rot_loss': q_rot_loss if isinstance(q_rot_loss, torch.Tensor) else torch.as_tensor(q_rot_loss, device=self._device),
            'losses/grip_loss': q_grip_loss if isinstance(q_grip_loss, torch.Tensor) else torch.as_tensor(q_grip_loss, device=self._device),
            'losses/collision_loss': q_collision_loss if isinstance(q_collision_loss, torch.Tensor) else torch.as_tensor(q_collision_loss, device=self._device),
            'losses/safety_bce': safety_loss if isinstance(safety_loss, torch.Tensor) else torch.as_tensor(safety_loss, device=self._device),
        }
        if self._chunk_len > 0:
            self._summaries['losses/chunk_loss'] = chunk_loss if isinstance(chunk_loss, torch.Tensor) else torch.as_tensor(chunk_loss, device=self._device)

        def _scalar(v):
            if isinstance(v, torch.Tensor):
                return float(v.detach().mean().cpu().item())
            return float(v)
        self._last_scalar_logs = {
            f'{self._name}/losses/trans_loss': _scalar(q_trans_loss),
            f'{self._name}/losses/rot_loss': _scalar(q_rot_loss),
            f'{self._name}/losses/grip_loss': _scalar(q_grip_loss),
            f'{self._name}/losses/collision_loss': _scalar(q_collision_loss),
            f'{self._name}/losses/total_loss': _scalar(total_loss),
            f'{self._name}/losses/safety_bce': _scalar(safety_loss),
        }
        if self._chunk_len > 0:
            self._last_scalar_logs[f'{self._name}/losses/chunk_loss'] = _scalar(chunk_loss)
        if self._last_p_unsafe_mean is not None:
            self._last_scalar_logs[f'{self._name}/stats/p_unsafe_mean'] = float(self._last_p_unsafe_mean)

        if self._lr_scheduler:
            self._scheduler.step()
            self._summaries['learning_rate'] = self._scheduler.get_last_lr()[0]

        self._vis_voxel_grid = voxel_grid[0]
        self._vis_translation_qvalue = self._softmax_q_trans(q_trans)[0]  # 전체(B,1,D,H,W)에 softmax 후 [0]
        self._vis_max_coordinate = coords[0]
        self._vis_gt_coordinate = action_trans[0]

        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [voxel_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [voxel_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [self._coordinate_bounds.repeat(bs, 1)]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        return {
            'total_loss': total_loss,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }

    def act(self, step: int, observation: dict, deterministic: bool = False) -> ActResult:
        deterministic = True
        bounds = self._coordinate_bounds
        prev_layer_voxel_grid = observation.get('prev_layer_voxel_grid', None)
        prev_layer_bounds = observation.get('prev_layer_bounds', None)

        if 'lang_goal_tokens' in observation and observation['lang_goal_tokens'] is not None:
            with torch.no_grad():
                tokens = observation['lang_goal_tokens'].long().to(self._device)
                lang_goal_emb, lang_token_embs = self._clip_rn50.encode_text_with_embeddings(tokens[0])
                lang_goal_emb = lang_goal_emb.to(self._device)
                lang_token_embs = lang_token_embs.to(self._device)
        else:
            lang_goal_emb = observation.get('lang_goal_emb', torch.zeros(1, 1024, device=self._device))
            lang_token_embs = observation.get('lang_token_embs', torch.zeros(1, 77, 512, device=self._device))

        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        proprio = observation['low_dim_state'] if self._include_low_dim_state else None

        obs, pcd = self._act_preprocess_inputs(observation)
        obs = [[o[0][0].to(self._device), o[1][0].to(self._device)] for o in obs]
        if self._include_low_dim_state:
            proprio = proprio[0].to(self._device)
        pcd = [p[0].to(self._device) for p in pcd]
        bounds = torch.as_tensor(bounds, device=self._device)

        prev_layer_voxel_grid = self._to_device_list(prev_layer_voxel_grid, self._device)
        prev_layer_bounds = self._to_device_list(prev_layer_bounds, self._device)

        if 'safety_now' in observation and observation['safety_now'] is not None:
            safety_now = observation['safety_now'][0].to(self._device)
            if safety_now.dim() == 1:
                safety_now = safety_now.unsqueeze(0)
        else:
            safety_now = torch.zeros(1, 2, device=self._device)

        q_trans, q_rot_grip, q_ignore_collisions, q_safety, chunk_pred, vox_grid = self._q(
            obs,
            proprio,
            pcd,
            lang_goal_emb,
            lang_token_embs,
            safety_now,
            bounds,
            prev_layer_bounds,
            prev_layer_voxel_grid,
        )

        q_trans = self._softmax_q_trans(q_trans)
        q_rot_grip = self._softmax_q_rot_grip(q_rot_grip)
        q_ignore_collisions = self._softmax_ignore_collision(q_ignore_collisions)

        coords, rot_and_grip_indicies, ignore_collisions = self._q.choose_highest_action(q_trans, q_rot_grip, q_ignore_collisions)

        in_chunk_safety_phase = False
        chunk_phase_label = 'disabled'
        if self._chunk_len > 0:
            chunk_phase_label = 'action'
            in_chunk_safety_phase = self._chunk_cycle_step >= self._chunk_action_steps
            if in_chunk_safety_phase:
                chunk_phase_label = 'safety'

        p_unsafe = 0.0
        if q_safety is not None:
            safety_logits = q_safety.view(-1)
            p_unsafe = torch.sigmoid(safety_logits)[0].item()
        self._last_act_p_unsafe = p_unsafe

        safety_override = None
        def _enforce_stop():
            nonlocal coords, rot_and_grip_indicies, ignore_collisions, safety_override
            if self._last_action_coords is not None:
                coords = self._last_action_coords.clone()
            if ignore_collisions is not None:
                ignore_collisions = torch.zeros_like(ignore_collisions, dtype=torch.int64)
            if rot_and_grip_indicies is not None:
                rot_and_grip_indicies[:, 3] = 1
            safety_override = "stop_or_avoid" if safety_override is None else safety_override

        if p_unsafe >= self._safety_tau_on:
            _enforce_stop()

        if in_chunk_safety_phase and safety_override != "stop_or_avoid":
            _enforce_stop()
            safety_override = "chunk_safety"

        rot_grip_action = rot_and_grip_indicies if q_rot_grip is not None else None
        ignore_collisions_action = ignore_collisions.int() if ignore_collisions is not None else None

        coords = coords.int()
        attention_coordinate = bounds[:, :3] + res * coords + res / 2
        self._last_action_coords = coords.detach().clone()

        if prev_layer_voxel_grid is None:
            prev_layer_voxel_grid = [vox_grid]
        else:
            prev_layer_voxel_grid = prev_layer_voxel_grid + [vox_grid]

        if prev_layer_bounds is None:
            prev_layer_bounds = [bounds]
        else:
            prev_layer_bounds = prev_layer_bounds + [bounds]

        observation_elements = {
            'attention_coordinate': attention_coordinate,
            'prev_layer_voxel_grid': prev_layer_voxel_grid,
            'prev_layer_bounds': prev_layer_bounds,
        }
        info = {
            f'voxel_grid_depth{self._layer}': vox_grid,
            f'q_depth{self._layer}': q_trans,
            f'voxel_idx_depth{self._layer}': coords,
            'p_unsafe': p_unsafe,
            'safety_override': safety_override,
            'chunk_phase': chunk_phase_label,
        }
        if self._chunk_len > 0:
            info['chunk_cycle_step'] = self._chunk_cycle_step

        self._act_voxel_grid = vox_grid[0]
        self._act_max_coordinate = coords[0]
        self._act_qvalues = q_trans[0].detach()

        if self._chunk_len > 0:
            self._chunk_cycle_step = (self._chunk_cycle_step + 1) % self._chunk_len

        return ActResult((coords, rot_grip_action, ignore_collisions_action), observation_elements=observation_elements, info=info)

    def update_summaries(self) -> List[Summary]:
        summaries = [
            ImageSummary(
                f'{self._name}/update_qattention',
                transforms.ToTensor()(
                    visualise_voxel(
                        self._vis_voxel_grid.detach().cpu().numpy(),
                        self._vis_translation_qvalue.detach().cpu().numpy(),
                        self._vis_max_coordinate.detach().cpu().numpy(),
                        self._vis_gt_coordinate.detach().cpu().numpy(),
                    )
                ),
            )
        ]
        for n, v in self._summaries.items():
            summaries.append(ScalarSummary(f'{self._name}/{n}', v))
        for tag, param in self._q.named_parameters():
            summaries.append(HistogramSummary(f'{self._name}/weight/{tag}', param.data))
            if param.grad is not None:
                summaries.append(HistogramSummary(f'{self._name}/gradient/{tag}', param.grad))
        if self._last_p_unsafe_mean is not None:
            summaries.append(ScalarSummary(f'{self._name}/stats/p_unsafe_mean', torch.tensor(self._last_p_unsafe_mean)))
        if self._last_act_p_unsafe is not None:
            summaries.append(ScalarSummary(f'{self._name}/stats/p_unsafe_act', torch.tensor(self._last_act_p_unsafe)))
        return summaries

    def act_summaries(self) -> List[Summary]:
        return [
            ImageSummary(
                f'{self._name}/act_Qattention',
                transforms.ToTensor()(
                    visualise_voxel(
                        self._act_voxel_grid.cpu().numpy(),
                        self._act_qvalues.cpu().numpy(),
                        self._act_max_coordinate.cpu().numpy(),
                    )
                ),
            )
        ]

    def get_scalar_logs(self) -> dict:
        return dict(self._last_scalar_logs)

    def load_weights(self, savedir: str):
        weight_file = os.path.join(savedir, f'{self._name}.pt')
        state_dict = torch.load(weight_file, map_location=self._device)
        merged_state = self._q.state_dict()
        for k, v in state_dict.items():
            if not self._training:
                k = k.replace('_qnet.module', '_qnet')
            if k in merged_state:
                merged_state[k] = v
            else:
                if '_voxelizer' not in k:
                    logging.warning(f"key {k} not found in checkpoint")
        self._q.load_state_dict(merged_state)
        print(f"loaded weights from {weight_file}")

    def save_weights(self, savedir: str):
        torch.save(self._q.state_dict(), os.path.join(savedir, f'{self._name}.pt'))
