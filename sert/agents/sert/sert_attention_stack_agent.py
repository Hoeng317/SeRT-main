# qattention_stack_agent.py  (완성본)

from typing import List, Optional, Tuple, Dict, Any

import numpy as np

import torch
from yarr.agents.agent import Agent, ActResult, Summary

from helpers import utils
from agents.peract_bc.qattention_peract_bc_agent import QAttentionPerActBCAgent

NAME = 'QAttentionStackAgent'


class QAttentionStackAgent(Agent):
    def __init__(
        self,
        qattention_agents: List[QAttentionPerActBCAgent],
        rotation_resolution: float,
        camera_names: List[str],
        rotation_prediction_depth: int = 0,
    ):
        super().__init__()
        self._qattention_agents = qattention_agents
        self._rotation_resolution = rotation_resolution
        self._camera_names = camera_names
        self._rotation_prediction_depth = rotation_prediction_depth
        self._device: Optional[torch.device] = None

    # --------------------------
    # Lifecycle
    # --------------------------
    def build(self, training: bool, device: Optional[torch.device] = None) -> None:
        self._device = device or torch.device('cpu')
        for qa in self._qattention_agents:
            qa.build(training, self._device)

    # --------------------------
    # Train
    # --------------------------
    def update(self, step: int, replay_sample: dict) -> dict:
        device = self._device or torch.device('cpu')
        # 0-dim torch scalar로 누적해 offline_train_runner의 .item()에 맞춘다
        total_losses = torch.zeros((), device=device)
        log_scalars: Dict[str, float] = {}

        for qa in self._qattention_agents:
            update_dict = qa.update(step, replay_sample)
            # prev_layer_* 등을 다음 레이어가 볼 수 있게 누적
            propagate = {k: v for k, v in update_dict.items() if k not in ('total_loss', 'log_scalars')}
            replay_sample.update(propagate)

            loss_t = update_dict['total_loss']
            if not isinstance(loss_t, torch.Tensor):
                loss_t = torch.as_tensor(loss_t, device=device)

            # 그래프 분리 후 누적
            total_losses = total_losses + loss_t.detach()
            for name, value in qa.get_scalar_logs().items():
                log_scalars[name] = value

        return {'total_losses': total_losses, 'log_scalars': log_scalars}

    # --------------------------
    # Helpers
    # --------------------------
    def _to_numpy_cam_mats(self, observation: dict, cam: str) -> Tuple:
        """
        RLBench 카메라 행렬 텐서의 다양한 배치 차원 케이스를 안전 처리.
        허용 형태:
          - (B,1,4,4) / (B,1,3,3)
          - (B,4,4)   / (B,3,3)
          - (4,4)     / (3,3)
        """
        ext = observation[f'{cam}_camera_extrinsics']
        intr = observation[f'{cam}_camera_intrinsics']

        ext_np = ext.detach().cpu().numpy()
        intr_np = intr.detach().cpu().numpy()
        if ext_np.ndim > 2:
            ext_np = ext_np.reshape(-1, ext_np.shape[-2], ext_np.shape[-1])[0]
        if intr_np.ndim > 2:
            intr_np = intr_np.reshape(-1, intr_np.shape[-2], intr_np.shape[-1])[0]
        return ext_np, intr_np

    # --------------------------
    # Act
    # --------------------------
    def act(self, step: int, observation: dict, deterministic: bool = False) -> ActResult:
        device = self._device or torch.device('cpu')

        observation_elements: Dict[str, Any] = {}
        translation_results: List[torch.Tensor] = []
        rot_grip_results: List[torch.Tensor] = []
        ignore_collisions_results: List[torch.Tensor] = []

        # 집계용 안전 통계
        infos: Dict[str, Any] = {}
        max_p_unsafe: float = 0.0
        any_safety_override: bool = False

        # 각 레이어를 순차 실행
        for depth, qagent in enumerate(self._qattention_agents):
            act_res = qagent.act(step, observation, deterministic)

            # 다음 레이어 bound 계산에 쓰일 주의(center) 좌표
            attn_coord = act_res.observation_elements['attention_coordinate']  # (B,3), tensor
            observation_elements[f'attention_coordinate_layer_{depth}'] = (
                attn_coord[0].detach().cpu().numpy()
            )

            # 레이어 출력(이산 인덱스)
            trans_idx, rot_grip_idx, ignore_idx = act_res.action
            translation_results.append(trans_idx)

            if rot_grip_idx is not None:
                rot_grip_results.append(rot_grip_idx)
            if ignore_idx is not None:
                ignore_collisions_results.append(ignore_idx)

            # 다음 레이어로 전달할 컨텍스트 복제
            observation['attention_coordinate'] = act_res.observation_elements['attention_coordinate']
            observation['prev_layer_voxel_grid'] = act_res.observation_elements['prev_layer_voxel_grid']
            observation['prev_layer_bounds'] = act_res.observation_elements['prev_layer_bounds']

            # 카메라별 픽셀 좌표 계산(시각화/크롭)
            attn_np = attn_coord[0].detach().cpu().numpy()
            for cam in self._camera_names:
                ext_np, intr_np = self._to_numpy_cam_mats(observation, cam)
                px, py = utils.point_to_pixel_index(attn_np, ext_np, intr_np)  # (x, y)
                pc_t = torch.tensor([[[py, px]]], dtype=torch.float32, device=device)
                observation[f'{cam}_pixel_coord'] = pc_t
                observation_elements[f'{cam}_pixel_coord'] = [int(py), int(px)]

            # 안전 통계/플래그 집계
            infos.update(act_res.info)
            if 'p_unsafe' in act_res.info and isinstance(act_res.info['p_unsafe'], (int, float)):
                max_p_unsafe = max(max_p_unsafe, float(act_res.info['p_unsafe']))
            if act_res.info.get('safety_override', None) is not None:
                any_safety_override = True

        # -------------------------
        # 최종 액션 구성
        # -------------------------
        coords_final = translation_results[-1]  # (B,3) tensor

        if len(rot_grip_results) > 0:
            rot_grip_final = rot_grip_results[-1]
        else:
            # 기본값: rot=(0,0,0), grip=open(1)
            rot_grip_final = torch.tensor([[0, 0, 0, 1]], dtype=torch.long, device=device)

        if len(ignore_collisions_results) > 0:
            ignore_final = ignore_collisions_results[-1]
        else:
            # 기본값: ignore_collisions = 0 (무시하지 않음)
            ignore_final = torch.tensor([[0]], dtype=torch.long, device=device)

        # 디버깅용 인덱스 로그(전 레이어 concat)
        trans_cat = torch.cat(translation_results, dim=1)[0].detach().cpu().numpy()
        observation_elements['trans_action_indicies'] = trans_cat
        observation_elements['rot_grip_action_indicies'] = rot_grip_final[0].detach().cpu().numpy()

        # 안전 메타 집계 결과도 info에 포함
        infos['p_unsafe_max_along_depth'] = max_p_unsafe
        if any_safety_override:
            infos['safety_override_any'] = True

        rlbench_action = self._build_rlbench_action(
            observation.get('attention_coordinate', None),
            rot_grip_final,
            ignore_final
        )

        return ActResult(
            rlbench_action,
            observation_elements=observation_elements,
            info=infos,
        )

    def _build_rlbench_action(self, attention_coordinate, rot_grip, ignore):
        if attention_coordinate is None:
            raise RuntimeError("attention_coordinate missing for action decoding")
        pos = attention_coordinate[0].detach().cpu().numpy().astype(np.float32)
        rot_idx = rot_grip[0, :3].detach().cpu().numpy().astype(np.int32)
        quat = utils.discrete_euler_to_quaternion(rot_idx, self._rotation_resolution)
        grip_val = float(rot_grip[0, 3].detach().cpu().item())
        ignore_val = float(ignore[0, 0].detach().cpu().item())
        arm = np.concatenate([pos, quat], axis=0)
        return np.concatenate([arm, [grip_val, ignore_val]], axis=0).astype(np.float32)

    # --------------------------
    # Summaries / Weights IO
    # --------------------------
    def update_summaries(self) -> List[Summary]:
        summaries: List[Summary] = []
        for qa in self._qattention_agents:
            summaries.extend(qa.update_summaries())
        return summaries

    def act_summaries(self) -> List[Summary]:
        out: List[Summary] = []
        for qa in self._qattention_agents:
            out.extend(qa.act_summaries())
        return out

    def load_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.load_weights(savedir)

    def save_weights(self, savedir: str):
        for qa in self._qattention_agents:
            qa.save_weights(savedir)
