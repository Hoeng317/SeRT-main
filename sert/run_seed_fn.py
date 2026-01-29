import os
import pickle
import gc
import logging
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from rlbench import CameraConfig, ObservationConfig
from yarr.replay_buffer.wrappers.pytorch_replay_buffer import PyTorchReplayBuffer
from yarr.runners.offline_train_runner import OfflineTrainRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
import torch.distributed as dist

from agents import c2farm_lingunet_bc
from agents import peract_bc
from agents import peract_rl
from agents import arm
from agents.baselines import bc_lang, vit_bc_lang

from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend.utils import task_file_to_task_class
from runners.lagppo_runner import LagPPORunner


class _EndEffectorPoseViaIKWithIgnore(EndEffectorPoseViaIK):
    def action(self, scene, action, ignore_collisions: bool = True):
        return super().action(scene, action)

def run_seed(rank,
             cfg: DictConfig,
             obs_config: ObservationConfig,
             cams,
             multi_task,
             seed,
             world_size) -> None:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    use_dist = not (cfg.method.name == 'LAGPPO' and world_size == 1)
    if use_dist:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    train_device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    task = cfg.rlbench.tasks[0]
    tasks = cfg.rlbench.tasks

    task_folder = task if not multi_task else 'multi'
    replay_path = os.path.join(cfg.replay.path, task_folder, cfg.method.name, f'seed{seed}')

    # ---------- 메서드 분기 ----------
    if cfg.method.name == 'ARM':
        raise NotImplementedError("ARM is not supported yet")

    elif cfg.method.name == 'BC_LANG':
        assert cfg.ddp.num_devices == 1, "BC_LANG only supports single GPU training"
        replay_buffer = bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.rlbench.camera_resolution)

        bc_lang.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams)

        agent = bc_lang.launch_utils.create_agent(
            cams[0], cfg.method.activation, cfg.method.lr,
            cfg.method.weight_decay, cfg.rlbench.camera_resolution,
            cfg.method.grad_clip)

    elif cfg.method.name == 'VIT_BC_LANG':
        assert cfg.ddp.num_devices == 1, "VIT_BC_LANG only supports single GPU training"
        replay_buffer = vit_bc_lang.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.rlbench.camera_resolution)

        vit_bc_lang.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams)

        agent = vit_bc_lang.launch_utils.create_agent(
            cams[0], cfg.method.activation, cfg.method.lr,
            cfg.method.weight_decay, cfg.rlbench.camera_resolution,
            cfg.method.grad_clip)

    elif cfg.method.name == 'C2FARM_LINGUNET_BC':
        replay_buffer = c2farm_lingunet_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)

        c2farm_lingunet_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = c2farm_lingunet_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'PERACT_BC':
        # 기존 PerAct-BC (safety 없이)
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams, cfg.method.voxel_sizes,
            cfg.rlbench.camera_resolution)

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams, cfg.rlbench.scene_bounds,
            cfg.method.voxel_sizes, cfg.method.bounds_offset,
            cfg.method.rotation_resolution, cfg.method.crop_augmentation,
            keypoint_method=cfg.method.keypoint_method)

        agent = peract_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'SAFECHUNK':
        # Safety 확장: 동일한 launch_utils를 재사용(네가 launch_utils에 safety 필드들을 이미 통합)
        replay_buffer = peract_bc.launch_utils.create_replay(
            cfg.replay.batch_size, cfg.replay.timesteps,
            cfg.replay.prioritisation,
            cfg.replay.task_uniform,
            replay_path if cfg.replay.use_disk else None,
            cams,                             # camera names
            cfg.method.voxel_sizes,           # voxel sizes
            cfg.rlbench.camera_resolution,    # image size
            replay_size=cfg.replay.replay_size if 'replay_size' in cfg.replay else 3e5,
            chunk_len=cfg.method.chunk_len,   # <<< K-step chunk
            safety_delta=cfg.method.safety_delta if 'safety_delta' in cfg.method else 0.12
        )

        peract_bc.launch_utils.fill_multi_task_replay(
            cfg, obs_config, rank,
            replay_buffer, tasks, cfg.rlbench.demos,
            cfg.method.demo_augmentation, cfg.method.demo_augmentation_every_n,
            cams,                              # cameras
            cfg.rlbench.scene_bounds,          # bounds
            cfg.method.voxel_sizes,            # voxel sizes
            cfg.method.bounds_offset,          # bounds_offset
            cfg.method.rotation_resolution,    # rot res
            cfg.method.crop_augmentation,      # crop aug
            keypoint_method=cfg.method.keypoint_method
        )

        agent = peract_bc.launch_utils.create_agent(cfg)

    elif cfg.method.name == 'LAGPPO':
        if cfg.ddp.num_devices != 1:
            raise ValueError('LAGPPO currently supports only a single GPU/process.')
        if rank != 0:
            return
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if isinstance(train_device, torch.device) and train_device.type == 'cuda':
                torch.cuda.set_device(train_device)
                mem_frac = float(getattr(cfg.method, 'cuda_memory_fraction', 0.99))
                mem_frac = max(0.01, min(0.99, mem_frac))
                try:
                    torch.cuda.set_per_process_memory_fraction(mem_frac, device=train_device)
                except TypeError:
                    torch.cuda.set_per_process_memory_fraction(mem_frac)
        gripper_mode = Discrete()
        action_mode_name = str(getattr(cfg.method, 'action_mode', 'planning')).lower()
        if action_mode_name == 'ik':
            ik_collision = bool(getattr(cfg.method, 'ik_collision_checking', False))
            arm_action_mode = _EndEffectorPoseViaIKWithIgnore(collision_checking=ik_collision)
        elif action_mode_name == 'planning':
            arm_action_mode = EndEffectorPoseViaPlanning()
        else:
            raise ValueError(f'Unknown action_mode "{action_mode_name}" for LAGPPO.')
        action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

        tasks = cfg.rlbench.tasks
        task_classes = [task_file_to_task_class(t) for t in tasks]
        time_in_state = getattr(cfg.rlbench, 'time_in_state', False)
        headless = getattr(cfg.rlbench, 'headless', True)
        include_lang = getattr(cfg.rlbench, 'include_lang_goal_in_obs', True)
        record_every_n = getattr(cfg.framework, 'record_every_n', 20)

        if len(tasks) > 1:
            env = CustomMultiTaskRLBenchEnv(
                task_classes=task_classes,
                observation_config=obs_config,
                action_mode=action_mode,
                dataset_root=cfg.rlbench.demo_path,
                episode_length=cfg.rlbench.episode_length,
                headless=headless,
                swap_task_every=getattr(cfg.rlbench, 'swap_task_every', 1),
                include_lang_goal_in_obs=include_lang,
                time_in_state=time_in_state,
                record_every_n=record_every_n,
                allow_invalid_actions=bool(getattr(cfg.method, 'allow_invalid_actions', False)),
            )
        else:
            env = CustomRLBenchEnv(
                task_class=task_classes[0],
                observation_config=obs_config,
                action_mode=action_mode,
                dataset_root=cfg.rlbench.demo_path,
                episode_length=cfg.rlbench.episode_length,
                headless=headless,
                include_lang_goal_in_obs=include_lang,
                time_in_state=time_in_state,
                record_every_n=record_every_n,
                allow_invalid_actions=bool(getattr(cfg.method, 'allow_invalid_actions', False)),
            )
        env.launch()

        agent = peract_rl.launch_utils.create_agent(cfg)
        job_dir = os.getcwd()
        logdir = os.path.join(job_dir, f'seed{seed}')
        weightsdir = os.path.join(logdir, 'weights')

        train_runner = LagPPORunner(
            agent=agent,
            env=env,
            cfg=cfg,
            train_device=train_device,
            logdir=logdir,
            weightsdir=weightsdir,
        )
        train_runner.start()
        env.shutdown()
        return

    else:
        raise ValueError(f'Method {cfg.method.name} does not exist.')

    # ---------- Runner ----------
    wrapped_replay = PyTorchReplayBuffer(replay_buffer, num_workers=cfg.framework.num_workers)
    stat_accum = SimpleAccumulator(eval_video_fps=30)

    # framework.logdir를 그대로 사용
    job_dir = os.getcwd()
    logdir = os.path.join(job_dir, f'seed{seed}')
    weightsdir = os.path.join(logdir, 'weights')

    train_runner = OfflineTrainRunner(
        agent=agent,
        wrapped_replay_buffer=wrapped_replay,
        train_device=train_device,
        stat_accumulator=stat_accum,
        iterations=cfg.framework.training_iterations,
        logdir=logdir,
        logging_level=cfg.framework.logging_level,
        log_freq=cfg.framework.log_freq,
        weightsdir=weightsdir,
        num_weights_to_keep=cfg.framework.num_weights_to_keep,
        save_freq=cfg.framework.save_freq,
        tensorboard_logging=cfg.framework.tensorboard_logging,
        csv_logging=cfg.framework.csv_logging,
        load_existing_weights=cfg.framework.load_existing_weights,
        rank=rank,
        world_size=world_size
    )

    train_runner.start()

    del train_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()
