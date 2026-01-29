import gc
import logging
import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from typing import List

import hydra
import numpy as np
import torch
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
from rlbench import CameraConfig, ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning, EndEffectorPoseViaIK
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import task as rlbench_task
from rlbench.backend.utils import task_file_to_task_class
from yarr.runners.independent_env_runner import IndependentEnvRunner
from yarr.utils.stat_accumulator import SimpleAccumulator

from agents import c2farm_lingunet_bc
from agents import peract_bc
from agents import peract_rl
from agents import arm
from agents.baselines import bc_lang, vit_bc_lang

from helpers.custom_rlbench_env import CustomRLBenchEnv, CustomMultiTaskRLBenchEnv
from helpers import utils
from helpers.safety_utils import SafetyState, get_scene_safety_distance, compute_safety_now, compute_cost

from yarr.utils.rollout_generator import RolloutGenerator
from torch.multiprocessing import Process, Manager
from yarr.utils.log_writer import LogWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class _EndEffectorPoseViaIKWithIgnore(EndEffectorPoseViaIK):
    def action(self, scene, action, ignore_collisions: bool = True):
        return super().action(scene, action)


def eval_seed(train_cfg,
              eval_cfg,
              logdir,
              cams,
              env_device,
              multi_task,
              seed,
              env_config) -> None:

    tasks = eval_cfg.rlbench.tasks
    rg = RolloutGenerator()

    if train_cfg.method.name == 'ARM':
        raise NotImplementedError('ARM not yet supported for eval.py')

    elif train_cfg.method.name == 'BC_LANG':
        agent = bc_lang.launch_utils.create_agent(
            cams[0],
            train_cfg.method.activation,
            train_cfg.method.lr,
            train_cfg.method.weight_decay,
            train_cfg.rlbench.camera_resolution,
            train_cfg.method.grad_clip)

    elif train_cfg.method.name == 'VIT_BC_LANG':
        agent = vit_bc_lang.launch_utils.create_agent(
            cams[0],
            train_cfg.method.activation,
            train_cfg.method.lr,
            train_cfg.method.weight_decay,
            train_cfg.rlbench.camera_resolution,
            train_cfg.method.grad_clip)

    elif train_cfg.method.name == 'C2FARM_LINGUNET_BC':
        agent = c2farm_lingunet_bc.launch_utils.create_agent(train_cfg)

    elif train_cfg.method.name in ('PERACT_BC', 'SAFECHUNK'):
        agent = peract_bc.launch_utils.create_agent(train_cfg)

    elif train_cfg.method.name == 'LAGPPO':
        agent = peract_rl.launch_utils.create_agent(train_cfg)

    else:
        raise ValueError('Method %s does not exists.' % train_cfg.method.name)

    if train_cfg.method.name == 'LAGPPO':
        weightsdir = os.path.join(logdir, 'weights')
        if not os.path.exists(weightsdir):
            raise Exception('No weights directory found.')

        if eval_cfg.framework.eval_type == 'missing':
            weight_folders = sorted(map(int, os.listdir(weightsdir)))
        elif eval_cfg.framework.eval_type == 'last':
            weight_folders = sorted(map(int, os.listdir(weightsdir)))
            weight_folders = [weight_folders[-1]]
        elif isinstance(eval_cfg.framework.eval_type, int):
            weight_folders = [int(eval_cfg.framework.eval_type)]
        else:
            weight_folders = sorted(map(int, os.listdir(weightsdir)))

        writer = None
        if eval_cfg.framework.eval_save_metrics:
            writer = LogWriter(logdir, True, True, env_csv='eval_data.csv')

        task_classes = []
        for t in eval_cfg.rlbench.tasks:
            task_classes.append(task_file_to_task_class(t))

        agent.build(training=False, device=env_device)
        for weight in weight_folders:
            agent.load_weights(os.path.join(weightsdir, str(weight)))

            results = {}
            for task, task_class in zip(eval_cfg.rlbench.tasks, task_classes):
                env = CustomRLBenchEnv(
                    task_class=task_class,
                    observation_config=env_config[1],
                    action_mode=env_config[2],
                    dataset_root=eval_cfg.rlbench.demo_path,
                    episode_length=eval_cfg.rlbench.episode_length,
                    headless=eval_cfg.rlbench.headless,
                    include_lang_goal_in_obs=eval_cfg.rlbench.include_lang_goal_in_obs,
                    time_in_state=eval_cfg.rlbench.time_in_state,
                    record_every_n=eval_cfg.framework.record_every_n,
                )
                env.eval = True
                env.launch()

                successes = []
                returns = []
                costs = []
                near_miss = []
                for ep in range(eval_cfg.framework.eval_episodes):
                    eval_demo_seed = ep + eval_cfg.framework.eval_from_eps_number
                    obs = env.reset_to_demo(eval_demo_seed)
                    safety_state = SafetyState()
                    obs = dict(obs)
                    obs = obs

                    episode_return = 0.0
                    episode_cost = 0.0
                    episode_near = 0
                    episode_steps = 0
                    done = False
                    while not done and episode_steps < eval_cfg.rlbench.episode_length:
                        d_now = get_scene_safety_distance(env)
                        safety_now, safety_state = compute_safety_now(
                            d_now,
                            safety_state,
                            train_cfg.method.cost.d_safe,
                            train_cfg.method.cost.ttc_max,
                        )
                        obs['safety_now'] = safety_now

                        act_res = agent.act(episode_steps, {k: torch.tensor([v]) for k, v in obs.items()}, deterministic=True)
                        transition = env.step(act_res)
                        reward = transition.reward
                        cost = compute_cost(
                            safety_now,
                            train_cfg.method.cost.d_safe,
                            train_cfg.method.cost.ttc_safe,
                            train_cfg.method.cost.rel_vel_safe,
                            train_cfg.method.cost.collision_distance,
                            train_cfg.method.cost.w_dist,
                            train_cfg.method.cost.w_ttc,
                            train_cfg.method.cost.w_near,
                            train_cfg.method.cost.w_rel_vel,
                            train_cfg.method.cost.w_collision,
                        )
                        episode_return += reward
                        episode_cost += cost
                        episode_near += int(safety_now[3] > 0.5)
                        episode_steps += 1
                        obs = dict(transition.observation)
                        done = transition.terminal

                    successes.append(1.0 if episode_return > 0 else 0.0)
                    returns.append(episode_return)
                    costs.append(episode_cost)
                    near_miss.append(episode_near / max(1, episode_steps))
                env.shutdown()

                results[task] = {
                    'return': float(np.mean(returns)),
                    'success_rate': float(np.mean(successes)),
                    'mean_cost': float(np.mean(costs)),
                    'near_miss_rate': float(np.mean(near_miss)),
                }

            if writer is not None:
                for task, stats in results.items():
                    writer.add_scalar(weight, f'eval_envs/return/{task}', stats['return'])
                    writer.add_scalar(weight, f'eval_envs/success_rate/{task}', stats['success_rate'])
                    writer.add_scalar(weight, f'eval_envs/mean_cost/{task}', stats['mean_cost'])
                    writer.add_scalar(weight, f'eval_envs/near_miss_rate/{task}', stats['near_miss_rate'])
                writer.end_iteration()

        if writer is not None:
            writer.close()
        return

    stat_accum = SimpleAccumulator(eval_video_fps=30)

    cwd = os.getcwd()
    weightsdir = os.path.join(logdir, 'weights')

    env_runner = IndependentEnvRunner(
        train_env=None,
        agent=agent,
        train_replay_buffer=None,
        num_train_envs=0,
        num_eval_envs=eval_cfg.framework.eval_envs,
        rollout_episodes=99999,
        eval_episodes=eval_cfg.framework.eval_episodes,
        training_iterations=train_cfg.framework.training_iterations,
        eval_from_eps_number=eval_cfg.framework.eval_from_eps_number,
        episode_length=eval_cfg.rlbench.episode_length,
        stat_accumulator=stat_accum,
        weightsdir=weightsdir,
        logdir=logdir,
        env_device=env_device,
        rollout_generator=rg,
        num_eval_runs=len(tasks),
        multi_task=multi_task)

    manager = Manager()
    save_load_lock = manager.Lock()
    writer_lock = manager.Lock()

    # evaluate all checkpoints (0, 1000, ...) which don't have results, i.e. validation phase
    if eval_cfg.framework.eval_type == 'missing':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))

        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            evaluated_weights = sorted(map(int, list(env_dict['step'].values())))
            weight_folders = [w for w in weight_folders if w not in evaluated_weights]

        print('Missing weights: ', weight_folders)

    # pick the best checkpoint from validation and evaluate, i.e. test phase
    elif eval_cfg.framework.eval_type == 'best':
        env_data_csv_file = os.path.join(logdir, 'eval_data.csv')
        if os.path.exists(env_data_csv_file):
            env_dict = pd.read_csv(env_data_csv_file).to_dict()
            existing_weights = list(map(int, sorted(os.listdir(os.path.join(logdir, 'weights')))))
            task_weights = {}
            for task in tasks:
                weights = list(env_dict['step'].values())

                if len(tasks) > 1:
                    task_score = list(env_dict['eval_envs/return/%s' % task].values())
                else:
                    task_score = list(env_dict['eval_envs/return'].values())

                avail_weights, avail_task_scores = [], []
                for step_idx, step in enumerate(weights):
                    if step in existing_weights:
                        avail_weights.append(step)
                        avail_task_scores.append(task_score[step_idx])

                assert(len(avail_weights) == len(avail_task_scores))
                best_weight = avail_weights[np.argwhere(avail_task_scores == np.amax(avail_task_scores)).flatten().tolist()[-1]]
                task_weights[task] = best_weight

            weight_folders = [task_weights]
            print("Best weights:", weight_folders)
        else:
            raise Exception('No existing eval_data.csv file found in %s' % logdir)

    # evaluate only the last checkpoint
    elif eval_cfg.framework.eval_type == 'last':
        weight_folders = os.listdir(weightsdir)
        weight_folders = sorted(map(int, weight_folders))
        weight_folders = [weight_folders[-1]]
        print("Last weight:", weight_folders)

    # evaluate a specific checkpoint
    elif type(eval_cfg.framework.eval_type) == int:
        weight_folders = [int(eval_cfg.framework.eval_type)]
        print("Weight:", weight_folders)

    else:
        raise Exception('Unknown eval type')

    num_weights_to_eval = np.arange(len(weight_folders))
    if len(num_weights_to_eval) == 0:
        logging.info("No weights to evaluate. Results are already available in eval_data.csv")
        sys.exit(0)

    # evaluate several checkpoints in parallel
    # NOTE: in multi-task settings, each task is evaluated serially, which makes everything slow!
    split_n = utils.split_list(num_weights_to_eval, eval_cfg.framework.eval_envs)
    for split in split_n:
        processes = []
        for e_idx, weight_idx in enumerate(split):
            weight = weight_folders[weight_idx]
            p = Process(target=env_runner.start,
                        args=(weight,
                              save_load_lock,
                              writer_lock,
                              env_config,
                              e_idx % torch.cuda.device_count(),
                              eval_cfg.framework.eval_save_metrics,
                              eval_cfg.cinematic_recorder))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    del env_runner
    del agent
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(config_name='eval', config_path='conf')
def main(eval_cfg: DictConfig) -> None:
    logging.info('\n' + OmegaConf.to_yaml(eval_cfg))

    start_seed = eval_cfg.framework.start_seed
    logdir = os.path.join(eval_cfg.framework.logdir,
                                eval_cfg.rlbench.task_name,
                                eval_cfg.method.name,
                                'seed%d' % start_seed)

    train_config_path = os.path.join(logdir, 'config.yaml')
    if os.path.exists(train_config_path):
        with open(train_config_path, 'r') as f:
            train_cfg = OmegaConf.load(f)
    else:
        raise Exception("Missing seed%d/config.yaml" % start_seed)

    env_device = utils.get_device(eval_cfg.framework.gpu)
    logging.info('Using env device %s.' % str(env_device))

    gripper_mode = Discrete()
    action_mode_name = 'planning'
    if train_cfg.method.name == 'LAGPPO':
        action_mode_name = str(getattr(train_cfg.method, 'action_mode', 'planning')).lower()
    if action_mode_name == 'ik':
        ik_collision = bool(getattr(train_cfg.method, 'ik_collision_checking', False))
        arm_action_mode = _EndEffectorPoseViaIKWithIgnore(collision_checking=ik_collision)
    elif action_mode_name == 'planning':
        arm_action_mode = EndEffectorPoseViaPlanning()
    else:
        raise ValueError(f'Unknown action_mode "{action_mode_name}" for eval.')
    action_mode = MoveArmThenGripper(arm_action_mode, gripper_mode)

    task_files = [t.replace('.py', '') for t in os.listdir(rlbench_task.TASKS_PATH)
                  if t != '__init__.py' and t.endswith('.py')]
    eval_cfg.rlbench.cameras = eval_cfg.rlbench.cameras if isinstance(
        eval_cfg.rlbench.cameras, ListConfig) else [eval_cfg.rlbench.cameras]
    obs_config = utils.create_obs_config(eval_cfg.rlbench.cameras,
                                         eval_cfg.rlbench.camera_resolution,
                                         train_cfg.method.name)

    if eval_cfg.cinematic_recorder.enabled:
        obs_config.record_gripper_closing = True

    # single-task or multi-task
    if len(eval_cfg.rlbench.tasks) > 1:
        tasks = eval_cfg.rlbench.tasks
        multi_task = True

        task_classes = []
        for task in tasks:
            if task not in task_files:
                raise ValueError('Task %s not recognised!.' % task)
            task_classes.append(task_file_to_task_class(task))

        env_config = (task_classes,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      eval_cfg.framework.eval_episodes,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)
    else:
        task = eval_cfg.rlbench.tasks[0]
        multi_task = False

        if task not in task_files:
            raise ValueError('Task %s not recognised!.' % task)
        task_class = task_file_to_task_class(task)

        env_config = (task_class,
                      obs_config,
                      action_mode,
                      eval_cfg.rlbench.demo_path,
                      eval_cfg.rlbench.episode_length,
                      eval_cfg.rlbench.headless,
                      train_cfg.rlbench.include_lang_goal_in_obs,
                      eval_cfg.rlbench.time_in_state,
                      eval_cfg.framework.record_every_n)

    logging.info('Evaluating seed %d.' % start_seed)
    eval_seed(train_cfg,
              eval_cfg,
              logdir,
              eval_cfg.rlbench.cameras,
              env_device,
              multi_task, start_seed,
              env_config)

if __name__ == '__main__':
    main()
