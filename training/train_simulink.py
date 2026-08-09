#!/usr/bin/env python3
"""
训练程序：柔性关节机械臂跟踪控制（PPO + TensorBoard）
优化版本：CPU 多核并行训练
用法：
    python train_simulink.py --exp_name my_experiment --target_type random --total_timesteps 60000000 --stack_frames 1 --n_envs 8
"""

import os
import sys
import json
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
from collections import deque

# Keep project-local packages importable when this file is run from training/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
import torch

from environments.simulink_env import RealFlexArmTrackingEnv


class EpisodeInfoCallback(BaseCallback):
    """
    自定义回调（未修改，保持原功能）
    """
    def __init__(self, eval_env, log_dir, save_freq=50000, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_dir = log_dir
        self.save_freq = save_freq

        self.episode_rewards = []
        self.episode_errors = []
        self.episode_current = {'mean': 0.0, 'abs': 0.0}
        self.episode_reward_components = {
            'tracking': 0.0, 'precision': 0.0, 'flex': 0.0,
            'velocity': 0.0, 'boundary': 0.0, 'total': 0.0, 'accelerate': 0.0
        }
        self.episode_in_precision = {
            '0.1': [], '0.5': [], '1': [], '5': [], '10': [], '30': []
        }
        self.bonus_triggers = 0
        self.step_count = 0
        self.episode_count = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]

        self.episode_rewards.append(reward)
        error_deg = info.get('error_deg', 0.0)
        current = info.get('current_A', 0.0)
        self.episode_errors.append(error_deg)
        self.episode_current['mean'] += current
        self.episode_current['abs'] += abs(current)

        self.episode_reward_components['tracking'] += info.get('reward_tracking', 0.0)
        self.episode_reward_components['precision'] += info.get('reward_precision', 0.0)
        self.episode_in_precision['0.1'].append(info.get('in_precision_zone_0.1', False))
        self.episode_in_precision['0.5'].append(info.get('in_precision_zone_0.5', False))
        self.episode_in_precision['1'].append(info.get('in_precision_zone_1', False))
        self.episode_in_precision['5'].append(info.get('in_precision_zone_5', False))
        self.episode_in_precision['10'].append(info.get('in_precision_zone_10', False))
        self.episode_in_precision['30'].append(info.get('in_precision_zone_30', False))
        self.episode_reward_components['flex'] += info.get('reward_flex', 0.0)
        self.episode_reward_components['velocity'] += info.get('reward_velocity', 0.0)
        self.episode_reward_components['accelerate'] += info.get('reward_accelerate', 0.0)
        self.episode_reward_components['boundary'] += info.get('reward_boundary', 0.0)
        self.episode_reward_components['total'] += reward

        if info.get('precision_bonus_triggered', False):
            self.bonus_triggers += 1

        done = self.locals['dones'][0]
        if done:
            self.episode_count += 1
            total_steps = len(self.episode_errors)

            mean_error = np.mean(self.episode_errors)
            min_error = np.min(self.episode_errors)
            total_reward = np.sum(self.episode_rewards)
            mean_current = self.episode_current['mean'] / total_steps
            abs_current = self.episode_current['abs'] / total_steps
            mean_tracking = self.episode_reward_components['tracking'] / total_steps
            mean_precision = self.episode_reward_components['precision'] / total_steps
            accuracy = {}
            for key in self.episode_in_precision:
                accuracy[key] = np.mean(self.episode_in_precision[key])
            mean_flex = self.episode_reward_components['flex'] / total_steps
            mean_velocity = self.episode_reward_components['velocity'] / total_steps
            mean_accelerate = self.episode_reward_components['accelerate'] / total_steps
            total_boundary = self.episode_reward_components['boundary']

            self.logger.record('episode/mean_error_deg', mean_error)
            self.logger.record('episode/min_error_deg', min_error)
            self.logger.record('episode/length', total_steps)
            self.logger.record('episode/total_reward', total_reward)
            self.logger.record('episode/mean_current', mean_current)
            self.logger.record('episode/abs_current', abs_current)
            self.logger.record('episode/trigger_count', self.bonus_triggers)

            self.logger.record('reward/tracking_mean', mean_tracking)
            self.logger.record('reward/precision_mean', mean_precision)
            self.logger.record('reward/flex_mean', mean_flex)
            self.logger.record('reward/velocity_mean', mean_velocity)
            self.logger.record('reward/accelerate_mean', mean_accelerate)
            self.logger.record('reward/boundary_total', total_boundary)

            self.logger.record('precision/accuracy_0.1deg', accuracy['0.1'])
            self.logger.record('precision/accuracy_0.5deg', accuracy['0.5'])
            self.logger.record('precision/accuracy_1deg', accuracy['1'])
            self.logger.record('precision/accuracy_5deg', accuracy['5'])
            self.logger.record('precision/accuracy_10deg', accuracy['10'])
            self.logger.record('precision/accuracy_30deg', accuracy['30'])

            if self.verbose > 0:
                print(f"Episode {self.episode_count} | steps: {total_steps} | "
                      f"mean_err: {mean_error:.2f}° | min_err: {min_error:.2f}° | "
                      f"triggers: {self.bonus_triggers} | total_reward: {total_reward:.2f}")

            # 重置累积变量
            self.episode_rewards = []
            self.episode_errors = []
            self.episode_current = {'mean': 0.0, 'abs': 0.0}
            for key in self.episode_reward_components:
                self.episode_reward_components[key] = 0.0
            for key in self.episode_in_precision:
                self.episode_in_precision[key] = []
            self.bonus_triggers = 0

        self.step_count += 1
        if self.step_count % self.save_freq == 0:
            model_path = os.path.join(self.log_dir, f"models/ppo_step_{self.step_count}.zip")
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"Model saved at step {self.step_count} -> {model_path}")

        return True


def make_env(render_mode=None, target_type='sine', target_params=None, seed=0):
    def _init():
        env = RealFlexArmTrackingEnv(
            render_mode=render_mode,
            dt=0.001,
            max_steps=5000,
            target_type=target_type,
            target_params=target_params,
            real_time_factor=0.0
        )
        env.reset(seed=seed)
        return env
    return _init


def evaluate_and_plot(env, model, episode_dir, num_episodes=1, stack_frames=1):
    """
    评估并绘制跟踪曲线。
    新增参数 stack_frames：用于手动堆叠历史观测，与训练时保持一致。
    """
    print("\n开始评估并生成误差曲线...")
    all_angles = []
    all_targets = []
    all_errors = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        # 初始化堆叠缓冲区（deque 自动限制长度）
        buffer = deque(maxlen=stack_frames)
        # 用第一个观测填充缓冲区（复制 stack_frames 次）
        for _ in range(stack_frames):
            buffer.append(obs.copy())

        done = False
        angles, targets, errors = [], [], []
        while not done:
            # 手动构造堆叠观测：顺序为 [最新, 次新, ..., 最旧] 与 VecFrameStack 默认一致
            stacked_obs = np.concatenate(list(buffer)[::-1], axis=-1)
            action, _ = model.predict(stacked_obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 更新缓冲区（加入新观测）
            buffer.append(obs.copy())

            # 记录原始角度（从当前观测中提取）
            q_deg = np.degrees(obs[0])
            q_target_deg = np.degrees(obs[4])
            angles.append(q_deg)
            targets.append(q_target_deg)
            errors.append(q_deg - q_target_deg)

        all_angles.append(angles)
        all_targets.append(targets)
        all_errors.append(errors)

        time_axis = np.arange(len(angles)) * env.dt
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, angles, label='Actual angle (deg)', linewidth=1.5)
        plt.plot(time_axis, targets, label='Target angle (deg)', linestyle='--', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title(f'Tracking Performance (Episode {ep+1})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(episode_dir, f'tracking_curve_ep{ep+1}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"保存误差曲线图: {plot_path}")

    np.savez(os.path.join(episode_dir, 'tracking_data.npz'),
             angles=all_angles, targets=all_targets, errors=all_errors)
    return all_errors


def main():
    parser = argparse.ArgumentParser(description='训练柔性关节机械臂跟踪控制')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称，默认为时间戳')
    parser.add_argument('--target_type', type=str, default='random',
                        choices=['fixed', 'sine', 'random'],
                        help='目标轨迹类型')
    parser.add_argument('--total_timesteps', type=int, default=60000000,
                        help='总训练步数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_freq', type=int, default=500000,
                        help='保存模型的频率（步数）')
    parser.add_argument('--stack_frames', type=int, default=8,
                        help='堆叠的历史帧数（>=1）')
    # ========== 新增 CPU 优化参数 ==========
    parser.add_argument('--n_envs', type=int, default=None,
                        help='并行环境数量，默认为 CPU 核心数减 1')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='训练批次大小，建议为 n_envs * n_steps 的约数')
    parser.add_argument('--torch_threads', type=int, default=None,
                        help='PyTorch 使用的 CPU 线程数，默认使用全部核心')
    # =====================================
    args = parser.parse_args()

    # ========== CPU 优化配置 ==========
    cpu_count = multiprocessing.cpu_count()

    # 设置 PyTorch CPU 线程数
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    else:
        torch.set_num_threads(cpu_count)

    # 设置 OpenMP / MKL / OpenBLAS 线程数
    os.environ['OMP_NUM_THREADS'] = str(cpu_count)
    os.environ['MKL_NUM_THREADS'] = str(cpu_count)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_count)

    # 并行环境数：默认 CPU 核心数 - 1
    if args.n_envs is None:
        args.n_envs = max(1, cpu_count - 1)

    print(f"=" * 60)
    print(f"CPU 优化配置")
    print(f"=" * 60)
    print(f"CPU 核心总数: {cpu_count}")
    print(f"并行环境数 (n_envs): {args.n_envs}")
    print(f"PyTorch CPU 线程数: {torch.get_num_threads()}")
    print(f"每个环境 rollout 步数 (n_steps): 2048")
    print(f"总 rollout buffer 大小: {args.n_envs * 2048}")
    print(f"训练批次大小 (batch_size): {args.batch_size}")
    print(f"=" * 60)

    # 创建实验目录
    if args.exp_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{time_str}_{args.target_type}_ppo_stack{args.stack_frames}_envs{args.n_envs}"
    else:
        exp_name = args.exp_name
    base_dir = os.path.join("experiments", exp_name)
    log_dir = os.path.join(base_dir, "logs")
    model_dir = os.path.join(base_dir, "models")
    config_dir = os.path.join(base_dir, "config")
    eval_dir = os.path.join(base_dir, "evaluation")
    for d in [log_dir, model_dir, config_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    set_random_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 目标参数配置
    target_params = None
    if args.target_type == 'fixed':
        target_params = {'fixed_angle': np.radians(45.0)}
    elif args.target_type == 'sine':
        target_params = {
            'sine_amplitude': 45.0,
            'sine_frequency': 0.5,
            'sine_offset': 0.0
        }
    elif args.target_type == 'random':
        target_params = {
            'random_min': -60.0,
            'random_max': 60.0,
            'random_hold_steps': 5000
        }

    # ========== 关键优化：SubprocVecEnv 替代 DummyVecEnv ==========
    print(f"\n创建 {args.n_envs} 个并行训练环境（SubprocVecEnv）...")

    def make_monitored_env(rank):
        def _init():
            env = RealFlexArmTrackingEnv(
                render_mode=None,
                dt=0.001,
                max_steps=5000,
                target_type=args.target_type,
                target_params=target_params,
                real_time_factor=0.0
            )
            env.reset(seed=args.seed + rank)
            env = Monitor(env, os.path.join(log_dir, f"monitor_{rank}"))
            return env
        return _init

    env_fns = [make_monitored_env(i) for i in range(args.n_envs)]
    train_env = SubprocVecEnv(env_fns)

    if args.stack_frames > 1:
        train_env = VecFrameStack(train_env, n_stack=args.stack_frames)
        print(f"训练环境使用帧堆叠: {args.stack_frames}")

    # 评估环境（单进程，不堆叠）
    eval_env = make_env(render_mode=None,
                        target_type=args.target_type,
                        target_params=target_params,
                        seed=args.seed)()

    # PPO 模型（batch_size 使用参数化值）
    policy_kwargs = dict(net_arch=[64, 64])
    model = PPO(
        policy='MlpPolicy',
        env=train_env,
        learning_rate=5e-5,
        n_steps=2048,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device="auto"
    )

    # 保存配置
    config = {
        'algorithm': 'PPO',
        'target_type': args.target_type,
        'target_params': target_params,
        'total_timesteps': args.total_timesteps,
        'seed': args.seed,
        'stack_frames': args.stack_frames,
        'policy_kwargs': policy_kwargs,
        'ppo_params': {
            'learning_rate': 5e-5,
            'n_steps': 2048,
            'batch_size': args.batch_size,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'ent_coef': 0.01
        },
        'cpu_optimization': {
            'cpu_count': cpu_count,
            'torch_threads': torch.get_num_threads(),
            'n_envs': args.n_envs,
            'vec_env_type': 'SubprocVecEnv',
            'batch_size': args.batch_size
        }
    }
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 回调
    callback = EpisodeInfoCallback(
        eval_env=eval_env,
        log_dir=base_dir,
        save_freq=args.save_freq,
        verbose=1
    )

    print(f"\n开始训练，日志保存在: {log_dir}")
    print(f"TensorBoard 命令: tensorboard --logdir {log_dir}")
    print(f"\n预估: 每次 policy update 处理 {args.n_envs * 2048} 个时间步")
    print(f"预估: 总 update 次数约 {args.total_timesteps // (args.n_envs * 2048)} 次")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True
    )

    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"\n训练完成，最终模型已保存: {final_model_path}")

    # 测试与绘图（传入 stack_frames）
    print("生成测试曲线...")
    test_env = RealFlexArmTrackingEnv(
        render_mode=None,
        target_type=args.target_type,
        target_params=target_params,
        max_steps=5000
    )
    evaluate_and_plot(test_env, model, eval_dir, num_episodes=1, stack_frames=args.stack_frames)
    test_env.close()

    train_env.close()
    eval_env.close()
    print("所有文件已保存，程序结束。")


if __name__ == "__main__":
    main()
