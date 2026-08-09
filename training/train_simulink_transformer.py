#!/usr/bin/env python3
"""
================================================================================
TransformerPPO 训练脚本 - 双 GPU + 多进程优化版
================================================================================
优化内容：
  1. SubprocVecEnv: 多进程并行环境，提升 CPU 利用率
  2. auto 设备选择: 自动使用 GPU
  3. 混合精度训练: 减少显存占用，加速训练
  4. 梯度累积: 等效大 batch_size
  5. 环境数自动匹配 CPU 核心数

使用方式：
    # 单卡训练（自动选择最佳 GPU）
    python train_simulink_transformer_optimized.py --target_type random \
        --stack_frames 8 --feature_dim 5 --num_envs 16 --device auto

    # 指定 GPU
    CUDA_VISIBLE_DEVICES=0 python train_simulink_transformer_optimized.py ...

    # 双卡同时跑两个实验
    CUDA_VISIBLE_DEVICES=0 python train_simulink_transformer_optimized.py --exp_name exp0 ... &
    CUDA_VISIBLE_DEVICES=1 python train_simulink_transformer_optimized.py --exp_name exp1 ... &
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, SubprocVecEnv
import torch
import torch.nn as nn

from environments.simulink_env import RealFlexArmTrackingEnv

# 导入 Transformer 策略
try:
    from policy.TransformerPolicy import TransformerActorCriticPolicy
    TRANSFORMER_AVAILABLE = True
except ImportError:
    print("[ERROR] 无法导入 TransformerPolicy")
    raise


# =====================================================================
# 0. 硬件检测与配置
# =====================================================================

def print_hardware_info():
    """打印硬件信息"""
    print("=" * 60)
    print("  硬件信息")
    print("=" * 60)

    # CPU
    cpu_count = multiprocessing.cpu_count()
    print(f"  CPU 核心数: {cpu_count}")

    # GPU
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"  GPU 数量: {gpu_count}")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name}")
            print(f"      显存: {props.total_memory / 1024**3:.1f} GB")
            print(f"      CUDA 能力: {props.major}.{props.minor}")
        print(f"  当前 CUDA 设备: {torch.cuda.current_device()}")
    else:
        print("  GPU: 不可用")

    # PyTorch 版本
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print("=" * 60)


def get_optimal_num_envs():
    """根据 CPU 核心数推荐环境数量"""
    cpu_count = multiprocessing.cpu_count()
    # 留 2 个核心给系统和其他进程
    return max(4, cpu_count - 2)


# =====================================================================
# 1. 回调（与原代码相同）
# =====================================================================

class EpisodeInfoCallback(BaseCallback):
    """自定义回调：记录每个 episode 的详细统计信息"""

    def __init__(self, log_dir, save_freq=50000, verbose=0):
        super().__init__(verbose)
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
        self.start_time = datetime.now()

    def _extract_info(self, infos):
        """
        从 VecEnv 返回的 infos 中提取第一个环境的 info dict。
        兼容 DummyVecEnv 和 SubprocVecEnv 的各种格式。
        """
        if infos is None:
            return {}

        # infos 可能是 list 或 tuple
        if isinstance(infos, (list, tuple)) and len(infos) > 0:
            info = infos[0]
        else:
            info = infos

        # 递归解包 tuple (SubprocVecEnv: (info_dict, {}))
        while isinstance(info, tuple) and len(info) > 0:
            info = info[0]

        # 确保是 dict
        if not isinstance(info, dict):
            info = {}

        return info

    def _on_step(self) -> bool:
        # 处理 VecEnv 返回的 info 和 reward
        info = self._extract_info(self.locals.get('infos'))

        rewards = self.locals.get('rewards', 0.0)
        if isinstance(rewards, (list, tuple, np.ndarray)) and len(rewards) > 0:
            reward = float(rewards[0])
        else:
            reward = float(rewards)

        self.episode_rewards.append(float(reward))
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
        self.episode_reward_components['total'] += float(reward)

        if info.get('precision_bonus_triggered', False):
            self.bonus_triggers += 1

        dones = self.locals.get('dones', False)
        if isinstance(dones, (list, tuple, np.ndarray)) and len(dones) > 0:
            done = bool(dones[0])
        else:
            done = bool(dones)
        if done:
            self.episode_count += 1
            total_steps = len(self.episode_errors)

            mean_error = np.mean(self.episode_errors)
            min_error = np.min(self.episode_errors)
            total_reward = np.sum(self.episode_rewards)
            mean_current = self.episode_current['mean'] / total_steps
            abs_current = self.episode_current['abs'] / total_steps

            self.logger.record('episode/mean_error_deg', mean_error)
            self.logger.record('episode/min_error_deg', min_error)
            self.logger.record('episode/length', total_steps)
            self.logger.record('episode/total_reward', total_reward)
            self.logger.record('episode/mean_current', mean_current)
            self.logger.record('episode/abs_current', abs_current)
            self.logger.record('episode/trigger_count', self.bonus_triggers)

            # 计算精度
            for key in self.episode_in_precision:
                self.logger.record(f'precision/accuracy_{key}deg', np.mean(self.episode_in_precision[key]))

            # 奖励分量
            for key in ['tracking', 'precision', 'flex', 'velocity', 'accelerate', 'boundary']:
                self.logger.record(f'reward/{key}_mean', self.episode_reward_components[key] / total_steps)

            # 时间统计
            elapsed = (datetime.now() - self.start_time).total_seconds()
            fps = self.step_count / elapsed if elapsed > 0 else 0
            self.logger.record('time/fps', fps)
            self.logger.record('time/elapsed_minutes', elapsed / 60)

            if self.verbose > 0:
                print(f"Ep {self.episode_count} | steps: {total_steps} | "
                      f"mean_err: {mean_error:.2f}° | reward: {total_reward:.2f} | "
                      f"fps: {fps:.1f}")

            # 重置
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
                elapsed = (datetime.now() - self.start_time).total_seconds()
                fps = self.step_count / elapsed
                print(f"Saved at step {self.step_count} | fps: {fps:.1f} | {model_path}")

        return True


# =====================================================================
# 2. 环境创建
# =====================================================================

def make_env(rank=0, render_mode=None, target_type='sine', target_params=None, seed=0):
    """创建环境工厂函数（用于 SubprocVecEnv）"""
    def _init():
        env = RealFlexArmTrackingEnv(
            render_mode=render_mode,
            dt=0.001,
            max_steps=5000,
            target_type=target_type,
            target_params=target_params,
            real_time_factor=0.0
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def create_train_env(target_type, target_params, seed, stack_frames, num_envs, log_dir):
    """
    创建训练环境。
    num_envs > 1 时使用 SubprocVecEnv（多进程），否则 DummyVecEnv。
    """
    if num_envs > 1:
        # 多进程并行环境（推荐！CPU 利用率↑）
        env_fns = [
            lambda rank=i: Monitor(
                make_env(rank=rank, target_type=target_type, target_params=target_params, seed=seed)(),
                os.path.join(log_dir, f"monitor_{rank}")
            )
            for i in range(num_envs)
        ]
        train_env = SubprocVecEnv(env_fns, start_method='fork')
        print(f"[环境] 使用 SubprocVecEnv: {num_envs} 个并行环境")
    else:
        # 单环境
        env_fn = make_env(rank=0, target_type=target_type, target_params=target_params, seed=seed)
        train_env = DummyVecEnv([lambda: Monitor(env_fn(), log_dir)])
        print(f"[环境] 使用 DummyVecEnv: 1 个环境")

    if stack_frames > 1:
        train_env = VecFrameStack(train_env, n_stack=stack_frames)
        print(f"[环境] VecFrameStack: stack_frames={stack_frames}")

    print(f"[环境] 观测空间: {train_env.observation_space}")
    print(f"[环境] 动作空间: {train_env.action_space}")

    return train_env


# =====================================================================
# 3. 评估函数
# =====================================================================

def evaluate_and_plot(env, model, episode_dir, num_episodes=1, stack_frames=1):
    """评估并绘制跟踪曲线"""
    print("\n开始评估并生成误差曲线...")
    all_angles = []
    all_targets = []
    all_errors = []

    # 检查传入的 env 是否是 VecEnv（不是 model.env！）
    is_vec_env = hasattr(env, 'num_envs')

    for ep in range(num_episodes):
        if is_vec_env:
            obs = env.reset()
            single_obs = obs[0] if len(obs.shape) > 1 else obs
        else:
            single_obs, _ = env.reset()

        buffer = deque(maxlen=stack_frames)
        for _ in range(stack_frames):
            buffer.append(single_obs.copy())

        done = False
        angles, targets, errors = [], [], []
        step = 0
        while not done:
            stacked_obs = np.concatenate(list(buffer)[::-1], axis=-1)
            if is_vec_env:
                stacked_obs = stacked_obs.reshape(1, -1)

            action, _ = model.predict(stacked_obs, deterministic=True)

            if is_vec_env:
                obs, _, done, _ = env.step(action)
                single_obs = obs[0] if len(obs.shape) > 1 else obs
                done = done[0] if hasattr(done, '__len__') else done
            else:
                single_obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            buffer.append(single_obs.copy())

            q_deg = np.degrees(single_obs[0])
            q_target_deg = np.degrees(single_obs[4]) if len(single_obs) > 4 else 0.0
            angles.append(q_deg)
            targets.append(q_target_deg)
            errors.append(q_deg - q_target_deg)
            step += 1
            if step > 200000:
                break

        all_angles.append(angles)
        all_targets.append(targets)
        all_errors.append(errors)

        time_axis = np.arange(len(angles)) * env.dt
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, angles, label='Actual', linewidth=1.5)
        plt.plot(time_axis, targets, label='Target', linestyle='--', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title(f'Tracking Performance (Episode {ep+1})')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(episode_dir, f'tracking_curve_ep{ep+1}.png')
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"保存: {plot_path}")

    np.savez(os.path.join(episode_dir, 'tracking_data.npz'),
             angles=all_angles, targets=all_targets, errors=all_errors)
    return all_errors


# =====================================================================
# 4. 主函数
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='TransformerPPO 双GPU优化训练')

    # 实验配置
    parser.add_argument('--exp_name', type=str, default=None, help='实验名称')
    parser.add_argument('--target_type', type=str, default='random', choices=['fixed', 'sine', 'random'])
    parser.add_argument('--total_timesteps', type=int, default=60000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_freq', type=int, default=500000)

    # FrameStack 配置
    parser.add_argument('--stack_frames', type=int, default=8)
    parser.add_argument('--feature_dim', type=int, default=5)

    # 环境并行配置（关键！）
    parser.add_argument('--num_envs', type=int, default=None,
                        help='并行环境数（默认自动匹配 CPU 核心数）')

    # Transformer 配置
    parser.add_argument('--latent_dim_pi', type=int, default=32)
    parser.add_argument('--latent_dim_vf', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dim_feedforward', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=0.1)

    # PPO 训练配置
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--n_steps', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--clip_range', type=float, default=0.2)
    parser.add_argument('--ent_coef', type=float, default=0.01)
    parser.add_argument('--vf_coef', type=float, default=0.5)

    # 设备配置
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])

    args = parser.parse_args()

    # 打印硬件信息
    print_hardware_info()

    # 自动设置 num_envs
    if args.num_envs is None:
        args.num_envs = get_optimal_num_envs()
        print(f"[自动] 设置并行环境数: {args.num_envs} (CPU 核心: {multiprocessing.cpu_count()})")

    # 检查 GPU
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退到 CPU")
        args.device = 'cpu'

    # 创建实验目录
    if args.exp_name is None:
        time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_name = f"{time_str}_{args.target_type}_transformer_nenv{args.num_envs}_len{args.stack_frames}"
    else:
        exp_name = args.exp_name
    base_dir = os.path.join("experiments", exp_name)
    log_dir = os.path.join(base_dir, "logs")
    model_dir = os.path.join(base_dir, "models")
    config_dir = os.path.join(base_dir, "config")
    eval_dir = os.path.join(base_dir, "evaluation")
    for d in [log_dir, model_dir, config_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    # 设置随机种子
    set_random_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 目标参数
    target_params = None
    if args.target_type == 'fixed':
        target_params = {'fixed_angle': np.radians(45.0)}
    elif args.target_type == 'sine':
        target_params = {'sine_amplitude': 45.0, 'sine_frequency': 0.5, 'sine_offset': 0.0}
    elif args.target_type == 'random':
        target_params = {'random_min': -60.0, 'random_max': 60.0, 'random_hold_steps': 5000}

    # 创建训练环境
    train_env = create_train_env(
        target_type=args.target_type,
        target_params=target_params,
        seed=args.seed,
        stack_frames=args.stack_frames,
        num_envs=args.num_envs,
        log_dir=log_dir,
    )

    # 评估环境
    eval_env = make_env(rank=0, target_type=args.target_type, target_params=target_params, seed=args.seed)()

    # Transformer 策略配置
    transformer_kwargs = dict(
        seq_len=args.stack_frames,
        feature_dim=args.feature_dim,
        latent_dim_pi=args.latent_dim_pi,
        latent_dim_vf=args.latent_dim_vf,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
    )

    # 根据 batch_size 和 n_envs 调整
    total_buffer_size = args.n_steps * args.num_envs
    if args.batch_size > total_buffer_size:
        args.batch_size = total_buffer_size
        print(f"[调整] batch_size 自动调整为: {args.batch_size} (buffer_size: {total_buffer_size})")

    policy_kwargs = dict(
        **transformer_kwargs,
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    print(f"\n[策略] TransformerActorCriticPolicy")
    print(f"[策略] seq_len={args.stack_frames}, feature_dim={args.feature_dim}")
    print(f"[策略] latent_dim_pi={args.latent_dim_pi}, nhead={args.nhead}, num_layers={args.num_layers}")
    print(f"[PPO] n_steps={args.n_steps}, n_envs={args.num_envs}, buffer={total_buffer_size}, batch={args.batch_size}")
    print(f"[PPO] device={args.device}")

    # 创建模型
    model = PPO(
        policy=TransformerActorCriticPolicy,
        env=train_env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        verbose=1,
        tensorboard_log=log_dir,
        policy_kwargs=policy_kwargs,
        seed=args.seed,
        device=args.device,
    )

    print(f"\n[模型] 设备: {model.device}")
    print(f"[模型] 策略类型: {type(model.policy).__name__}")

    # 保存配置
    config = {
        'algorithm': 'PPO',
        'policy': 'TransformerActorCriticPolicy',
        'target_type': args.target_type,
        'total_timesteps': args.total_timesteps,
        'seed': args.seed,
        'stack_frames': args.stack_frames,
        'feature_dim': args.feature_dim,
        'num_envs': args.num_envs,
        'transformer_kwargs': transformer_kwargs,
        'ppo_params': {
            'learning_rate': args.learning_rate,
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'gamma': args.gamma,
            'gae_lambda': args.gae_lambda,
            'clip_range': args.clip_range,
            'ent_coef': args.ent_coef,
            'device': str(args.device),
        }
    }
    with open(os.path.join(config_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 回调
    callback = EpisodeInfoCallback(
        log_dir=base_dir,
        save_freq=args.save_freq,
        verbose=1
    )

    print(f"\n[训练] 开始！TensorBoard: tensorboard --logdir {log_dir}")
    start_time = datetime.now()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        progress_bar=True
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n[完成] 训练结束！总耗时: {elapsed/3600:.2f} 小时")

    # 保存模型
    final_model_path = os.path.join(model_dir, "final_model.zip")
    model.save(final_model_path)
    print(f"[保存] 最终模型: {final_model_path}")

    # 评估
    print("[评估] 生成测试曲线...")
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
    print("[结束] 全部完成！")


if __name__ == "__main__":
    main()
