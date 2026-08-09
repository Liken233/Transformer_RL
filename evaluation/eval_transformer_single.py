#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TransformerPPO 模型评估工具（独立评估模式）
================================================================================
功能：加载训练好的 PPO 模型，在 Simulink 环境中运行指定 episode，
      绘制 2×2 子图（轨迹跟踪、电机角度、误差、控制量），并输出统计结果。

使用示例：
    python eval_only.py --model_path model.zip \
        --stack_frames 8 --feature_dim 5 \
        --target_type step \
        --target_params '{"final_target": 0.785, "step_time": 2.0}' \
        --episodes 1 --max_steps 10000
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict

# Keep project-local packages importable when this file is run from evaluation/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ========== Stable-Baselines3 ==========
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
except ImportError:
    print("[ERROR] 请安装 stable-baselines3: pip install stable-baselines3")
    raise

# ========== TensorBoard ==========
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboard import SummaryWriter
    except ImportError:
        print("[WARN] 未安装 tensorboard，TensorBoard 记录将禁用")
        SummaryWriter = None

# ========== 导入环境 ==========
try:
    from environments.simulink_3d_env import RealFlexArmTrackingEnv
except ImportError:
    try:
        from environments.simulink_env import RealFlexArmTrackingEnv
    except ImportError:
        print("[ERROR] 未找到环境模块，请确保 simulink_3d_env.py 在路径中")
        sys.exit(1)


# =====================================================================
# 辅助工具
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='TransformerPPO 模型独立评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python eval_only.py --model_path model.zip \\
      --stack_frames 8 --feature_dim 5 --target_type step \\
      --target_params '{"final_target": 0.785, "step_time": 2.0}' \\
      --episodes 1 --max_steps 10000
        """
    )

    # 模型和环境参数
    parser.add_argument('--model_path', type=str, default="/home/y/vscode_ws/xuan/PPTransformer/experiments/transformer/20260713_125014_random_transformer_nenv30_len8/models/final_model.zip",
                        help='PPO 模型路径 (.zip)')
    parser.add_argument('--stack_frames', type=int, default=8,
                        help='堆叠帧数（必须与训练时一致）')
    parser.add_argument('--feature_dim', type=int, default=5,
                        help='单帧观测维度（不是总维度）')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='仿真步长')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='eval_data',
                        help='结果保存目录')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='禁用 TensorBoard 记录')

    # 目标轨迹参数
    parser.add_argument('--target_type', type=str, default='sine',
                        choices=['fixed', 'sine', 'random', 'step'],
                        help='目标轨迹类型')
    parser.add_argument('--target_params', type=str, default=None,
                        help='目标参数 JSON 字符串')

    # 评估参数
    parser.add_argument('--episodes', type=int, default=1,
                        help='评估 episode 数量')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='每 episode 最大步数')
    parser.add_argument('--render', default=True,
                        help='是否开启 GUI 渲染（默认关闭）')

    return parser.parse_args()


def load_target_params(target_type: str, target_params_str: Optional[str]) -> dict:
    """加载并转换目标参数（角度转弧度）"""
    if target_params_str:
        params = json.loads(target_params_str)
        # 角度转弧度（仅对涉及角度的字段）
        if target_type == 'sine':
            if 'sine_amplitude' in params:
                params['sine_amplitude'] = np.radians(params['sine_amplitude'])
            if 'sine_offset' in params:
                params['sine_offset'] = np.radians(params['sine_offset'])
        elif target_type == 'fixed':
            if 'fixed_angle' in params:
                params['fixed_angle'] = np.radians(params['fixed_angle'])
        elif target_type == 'step':
            if 'final_target' in params:
                params['final_target'] = np.radians(params['final_target'])
        return params

    defaults = {
        'fixed': {'fixed_angle': np.radians(45.0)},
        'sine': {
            'sine_amplitude': np.radians(15.0),
            'sine_frequency': 0.5,
            'sine_offset': np.radians(0.0)
        },
        'random': {
            'random_min': -90.0,
            'random_max': 90.0,
            'random_hold_steps': 500
        },
        'step': {
            'final_target': np.radians(45.0),
            'step_time': 0.5,
            'initial_q': 0.0,
            'initial_dq': 0.0,
            'initial_th': 0.0,
            'initial_dth': 0.0,
            'use_initial_state': True
        }
    }
    return defaults.get(target_type, {})


def create_env(target_type='random', target_params=None, dt=0.001,
               max_steps=200000, render_mode='human', seed=None):
    """创建环境实例"""
    env = RealFlexArmTrackingEnv(
        render_mode=render_mode,
        dt=dt,
        max_steps=max_steps,
        target_type=target_type,
        target_params=target_params,
        real_time_factor=0.0 if render_mode is None else 1.0
    )
    if seed is not None:
        env.reset(seed=seed)
    return env


def create_vec_env_with_stack(target_type='random', target_params=None,
                               dt=0.001, max_steps=200000, seed=None,
                               stack_frames=4, feature_dim=5):
    """
    创建带 VecFrameStack 的环境，与训练时一致。
    返回 (vec_env, raw_env) 元组。
    """
    def _make_env():
        env = create_env(target_type, target_params, dt, max_steps, seed=seed)
        return env

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecFrameStack(vec_env, n_stack=stack_frames)

    # 获取原始环境引用
    raw_env = vec_env.envs[0] if hasattr(vec_env, 'envs') else None

    return vec_env, raw_env


# =====================================================================
# 评估器
# =====================================================================

class Evaluator:
    """独立评估模式：加载模型，在环境上运行多个 episode"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.vec_env = None
        self.raw_env = None
        self.writer = None

        os.makedirs(args.save_dir, exist_ok=True)

        if not args.no_tensorboard and SummaryWriter is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(args.save_dir, f"tb_{timestamp}")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[TensorBoard] 日志: {os.path.abspath(log_dir)}")

    def load_model(self):
        """加载 PPO 模型"""
        if self.args.model_path is None:
            raise ValueError("请指定 --model_path")
        if not os.path.isfile(self.args.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.args.model_path}")

        self.model = PPO.load(self.args.model_path)
        print(f"[模型] 加载成功: {self.args.model_path}")
        print(f"[模型] 策略类型: {type(self.model.policy).__name__}")

    def create_env(self):
        """创建评估环境"""
        target_params = load_target_params(self.args.target_type, self.args.target_params)
        render_mode = 'human' if self.args.render else None

        self.vec_env, self.raw_env = create_vec_env_with_stack(
            target_type=self.args.target_type,
            target_params=target_params,
            dt=self.args.dt,
            max_steps=self.args.max_steps,
            seed=self.args.seed,
            stack_frames=self.args.stack_frames,
            feature_dim=self.args.feature_dim,
        )
        # 设置渲染模式
        if self.raw_env is not None:
            self.raw_env.render_mode = render_mode
        print(f"[环境] 创建完成 | stack_frames={self.args.stack_frames}, feature_dim={self.args.feature_dim}")
        print(f"[环境] 观测空间: {self.vec_env.observation_space}")

    def evaluate_episode(self, episode_idx: int, deterministic: bool = True) -> dict:
        """运行单个 episode"""
        obs = self.vec_env.reset()
        done = np.array([False])

        times = []
        angles = []
        motor_angles = []          # 新增：电机侧角度
        targets = []
        errors = []
        rewards = []
        reward_details = {
            'tracking': [], 'precision': [], 'flex': [],
            'velocity': [], 'boundary': [], 'total': []
        }
        actions = []

        step = 0
        while not done[0]:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = self.vec_env.step(action)

            # 从 info[0] 获取原始环境信息（VecEnv 返回 list）
            info_dict = info[0] if isinstance(info, list) else info

            # 获取最新帧观测（堆叠观测的第一块）
            if len(obs.shape) > 1:
                latest_obs = obs[0][:self.args.feature_dim]
            else:
                latest_obs = obs[:self.args.feature_dim]

            q = latest_obs[0]          # 负载角度
            th = latest_obs[2]         # 电机角度
            target_angle = latest_obs[4] if len(latest_obs) > 4 else 0.0

            times.append(step * self.args.dt)
            angles.append(np.degrees(q))
            motor_angles.append(np.degrees(th))
            targets.append(np.degrees(target_angle))
            errors.append(np.degrees(q - target_angle))
            rewards.append(float(reward[0]) if isinstance(reward, np.ndarray) else float(reward))
            actions.append(float(action[0]) if isinstance(action, np.ndarray) else float(action))

            reward_details['tracking'].append(info_dict.get('reward_tracking', 0.0))
            reward_details['precision'].append(info_dict.get('reward_precision', 0.0))
            reward_details['flex'].append(info_dict.get('reward_flex', 0.0))
            reward_details['velocity'].append(info_dict.get('reward_velocity', 0.0))
            reward_details['boundary'].append(info_dict.get('reward_boundary', 0.0))
            reward_details['total'].append(rewards[-1])

            # TensorBoard 记录
            if self.writer is not None and step % 50 == 0:
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/angle', angles[-1], step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/motor_angle', motor_angles[-1], step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/target', targets[-1], step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/error', abs(errors[-1]), step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/reward', rewards[-1], step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/action', actions[-1], step)

            step += 1
            if step >= self.args.max_steps:
                break

        return {
            'times': np.array(times[:-1]),
            'angles': np.array(angles[:-1]),
            'motor_angles': np.array(motor_angles[:-1]),
            'targets': np.array(targets[:-1]),
            'errors': np.array(errors[:-1]),
            'rewards': np.array(rewards[:-1]),
            'actions': np.array(actions[:-1]),
            'reward_details': {k: np.array(v) for k, v in reward_details.items()},
            'steps': step
        }

    def compute_statistics(self, episode_data_list: List[dict]) -> dict:
        """计算统计指标"""
        mean_errors = []
        min_errors = []
        final_errors = []
        total_rewards = []
        mean_tracking = []
        mean_precision = []
        mean_flex = []
        mean_velocity = []
        total_boundary = []

        for data in episode_data_list:
            abs_errors = np.abs(data['errors'])
            mean_errors.append(np.mean(abs_errors))
            min_errors.append(np.min(abs_errors))
            final_errors.append(abs_errors[-1] if len(abs_errors) > 0 else np.nan)
            total_rewards.append(np.sum(data['rewards']))
            mean_tracking.append(np.mean(data['reward_details']['tracking']))
            mean_precision.append(np.mean(data['reward_details']['precision']))
            mean_flex.append(np.mean(data['reward_details']['flex']))
            mean_velocity.append(np.mean(data['reward_details']['velocity']))
            total_boundary.append(np.sum(data['reward_details']['boundary']))

        stats = {
            'mean_error_deg': float(np.mean(mean_errors)),
            'std_error_deg': float(np.std(mean_errors)),
            'min_error_deg': float(np.min(min_errors)),
            'final_error_deg': float(np.mean(final_errors)),
            'total_reward': float(np.mean(total_rewards)),
            'std_total_reward': float(np.std(total_rewards)),
            'mean_tracking_reward': float(np.mean(mean_tracking)),
            'mean_precision_reward': float(np.mean(mean_precision)),
            'mean_flex_penalty': float(np.mean(mean_flex)),
            'mean_velocity_penalty': float(np.mean(mean_velocity)),
            'total_boundary_penalty': float(np.mean(total_boundary)),
        }
        return stats

    def plot_results(self, episode_data_list: List[dict]):
        """绘制 2×2 子图：轨迹跟踪、电机角度、误差、控制量"""
        for i, data in enumerate(episode_data_list):
            time_axis = data['times']
            angles = data['angles']
            motor_angles = data.get('motor_angles', np.zeros_like(angles))  # 兼容旧数据
            targets = data['targets']
            errors = data['errors']
            actions = data['actions']

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            fig.subplots_adjust(hspace=0.3, wspace=0.3)

            # (a) 轨迹跟踪图（左上）
            ax = axes[0, 0]
            ax.plot(time_axis, angles, label='Actual (q)', linewidth=1.2)
            ax.plot(time_axis, targets, label='Target', linestyle='--', linewidth=1.2)
            ax.set_ylabel('Angle (deg)')
            ax.set_title('Tracking Performance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.text(0.5, -0.15, '(a)', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

            # (b) 电机侧角度图（右上）
            ax = axes[0, 1]
            ax.plot(time_axis, motor_angles, label='Motor angle (θ)', color='orange', linewidth=1.2)
            #ax.plot(time_axis, targets, label='Target', linestyle='--', linewidth=1.0, alpha=0.6)
            ax.set_ylabel('Angle (deg)')
            ax.set_title('Motor Side Angle')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.text(0.5, -0.15, '(b)', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

            # (c) 误差图（左下）
            ax = axes[1, 0]
            ax.plot(time_axis, errors, label='Error', color='red', linewidth=1.0)
            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Error (deg)')
            ax.set_title('Tracking Error')
            ax.grid(True, alpha=0.3)
            ax.text(0.5, -0.15, '(c)', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

            # (d) 控制量图（右下）
            ax = axes[1, 1]
            ax.plot(time_axis, actions, label='Control', color='green', linewidth=1.0)
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Control Input')
            ax.set_title('Control Action')
            ax.grid(True, alpha=0.3)
            ax.text(0.5, -0.15, '(d)', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(self.args.save_dir, f'tracking_ep{i+1}.png'), dpi=150)
            plt.close()

        # 各 episode 对比柱状图
        mean_errors = [np.mean(np.abs(d['errors'])) for d in episode_data_list]
        total_rewards = [np.sum(d['rewards']) for d in episode_data_list]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # 修复：使用 np.abs 代替内置 abs
        axes[0].bar(range(1, len(mean_errors)+1), np.abs(mean_errors), color='steelblue')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Mean Absolute Error (deg)')
        axes[0].set_title('Mean Tracking Error per Episode')
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)

        axes[1].bar(range(1, len(total_rewards)+1), total_rewards, color='coral')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Total Reward')
        axes[1].set_title('Total Reward per Episode')
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_dir, 'episode_comparison.png'), dpi=150)
        plt.close()

    def save_csv(self, episode_data_list: List[dict]):
        """保存 episode 数据到 CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = os.path.join(self.args.save_dir, f'eval_{timestamp}.csv')

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'step', 'time(s)', 'angle(deg)', 'motor_angle(deg)',
                'target(deg)', 'error(deg)', 'action', 'total_reward',
                'reward_tracking', 'reward_precision', 'reward_flex',
                'reward_velocity', 'reward_boundary'
            ])
            for ep_idx, data in enumerate(episode_data_list):
                n = len(data['times'])
                for i in range(n):
                    writer.writerow([
                        ep_idx + 1, i,
                        f"{data['times'][i]:.6f}",
                        f"{data['angles'][i]:.6f}",
                        f"{data['motor_angles'][i]:.6f}",
                        f"{data['targets'][i]:.6f}",
                        f"{data['errors'][i]:.6f}",
                        f"{data['actions'][i]:.6f}",
                        f"{data['reward_details']['total'][i]:.6f}",
                        f"{data['reward_details']['tracking'][i]:.6f}",
                        f"{data['reward_details']['precision'][i]:.6f}",
                        f"{data['reward_details']['flex'][i]:.6f}",
                        f"{data['reward_details']['velocity'][i]:.6f}",
                        f"{data['reward_details']['boundary'][i]:.6f}",
                    ])
        print(f"[CSV] 数据已保存: {csv_path}")
        return csv_path

    def run(self):
        """运行评估"""
        sep = "=" * 60
        print(sep)
        print("  TransformerPPO 独立评估模式")
        print(sep)

        self.load_model()
        self.create_env()

        episode_data_list = []
        for ep in range(self.args.episodes):
            print(f"\n[Episode {ep+1}/{self.args.episodes}] 运行中...")
            data = self.evaluate_episode(ep, deterministic=True)
            episode_data_list.append(data)
            print(f"  步数: {data['steps']}, 最终误差: {abs(data['errors'][-1]):.3f}°")
            print(f"  总奖励: {np.sum(data['rewards']):.2f}, 平均误差: {np.mean(np.abs(data['errors'])):.3f}°")

        # 统计
        stats = self.compute_statistics(episode_data_list)
        print(f"\n{sep}")
        print("  评估统计结果")
        print(sep)
        for k, v in stats.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

        # 保存
        stats_path = os.path.join(self.args.save_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"\n[统计] 已保存: {stats_path}")

        self.save_csv(episode_data_list)
        self.plot_results(episode_data_list)
        print(f"[图表] 已保存到: {self.args.save_dir}")

        if self.writer is not None:
            self.writer.close()
        if self.vec_env is not None:
            self.vec_env.close()

        print("\n[完成] 评估结束")


# =====================================================================
# 主入口
# =====================================================================

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(args.save_dir, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TransformerPPO 评估工具（仅评估模式）")
    print(f"  堆叠帧数: {args.stack_frames} | 单帧维度: {args.feature_dim}")
    print(sep)
    print()

    evaluator = Evaluator(args)
    evaluator.run()


if __name__ == "__main__":
    main()
