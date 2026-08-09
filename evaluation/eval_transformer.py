#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TransformerPPO 模型评估与 MATLAB 同步工具
================================================================================
兼容 TransformerActorCriticPolicy，支持 VecFrameStack 堆叠观测。

功能模式:
    1. evaluate  - 独立评估模式（仿 eval_model.py）
    2. sync      - MATLAB 同步模式（仿 model_test.py）
    3. compare   - 对比模式（多种控制源并排测试）

使用方式:
    # 独立评估
    python eval_transformer.py --mode evaluate \
        --model_path experiments/.../final_model.zip \
        --stack_frames 8 --feature_dim 5 \
        --target_type sine --episodes 5

    # MATLAB 同步
    python eval_transformer.py --mode sync \
        --model_path experiments/.../final_model.zip \
        --stack_frames 8 --feature_dim 5 \
        --action_source model --num_steps 50000

    # 对比多种控制源
    python eval_transformer.py --mode compare \
        --model_path experiments/.../final_model.zip \
        --stack_frames 8 --feature_dim 5

关键参数:
    --stack_frames    : 堆叠帧数，必须与训练时一致
    --feature_dim     : 单帧观测维度（不是总维度！）
    --model_path      : 训练好的 PPO 模型路径
    --target_type     : fixed / sine / random
"""

import os
import sys
import argparse
import json
import csv
import struct
import socket
import time
import math
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any

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

# ========== 导入环境（根据实际文件名调整）==========
try:
    from environments.simulink_3d_env import RealFlexArmTrackingEnv
except ImportError:
    try:
        print("[WARN] 未找到3d环境模块，请确保 simulink_3d_env.py 在路径中")
        from environments.simulink_env import RealFlexArmTrackingEnv
    except ImportError:
        print("[WARN] 未找到环境模块，请确保 simulink_env.py 或 simulink_3d_env.py 在路径中")
        RealFlexArmTrackingEnv = None


# =====================================================================
# 0. 工具函数
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='TransformerPPO 模型评估与 MATLAB 同步工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  评估模式:
    python eval_transformer.py --mode evaluate --model_path model.zip \
        --stack_frames 8 --feature_dim 5 --episodes 5

  同步模式:
    python eval_transformer.py --mode sync --model_path model.zip \
        --stack_frames 8 --feature_dim 5 --action_source model

  对比模式:
    python eval_transformer.py --mode compare --model_path model.zip \
        --stack_frames 8 --feature_dim 5
        """
    )

    # 通用参数
    parser.add_argument('--mode', type=str, default='evaluate',
                        choices=['evaluate', 'sync', 'compare'],
                        help='运行模式')
    parser.add_argument('--model_path', type=str, default='/home/y/vscode_ws/xuan/PPTransformer/experiments/transformer/20260713_125014_random_transformer_nenv30_len8/models/final_model.zip',
                        help='PPO 模型路径 (.zip)')
    parser.add_argument('--stack_frames', type=int, default=8,
                        help='堆叠帧数（必须与训练时一致）')
    parser.add_argument('--feature_dim', type=int, default=5,
                        help='单帧观测维度（不是总维度！）')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='仿真步长')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save_dir', type=str, default='eval_data',
                        help='结果保存目录')
    parser.add_argument('--no_tensorboard', action='store_true',
                        help='禁用 TensorBoard 记录')

    # 评估模式参数
    parser.add_argument('--target_type', type=str, default='random',
                        choices=['fixed', 'sine', 'random'],
                        help='目标轨迹类型')
    parser.add_argument('--target_params', type=str, default=None,
                        help='目标参数 JSON 字符串')
    parser.add_argument('--episodes', type=int, default=1,
                        help='评估 episode 数量')
    parser.add_argument('--max_steps', type=int, default=20000,
                        help='每 episode 最大步数')
    parser.add_argument('--random_hold_steps', type=int, default=1000,
                        help='随机保持步数')
    parser.add_argument('--render', action='store_true',
                        help='是否开启 GUI 渲染')

    # 同步模式参数
    parser.add_argument('--action_source', type=str, default='model',
                        choices=['zero', 'sine', 'matlab', 'model'],
                        help='控制源类型')
    parser.add_argument('--num_steps', type=int, default=50000,
                        help='同步测试步数')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='TensorBoard 记录间隔')
    parser.add_argument('--send_ip', type=str, default='127.0.0.1',
                        help='MATLAB 接收 IP')
    parser.add_argument('--send_port', type=int, default=9998,
                        help='MATLAB 接收端口')
    parser.add_argument('--recv_ip', type=str, default='0.0.0.0',
                        help='Python 接收 IP')
    parser.add_argument('--recv_port', type=int, default=9999,
                        help='Python 接收端口')
    parser.add_argument('--timeout', type=float, default=2.0,
                        help='UDP 超时时间')
    parser.add_argument('--deterministic', action='store_true', default=True,
                        help='模型确定性预测')

    # 对比模式参数
    parser.add_argument('--compare_sources', type=str,
                        default='zero,sine,matlab,model',
                        help='对比的控制源，逗号分隔')

    return parser.parse_args()


def load_target_params(target_type: str, target_params_str: Optional[str]) -> dict:
    """加载并转换目标参数（角度转弧度）"""
    if target_params_str:
        params = json.loads(target_params_str)
        if target_type == 'sine':
            if 'sine_amplitude' in params:
                params['sine_amplitude'] = np.radians(params['sine_amplitude'])
            if 'sine_offset' in params:
                params['sine_offset'] = np.radians(params['sine_offset'])
        elif target_type == 'fixed':
            if 'fixed_angle' in params:
                params['fixed_angle'] = np.radians(params['fixed_angle'])
        return params

    defaults = {
        'fixed': {'fixed_angle': np.radians(45.0)},
        'sine': {
            'sine_amplitude': np.radians(45.0),
            'sine_frequency': 0.5,
            'sine_offset': np.radians(0.0)
        },
        'random': {
            'random_min': -90.0,
            'random_max': 90.0,
            'random_hold_steps': 500
        }
    }
    return defaults.get(target_type, {})


def create_env(target_type='random', target_params=None, dt=0.001,
               max_steps=200000, render_mode='human', seed=None):
    """创建环境实例"""
    if RealFlexArmTrackingEnv is None:
        raise RuntimeError("环境类未加载，请检查 simulink_env.py 路径")

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
# 1. FrameStack 缓冲区管理器
# =====================================================================

class FrameStackBuffer:
    """
    手动管理帧堆叠缓冲区，兼容 VecFrameStack 的拼接逻辑。

    VecFrameStack 的拼接顺序：最新帧在前（最左边）
    例如 stack_frames=4: [obs_t, obs_{t-1}, obs_{t-2}, obs_{t-3}]
    """

    def __init__(self, stack_frames: int, feature_dim: int):
        self.stack_frames = stack_frames
        self.feature_dim = feature_dim
        self.buffer = deque(maxlen=stack_frames)
        self._initialized = False

    def reset(self, obs: np.ndarray):
        """用初始观测填充缓冲区"""
        self.buffer.clear()
        for _ in range(self.stack_frames):
            self.buffer.append(obs.copy())
        self._initialized = True

    def step(self, obs: np.ndarray) -> np.ndarray:
        """添加新观测，返回堆叠后的观测"""
        self.buffer.append(obs.copy())
        # VecFrameStack 顺序：最新在前
        stacked = np.concatenate(list(self.buffer)[::-1], axis=-1)
        return stacked

    def get(self) -> np.ndarray:
        """获取当前堆叠观测（不添加新帧）"""
        return np.concatenate(list(self.buffer)[::-1], axis=-1)

    @property
    def is_initialized(self) -> bool:
        return self._initialized


# =====================================================================
# 2. 评估模式（仿 eval_model.py）
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

        self.vec_env, self.raw_env = create_vec_env_with_stack(
            target_type=self.args.target_type,
            target_params=target_params,
            dt=self.args.dt,
            max_steps=self.args.max_steps,
            seed=self.args.seed,
            stack_frames=self.args.stack_frames,
            feature_dim=self.args.feature_dim,
        )
        print(f"[环境] 创建完成 | stack_frames={self.args.stack_frames}, feature_dim={self.args.feature_dim}")
        print(f"[环境] 观测空间: {self.vec_env.observation_space}")

    def evaluate_episode(self, episode_idx: int, deterministic: bool = True) -> dict:
        """运行单个 episode"""
        obs = self.vec_env.reset()
        done = np.array([False])

        times = []
        angles = []
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

            # 获取状态（从观测中解析最新帧）
            # VecFrameStack 拼接顺序：最新帧在前
            latest_obs = obs[0][:self.args.feature_dim] if len(obs.shape) > 1 else obs[:self.args.feature_dim]
            q = latest_obs[0]  # 假设第一维是角度
            target_angle = latest_obs[4] if len(latest_obs) > 4 else 0.0  # 假设第5维是目标

            times.append(step * self.args.dt)
            angles.append(np.degrees(q))
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
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/target', targets[-1], step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/error', abs(errors[-1]), step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/reward', rewards[-1], step)
                self.writer.add_scalar(f'eval_ep{episode_idx+1}/action', actions[-1], step)

            step += 1
            if step >= self.args.max_steps:
                break

        return {
            'times': np.array(times),
            'angles': np.array(angles),
            'targets': np.array(targets),
            'errors': np.array(errors),
            'rewards': np.array(rewards),
            'actions': np.array(actions),
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
        """绘制结果图表"""
        for i, data in enumerate(episode_data_list):
            time_axis = data['times']

            # 跟踪曲线
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            axes[0].plot(time_axis, data['angles'], label='Actual', linewidth=1.2)
            axes[0].plot(time_axis, data['targets'], label='Target', linestyle='--', linewidth=1.2)
            axes[0].set_ylabel('Angle (deg)')
            axes[0].set_title(f'Episode {i+1} - Tracking Performance')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(time_axis, data['errors'], label='Error', color='red', linewidth=1.0)
            axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            axes[1].set_ylabel('Error (deg)')
            axes[1].set_title('Tracking Error')
            axes[1].grid(True, alpha=0.3)

            axes[2].plot(time_axis, data['actions'], label='Control', color='green', linewidth=1.0)
            axes[2].set_xlabel('Time (s)')
            axes[2].set_ylabel('Control Input')
            axes[2].set_title('Control Action')
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.args.save_dir, f'tracking_ep{i+1}.png'), dpi=150)
            plt.close()

        # 各 episode 对比柱状图
        mean_errors = [np.mean(np.abs(d['errors'])) for d in episode_data_list]
        total_rewards = [np.sum(d['rewards']) for d in episode_data_list]

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].bar(range(1, len(mean_errors)+1), mean_errors, color='steelblue')
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
                'episode', 'step', 'time(s)', 'angle(deg)', 'target(deg)',
                'error(deg)', 'action', 'total_reward', 'reward_tracking',
                'reward_precision', 'reward_flex', 'reward_velocity', 'reward_boundary'
            ])
            for ep_idx, data in enumerate(episode_data_list):
                n = len(data['times'])
                for i in range(n):
                    writer.writerow([
                        ep_idx + 1, i,
                        f"{data['times'][i]:.6f}",
                        f"{data['angles'][i]:.6f}",
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
# 3. MATLAB 同步模式（仿 model_test.py）
# =====================================================================

class MATLABSyncClient:
    """MATLAB UDP 同步客户端，支持 TransformerPPO 模型控制"""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.env = None
        self.writer = None
        self.records = []

        # UDP
        self.send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.matlab_addr = (args.send_ip, args.send_port)
        self.recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.recv_sock.bind((args.recv_ip, args.recv_port))
        self.recv_sock.setblocking(False)
        self.connected = False

        # 堆叠缓冲区
        self.obs_buffer = FrameStackBuffer(args.stack_frames, args.feature_dim)
        self.model_obs = None  # 当前堆叠观测（用于模型预测）

        # CSV
        self.csv_file = None
        self.csv_writer = None
        self.csv_path = None

        # TensorBoard
        if not args.no_tensorboard and SummaryWriter is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(args.save_dir, f"sync_tb_{timestamp}")
            self.writer = SummaryWriter(log_dir=log_dir)
            print(f"[TensorBoard] 日志: {os.path.abspath(log_dir)}")

        print(f"[UDP] 发送: {args.send_ip}:{args.send_port}")
        print(f"[UDP] 接收: {args.recv_ip}:{args.recv_port}")
        print(f"[配置] 堆叠帧数: {args.stack_frames}, 单帧维度: {args.feature_dim}")

    def load_model(self):
        """加载 PPO 模型"""
        if self.args.model_path is None:
            raise ValueError("请指定 --model_path")
        self.model = PPO.load(self.args.model_path)
        print(f"[模型] 加载成功: {self.args.model_path}")

    def create_env(self, target_type=None, target_params=None):
        """创建环境"""
        _type = target_type or self.args.target_type
        _params = target_params or load_target_params(_type, self.args.target_params)
        self.env = create_env(
            target_type=_type,
            target_params=_params,
            dt=self.args.dt,
            max_steps=self.args.max_steps,
        )
        print(f"[环境] 创建完成: target_type={_type}")

    def _send_current(self, current: float):
        """发送控制量到 MATLAB"""
        packet = struct.pack('<f', float(current))
        self.send_sock.sendto(packet, self.matlab_addr)

    def _recv_state(self, timeout=None):
        """接收 MATLAB 状态"""
        timeout = timeout or self.args.timeout
        self.recv_sock.settimeout(timeout)
        try:
            for _ in range(3):
                data, addr = self.recv_sock.recvfrom(1024)
                if len(data) < 32:
                    return None, data
                q, dq, th, dth = struct.unpack('<4d', data[:32])
                if not all(math.isfinite(x) for x in [q, dq, th, dth]):
                    return None, data
            return np.array([q, dq, th, dth], dtype=np.float64), data
        except socket.timeout:
            return None, None
        except Exception:
            return None, None

    def _handshake(self, timeout=10.0):
        """与 MATLAB 握手"""
        print(f"\n[HANDSHAKE] 发送零控制量到 {self.matlab_addr}")
        self._send_current(0.0)
        start = time.time()
        while time.time() - start < timeout:
            state, raw = self._recv_state(timeout=0.5)
            if state is not None:
                self.connected = True
                print(f"[HANDSHAKE] 成功! MATLAB 已连接! q={np.degrees(state[0]):.4f}°")
                # 同步环境状态
                self.env.q = state[0]
                self.env.dq = state[1]
                self.env.th = state[2]
                self.env.dth = state[3]
                self.env.last_current = 0.0
                self.env.last_dth = state[3]
                return True
            self._send_current(0.0)
            time.sleep(self.args.dt)
        print("[HANDSHAKE] 超时失败")
        return False

    def _init_obs_buffer(self):
        """初始化堆叠观测缓冲区"""
        target = getattr(self.env, 'target_angle', 0.0)
        obs = np.array([
            self.env.q, self.env.dq, self.env.th, self.env.dth, target
        ], dtype=np.float32)
        self.obs_buffer.reset(obs)
        self.model_obs = self.obs_buffer.get()
        print(f"[缓冲区] 已初始化，形状: {self.model_obs.shape}")

    def _compute_action(self, action_source: str, t: float, ml_state: np.ndarray = None) -> float:
        """计算控制量"""
        if action_source == 'zero':
            return 0.0
        elif action_source == 'sine':
            return 0.2 * np.sin(2 * np.pi * 0.5 * t)
        elif action_source == 'model':
            if self.model is None:
                raise ValueError("模型未加载")
            # model.predict 需要 [batch, obs_dim] 形状
            obs_input = self.model_obs.reshape(1, -1)
            action, _ = self.model.predict(obs_input, deterministic=self.args.deterministic)
            return float(action[0]) if hasattr(action, '__len__') else float(action)
        elif action_source == 'matlab':
            # PD 控制
            q = ml_state[0] if ml_state is not None else self.env.q
            dq = ml_state[1] if ml_state is not None else self.env.dq
            Kp, Kd = 5.0, 0.5
            return Kp * (0.0 - q) - Kd * dq
        else:
            return 0.0

    def _log_tensorboard(self, step: int, t: float, current: float,
                         py_state: np.ndarray, ml_state: np.ndarray, target: float):
        """记录到 TensorBoard"""
        if self.writer is None:
            return

        py_q, py_dq, py_th, py_dth = py_state
        ml_q, ml_dq, ml_th, ml_dth = ml_state

        self.writer.add_scalars('sync/angle_q', {
            'python': np.degrees(py_q),
            'matlab': np.degrees(ml_q),
            'target': np.degrees(target),
            'error': np.degrees(py_q - ml_q)
        }, step)

        self.writer.add_scalars('sync/angle_theta', {
            'python': np.degrees(py_th),
            'matlab': np.degrees(ml_th),
            'error': np.degrees(py_th - ml_th)
        }, step)

        self.writer.add_scalars('sync/velocity', {
            'py_dq': np.degrees(py_dq),
            'ml_dq': np.degrees(ml_dq),
            'py_dth': np.degrees(py_dth),
            'ml_dth': np.degrees(ml_dth)
        }, step)

        self.writer.add_scalar('sync/control', current, step)
        self.writer.add_scalars('sync/tracking_error', {
            'python': np.degrees(py_q - target),
            'matlab': np.degrees(ml_q - target)
        }, step)

    def _init_csv(self):
        """初始化 CSV 记录"""
        os.makedirs(self.args.save_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.args.save_dir, f'sync_{timestamp}.csv')
        self.csv_file = open(self.csv_path, 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'step', 't', 'current',
            'py_q', 'py_dq', 'py_th', 'py_dth',
            'ml_q', 'ml_dq', 'ml_th', 'ml_dth',
            'target'
        ])
        print(f"[CSV] 实时记录: {self.csv_path}")

    def _write_csv(self, record: dict):
        """写入 CSV 行"""
        if self.csv_writer is None:
            return
        self.csv_writer.writerow([
            record['step'], f"{record['t']:.6f}", f"{record['current']:.6f}",
            f"{record['py_q']:.6f}", f"{record['py_dq']:.6f}",
            f"{record['py_th']:.6f}", f"{record['py_dth']:.6f}",
            f"{record['ml_q']:.6f}", f"{record['ml_dq']:.6f}",
            f"{record['ml_th']:.6f}", f"{record['ml_dth']:.6f}",
            f"{record['target']:.6f}",
        ])

    def run_sync(self, action_source: str = None, num_steps: int = None):
        """运行同步测试"""
        _source = action_source or self.args.action_source
        _steps = num_steps or self.args.num_steps

        sep = "=" * 60
        print(f"\n{sep}")
        print(f"  MATLAB 同步测试 | 控制源: {_source} | 步数: {_steps}")
        print(sep)

        if _source == 'model' and self.model is None:
            raise ValueError("模型未加载，请先调用 load_model()")

        if not self._handshake(timeout=10.0):
            return []

        self._init_csv()
        self.recv_sock.setblocking(True)
        self._init_obs_buffer()

        fail_count = 0
        max_fail = 50

        for i in range(_steps):
            t = i * self.args.dt

            # 先接收 MATLAB 状态（用于 matlab 控制源）
            self._send_current(0.0)  # 先发送一个占位值
            ml_state, raw = self._recv_state()

            if ml_state is None:
                fail_count += 1
                if fail_count >= max_fail:
                    print("[ERROR] 通信失败次数过多，停止")
                    break
                continue
            fail_count = 0

            # 计算控制量（此时 ml_state 已可用）
            current = self._compute_action(_source, t, ml_state)

            # 重新发送实际控制量
            self._send_current(current)

            # Python 环境步进
            action_array = np.array([current], dtype=np.float32)
            py_obs, _, terminated, truncated, _ = self.env.step(action_array)
            py_state = py_obs[:4]
            target = py_obs[4] if len(py_obs) > 4 else 0.0

            # 更新堆叠缓冲区（使用 MATLAB 状态）
            ml_obs = np.array([
                ml_state[0], ml_state[1], ml_state[2], ml_state[3], target
            ], dtype=np.float32)
            self.model_obs = self.obs_buffer.step(ml_obs)

            # 记录
            record = {
                'step': i, 't': t, 'current': current,
                'py_q': py_state[0], 'py_dq': py_state[1],
                'py_th': py_state[2], 'py_dth': py_state[3],
                'ml_q': ml_state[0], 'ml_dq': ml_state[1],
                'ml_th': ml_state[2], 'ml_dth': ml_state[3],
                'target': float(target)
            }
            self.records.append(record)
            self._write_csv(record)

            # TensorBoard
            if i % self.args.log_interval == 0:
                self._log_tensorboard(i, t, current, py_state, ml_state, target)
                print(f"  Step {i:5d} | I={current:+.4f}A | "
                      f"py_q={np.degrees(py_state[0]):.2f}° | "
                      f"ml_q={np.degrees(ml_state[0]):.2f}° | "
                      f"target={np.degrees(target):.2f}°")

            # 发散检测
            if not all(np.isfinite(py_state)) or not all(np.isfinite(ml_state)):
                print(f"[ERROR] 模型发散于第{i}步!")
                break

            if terminated or truncated:
                print(f"[INFO] Episode 终止于第{i}步")
                break

        self._close_csv()
        self._log_final_stats()

        print(f"\n{sep}")
        print("  同步测试完成")
        print(f"  总步数: {len(self.records)}")
        print(f"  CSV: {self.csv_path}")
        print(sep)

        return self.records

    def _close_csv(self):
        if self.csv_file is not None:
            self.csv_file.close()
            print(f"[CSV] 已保存: {self.csv_path}")
            self.csv_file = None
            self.csv_writer = None

    def _log_final_stats(self):
        """记录最终统计"""
        if not self.records:
            return

        err_q = [abs(np.degrees(r['py_q'] - r['ml_q'])) for r in self.records]
        err_th = [abs(np.degrees(r['py_th'] - r['ml_th'])) for r in self.records]

        print(f"\n[统计]")
        print(f"  q    RMSE: {np.sqrt(np.mean(np.array(err_q)**2)):.4f}°  Max: {max(err_q):.4f}°")
        print(f"  θ    RMSE: {np.sqrt(np.mean(np.array(err_th)**2)):.4f}°  Max: {max(err_th):.4f}°")

        if self.writer is not None:
            self.writer.add_text('final/stats',
                f'q RMSE: {np.sqrt(np.mean(np.array(err_q)**2)):.4f}°\n'
                f'theta RMSE: {np.sqrt(np.mean(np.array(err_th)**2)):.4f}°', 0)

    def close(self):
        self._close_csv()
        if self.writer is not None:
            self.writer.close()
        self.send_sock.close()
        self.recv_sock.close()
        print("[Client] 已关闭")


# =====================================================================
# 4. 对比模式
# =====================================================================

class Comparator:
    """多种控制源对比模式"""

    def __init__(self, args):
        self.args = args
        self.results = {}

    def run(self):
        """运行对比测试"""
        sep = "=" * 60
        print(sep)
        print("  多种控制源对比模式")
        print(sep)

        sources = self.args.compare_sources.split(',')

        for source in sources:
            source = source.strip()
            sub_sep = "-" * 40
            print(f"\n{sub_sep}")
            print(f"  测试控制源: {source}")
            print(sub_sep)

            client = MATLABSyncClient(self.args)

            if source == 'model':
                client.load_model()

            client.create_env()
            records = client.run_sync(action_source=source)
            client.close()

            self.results[source] = records

        # 对比图表
        self._plot_comparison()
        print("\n[完成] 对比测试结束")

    def _plot_comparison(self):
        """绘制对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        colors = {'zero': 'gray', 'sine': 'blue', 'matlab': 'green', 'model': 'red'}

        for source, records in self.results.items():
            if not records:
                continue
            t = [r['t'] for r in records]
            q = [np.degrees(r['ml_q']) for r in records]
            target = [np.degrees(r['target']) for r in records]
            err = [np.degrees(r['ml_q'] - r['target']) for r in records]
            current = [r['current'] for r in records]

            color = colors.get(source, 'black')
            axes[0, 0].plot(t, q, label=source, color=color, alpha=0.8)
            axes[0, 1].plot(t, err, label=source, color=color, alpha=0.8)
            axes[1, 0].plot(t, current, label=source, color=color, alpha=0.8)

        # 目标轨迹只画一次
        if self.results:
            first = list(self.results.values())[0]
            if first:
                t = [r['t'] for r in first]
                target = [np.degrees(r['target']) for r in first]
                axes[0, 0].plot(t, target, 'k--', label='target', linewidth=1.5)

        axes[0, 0].set_ylabel('Angle (deg)')
        axes[0, 0].set_title('Angle Tracking')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].set_ylabel('Error (deg)')
        axes[0, 1].set_title('Tracking Error')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].set_ylabel('Current (A)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_title('Control Input')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # RMSE 柱状图
        rmse_data = {}
        for source, records in self.results.items():
            if records:
                errs = [np.degrees(r['ml_q'] - r['target']) for r in records]
                rmse_data[source] = np.sqrt(np.mean(np.array(errs)**2))

        if rmse_data:
            axes[1, 1].bar(rmse_data.keys(), rmse_data.values(), 
                          color=[colors.get(s, 'black') for s in rmse_data.keys()])
            axes[1, 1].set_ylabel('RMSE (deg)')
            axes[1, 1].set_title('Tracking RMSE Comparison')
            axes[1, 1].grid(axis='y', linestyle='--', alpha=0.5)

        plt.tight_layout()
        save_path = os.path.join(self.args.save_dir, 'comparison.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[图表] 对比图已保存: {save_path}")


# =====================================================================
# 5. 主入口
# =====================================================================

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(args.save_dir,timestamp)
    os.makedirs(args.save_dir, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TransformerPPO 评估工具")
    print(f"  模式: {args.mode}")
    print(f"  堆叠帧数: {args.stack_frames} | 单帧维度: {args.feature_dim}")
    print(sep)
    print()

    if args.mode == 'evaluate':
        evaluator = Evaluator(args)
        evaluator.run()

    elif args.mode == 'sync':
        client = MATLABSyncClient(args)
        if args.action_source == 'model':
            client.load_model()
        client.create_env()
        try:
            client.run_sync()
        finally:
            client.close()

    elif args.mode == 'compare':
        comparator = Comparator(args)
        comparator.run()


if __name__ == "__main__":
    main()
