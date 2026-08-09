#!/usr/bin/env python3
"""
评估脚本：加载训练好的 PPO 模型，测试在 RealFlexArmTrackingEnv 上的表现
用法：
    python eval_model.py --model_path experiments/.../final_model.zip --target_type sine --episodes 5 --stack_frames 4
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from collections import deque
from datetime import datetime

# Keep project-local packages importable when this file is run from evaluation/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environments.simulink_3d_env import RealFlexArmTrackingEnv   # 请根据实际环境文件名调整


def parse_args():
    parser = argparse.ArgumentParser(description='评估柔性关节机械臂跟踪控制模型')
    parser.add_argument('--model_path', type=str,
                        default='/home/y/vscode_ws/xuan/PPTransformer/experiments/20260716_150316_random_ppo_stack16_envs31/models/final_model.zip',
                        help='训练好的模型路径（.zip文件）')
    parser.add_argument('--target_type', type=str, default='random',
                        choices=['fixed', 'sine', 'random'],
                        help='目标轨迹类型')
    parser.add_argument('--target_params', type=str, default=None,
                        help='目标参数 JSON 字符串，例如 \'{"sine_amplitude": 45, "sine_frequency": 0.5}\'')
    parser.add_argument('--episodes', type=int, default=5,
                        help='测试的 episode 数量')
    parser.add_argument('--max_steps', type=int, default=200000,
                        help='每个 episode 最大步数')
    parser.add_argument('--render', action='store_true', default=True,
                        help='是否开启 GUI 渲染')
    parser.add_argument('--save_dir', type=str, default='evaluation_results',
                        help='评估结果保存目录')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--stack_frames', type=int, default=16,
                        help='堆叠的历史帧数（需与训练时一致）')
    return parser.parse_args()


def load_target_params(target_type, target_params_str):
    """加载目标参数（弧度转换）"""
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
    else:
        if target_type == 'fixed':
            return {'fixed_angle': np.radians(45.0)}
        elif target_type == 'sine':
            return {
                'sine_amplitude': np.radians(45.0),
                'sine_frequency': 0.5,
                'sine_offset': np.radians(0.0)
            }
        elif target_type == 'random':
            return {
                'random_min': -90.0,
                'random_max': 90.0,
                'random_hold_steps': 500
            }
        else:
            return None


def evaluate_episode(env, model, stack_frames=1, deterministic=True, dt=0.001):
    """
    运行单个 episode，支持堆叠观测。
    stack_frames：堆叠帧数。
    """
    obs, _ = env.reset()
    # 初始化堆叠缓冲区
    buffer = deque(maxlen=stack_frames)
    for _ in range(stack_frames):
        buffer.append(obs.copy())

    done = False
    times = []
    angles = []
    targets = []
    errors = []
    rewards = []
    reward_details = {
        'tracking': [],
        'precision': [],
        'flex': [],
        'velocity': [],
        'boundary': [],
        'total': []
    }

    step = 0
    while not done:
        # 构造堆叠观测（顺序：最新在前）
        stacked_obs = np.concatenate(list(buffer)[::-1], axis=-1)
        action, _ = model.predict(stacked_obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # 更新缓冲区
        buffer.append(obs.copy())

        q_deg = np.degrees(obs[0])
        q_target_deg = np.degrees(obs[4])
        times.append(step * dt)
        angles.append(q_deg)
        targets.append(q_target_deg)
        errors.append(q_deg - q_target_deg)
        rewards.append(reward)

        reward_details['tracking'].append(info.get('reward_tracking', 0.0))
        reward_details['precision'].append(info.get('reward_precision', 0.0))
        reward_details['flex'].append(info.get('reward_flex', 0.0))
        reward_details['velocity'].append(info.get('reward_velocity', 0.0))
        reward_details['boundary'].append(info.get('reward_boundary', 0.0))
        reward_details['total'].append(reward)

        step += 1

    return {
        'times': np.array(times),
        'angles': np.array(angles),
        'targets': np.array(targets),
        'errors': np.array(errors),
        'rewards': np.array(rewards),
        'reward_details': {k: np.array(v) for k, v in reward_details.items()},
        'steps': len(angles)
    }


def compute_statistics(episode_data_list):
    """计算所有 episode 的统计指标（与原代码相同）"""
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
        'mean_error_deg': np.mean(mean_errors),
        'std_error_deg': np.std(mean_errors),
        'min_error_deg': np.min(min_errors),
        'final_error_deg': np.mean(final_errors),
        'total_reward': np.mean(total_rewards),
        'std_total_reward': np.std(total_rewards),
        'mean_tracking_reward': np.mean(mean_tracking),
        'mean_precision_reward': np.mean(mean_precision),
        'mean_flex_penalty': np.mean(mean_flex),
        'mean_velocity_penalty': np.mean(mean_velocity),
        'total_boundary_penalty': np.mean(total_boundary),
    }
    return stats


def plot_trajectories(episode_data_list, save_dir, dt=0.001):
    """绘制曲线（与原代码相同）"""
    os.makedirs(save_dir, exist_ok=True)
    for i, data in enumerate(episode_data_list):
        time_axis = np.arange(len(data['angles'])) * dt
        plt.figure(figsize=(12, 6))
        plt.plot(time_axis, data['angles'], label='Actual angle (deg)', linewidth=1.5)
        plt.plot(time_axis, data['targets'], label='Target angle (deg)', linestyle='--', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Angle (deg)')
        plt.title(f'Tracking Performance - Episode {i+1}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'tracking_curve_ep{i+1}.png'), dpi=150)
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(time_axis, data['errors'], label='Tracking error (deg)', color='red')
        plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Error (deg)')
        plt.title(f'Tracking Error - Episode {i+1}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'error_curve_ep{i+1}.png'), dpi=150)
        plt.close()

    mean_errors = [np.mean(np.abs(d['errors'])) for d in episode_data_list]
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(mean_errors)+1), mean_errors, color='steelblue')
    plt.xlabel('Episode')
    plt.ylabel('Mean Absolute Error (deg)')
    plt.title('Mean Tracking Error per Episode')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'mean_errors_bar.png'), dpi=150)
    plt.close()


def save_episode_data_to_csv(episode_data_list, csv_path):
    """保存时间戳数据为 CSV（与原代码相同）"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'step', 'time(s)', 'angle(deg)', 'target(deg)',
            'error(deg)', 'total_reward', 'reward_tracking', 'reward_precision',
            'reward_flex', 'reward_velocity', 'reward_boundary'
        ])
        for ep_idx, data in enumerate(episode_data_list):
            n_steps = len(data['times'])
            for i in range(n_steps):
                writer.writerow([
                    ep_idx + 1,
                    i,
                    f"{data['times'][i]:.6f}",
                    f"{data['angles'][i]:.6f}",
                    f"{data['targets'][i]:.6f}",
                    f"{data['errors'][i]:.6f}",
                    f"{data['reward_details']['total'][i]:.6f}",
                    f"{data['reward_details']['tracking'][i]:.6f}",
                    f"{data['reward_details']['precision'][i]:.6f}",
                    f"{data['reward_details']['flex'][i]:.6f}",
                    f"{data['reward_details']['velocity'][i]:.6f}",
                    f"{data['reward_details']['boundary'][i]:.6f}",
                ])
    print(f"时间戳数据已保存至: {csv_path}")


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    eval_data_dir = os.path.join(os.getcwd(), 'eval_data')
    os.makedirs(eval_data_dir, exist_ok=True)
    start_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(eval_data_dir, f'eval_{start_time_str}.csv')

    target_params = load_target_params(args.target_type, args.target_params)

    env = RealFlexArmTrackingEnv(
        render_mode='human' if args.render else None,
        dt=0.001,
        max_steps=args.max_steps,
        target_type=args.target_type,
        target_params=target_params,
        real_time_factor=0.0
    )
    env.reset(seed=args.seed)

    model = PPO.load(args.model_path)
    print(f"模型加载成功: {args.model_path}")
    print(f"堆叠帧数: {args.stack_frames}")

    episode_data_list = []
    for ep in range(args.episodes):
        print(f"正在运行 Episode {ep+1}/{args.episodes}...")
        data = evaluate_episode(env, model, stack_frames=args.stack_frames,
                                deterministic=True, dt=env.dt)
        episode_data_list.append(data)
        print(f"  Steps: {data['steps']}, Final error: {np.abs(data['errors'][-1]):.2f}°, "
              f"Total reward: {np.sum(data['rewards']):.2f}")

    save_episode_data_to_csv(episode_data_list, csv_path)

    stats = compute_statistics(episode_data_list)

    print("\n========== 评估结果 ==========")
    print(f"平均绝对误差（度）: {stats['mean_error_deg']:.3f} ± {stats['std_error_deg']:.3f}")
    print(f"最小误差（度）: {stats['min_error_deg']:.3f}")
    print(f"最终误差（度）: {stats['final_error_deg']:.3f}")
    print(f"总奖励: {stats['total_reward']:.2f} ± {stats['std_total_reward']:.2f}")
    print(f"平均跟踪奖励: {stats['mean_tracking_reward']:.3f}")
    print(f"平均精度奖励: {stats['mean_precision_reward']:.3f}")
    print(f"平均柔性惩罚: {stats['mean_flex_penalty']:.3f}")
    print(f"平均速度惩罚: {stats['mean_velocity_penalty']:.3f}")
    print(f"总边界惩罚: {stats['total_boundary_penalty']:.3f}")

    stats_file = os.path.join(args.save_dir, 'evaluation_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=4)
    print(f"\n统计结果已保存至: {stats_file}")

    plot_trajectories(episode_data_list, args.save_dir, dt=env.dt)
    print(f"跟踪曲线图已保存至: {args.save_dir}")

    np.savez(os.path.join(args.save_dir, 'all_episode_data.npz'),
             episode_data=episode_data_list)

    env.close()
    print("评估完成。")


if __name__ == "__main__":
    main()
