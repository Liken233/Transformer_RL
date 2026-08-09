#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
多模型对比评估工具（支持四种独立模型参数）
================================================================================
在同一目标轨迹下，分别加载 Transformer、MLP、LSTM、CNN 四种 PPO 模型，
运行相同 episode，对比跟踪性能并绘制图表。

支持通过参数为每个子图指定放大区域（图中图），用于观察变化密集或差异微小的区域。
支持为每个模型独立设置堆叠帧数。

使用示例：
    python eval_single.py \\
        --transformer model_transformer.zip \\
        --mlp model_mlp.zip \\
        --lstm model_lstm.zip \\
        --cnn model_cnn.zip \\
        --stack_frames 8 \\
        --transformer_frames 16 \\
        --mlp_frames 8 \\
        --target_type step --episodes 1 --max_steps 10000 \\
        --zoom_a "0.5,2.5,43,53" \\
        --zoom_c "0.5,2.5,-3,7"
"""

import os
import sys
import argparse
import json
import csv
from pathlib import Path
from collections import deque
from datetime import datetime
from typing import Optional, List, Dict, Tuple

# Keep project-local packages importable when this file is run from evaluation/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ========== Stable-Baselines3 ==========
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
except ImportError:
    print("[ERROR] 请安装 stable-baselines3: pip install stable-baselines3")
    raise

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
# 1. 工具函数
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='多模型对比评估工具（支持 Transformer/MLP/LSTM/CNN）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python eval_single.py \\
      --transformer model_transformer.zip \\
      --mlp model_mlp.zip \\
      --lstm model_lstm.zip \\
      --cnn model_cnn.zip \\
      --stack_frames 8 \\
      --transformer_frames 16 \\
      --mlp_frames 8 \\
      --target_type step --episodes 1 --max_steps 10000 \\
      --zoom_a "0.5,2.5,43,53" \\
      --zoom_c "0.5,2.5,-3,7"
        """
    )

    # 四个独立模型路径参数
    parser.add_argument('--transformer', type=str, default="/home/y/vscode_ws/xuan/PPTransformer/experiments/transformer/20260722_111621_random_transformer_nenv30_len8/models/final_model.zip",
                        help='Transformer 模型路径 (.zip)')
    parser.add_argument('--mlp', type=str, default="/home/y/vscode_ws/xuan/PPTransformer/experiments/mlp/20260713_161023_random_ppo_stack1_envs31/models/final_model.zip",
                        help='MLP 模型路径 (.zip)')
    parser.add_argument('--lstm', type=str, default="/home/y/vscode_ws/xuan/PPTransformer/experiments/lstm/20260722_142736_random_lstm_nenv30_len8/models/final_model.zip",
                        help='LSTM 模型路径 (.zip)')
    parser.add_argument('--cnn', type=str, default="/home/y/vscode_ws/xuan/PPTransformer/experiments/cnn/20260722_163547_random_cnn_nenv30_len8/models/final_model.zip",
                        help='CNN 模型路径 (.zip)')

    # 每个模型的独立堆叠帧数参数（可选）
    parser.add_argument('--transformer_frames', type=int, default=None,
                        help='Transformer 模型的堆叠帧数（若不指定则使用 --stack_frames）')
    parser.add_argument('--mlp_frames', type=int, default=1,
                        help='MLP 模型的堆叠帧数（若不指定则使用 --stack_frames）')
    parser.add_argument('--lstm_frames', type=int, default=None,
                        help='LSTM 模型的堆叠帧数（若不指定则使用 --stack_frames）')
    parser.add_argument('--cnn_frames', type=int, default=None,
                        help='CNN 模型的堆叠帧数（若不指定则使用 --stack_frames）')

    # 环境与评估通用参数
    parser.add_argument('--stack_frames', type=int, default=8,
                        help='默认堆叠帧数（当模型未单独指定时使用）')
    parser.add_argument('--feature_dim', type=int, default=5,
                        help='单帧观测维度（不是总维度）')
    parser.add_argument('--target_type', type=str, default='step',
                        choices=['fixed', 'sine', 'random', 'step'],
                        help='目标轨迹类型')
    parser.add_argument('--target_params', type=str, default=None,
                        help='目标参数 JSON 字符串')
    parser.add_argument('--dt', type=float, default=0.001,
                        help='仿真步长')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（用于环境初始化，保证公平）')
    parser.add_argument('--episodes', type=int, default=1,
                        help='每个模型运行的 episode 数量（通常1个足够）')
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='每 episode 最大步数')
    parser.add_argument('--render', action='store_true',
                        help='是否开启 GUI 渲染（默认关闭）')
    parser.add_argument('--save_dir', type=str, default='compare_results',
                        help='结果保存目录')

    # 图中图放大区域参数（四个子图各自独立）
    parser.add_argument('--zoom_a', type=str, default="0.5,2.0,80,110",  # "0.5,2.5,39,55"
                        help='子图(a)放大区域，格式: "x1,x2,y1,y2"（数据坐标），例如 "0.5,0.8,-10,10"')
    parser.add_argument('--zoom_b', type=str, default="0.5,2.5,80,120",  # "0.5,2.5,45,56"
                        help='子图(b)放大区域，格式同上')
    parser.add_argument('--zoom_c', type=str, default="0.5,2.5,-15,15",  # "0.5,2.5,-3,7"
                        help='子图(c)放大区域，格式同上')
    parser.add_argument('--zoom_d', type=str, default=None,  # "0.5,2.5,-100,100"
                        help='子图(d)放大区域，格式同上')

    return parser.parse_args()


def parse_zoom_region(zoom_str: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    """解析放大区域字符串，返回 (x1, x2, y1, y2) 元组"""
    if zoom_str is None:
        return None
    parts = [float(x.strip()) for x in zoom_str.split(',')]
    if len(parts) != 4:
        raise ValueError(f"放大区域需要四个数值（x1,x2,y1,y2），但得到 {len(parts)} 个")
    return tuple(parts)


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
            'final_target': np.radians(90.0),
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
        env = create_env(target_type, target_params, dt, max_steps,
                         render_mode=None, seed=seed)
        return env

    vec_env = DummyVecEnv([_make_env])
    vec_env = VecFrameStack(vec_env, n_stack=stack_frames)

    raw_env = vec_env.envs[0] if hasattr(vec_env, 'envs') else None
    return vec_env, raw_env


# =====================================================================
# 2. 单模型评估函数（丢弃最后一帧）
# =====================================================================

def evaluate_episode(vec_env, model, args, episode_idx: int, deterministic: bool = True) -> dict:
    """运行单个 episode，返回记录数据（已丢弃最后一帧）"""
    obs = vec_env.reset()
    done = np.array([False])

    times = []
    angles = []
    motor_angles = []
    targets = []
    errors = []
    rewards = []
    actions = []

    step = 0
    while not done[0]:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = vec_env.step(action)

        if len(obs.shape) > 1:
            latest_obs = obs[0][:args.feature_dim]
        else:
            latest_obs = obs[:args.feature_dim]

        q = latest_obs[0]
        th = latest_obs[2]
        target_angle = latest_obs[4] if len(latest_obs) > 4 else 0.0

        times.append(step * args.dt)
        angles.append(np.degrees(q))
        motor_angles.append(np.degrees(th))
        targets.append(np.degrees(target_angle))
        errors.append(np.degrees(q - target_angle))
        rewards.append(float(reward[0]) if isinstance(reward, np.ndarray) else float(reward))
        if isinstance(action, np.ndarray):
            action_val = float(action.item()) if action.size == 1 else float(action[0])
        else:
            action_val = float(action)
        actions.append(action_val)

        step += 1
        if step >= args.max_steps:
            break

    # 丢弃最后一帧
    return {
        'times': np.array(times[:-1]) if len(times) > 1 else np.array(times),
        'angles': np.array(angles[:-1]) if len(angles) > 1 else np.array(angles),
        'motor_angles': np.array(motor_angles[:-1]) if len(motor_angles) > 1 else np.array(motor_angles),
        'targets': np.array(targets[:-1]) if len(targets) > 1 else np.array(targets),
        'errors': np.array(errors[:-1]) if len(errors) > 1 else np.array(errors),
        'rewards': np.array(rewards[:-1]) if len(rewards) > 1 else np.array(rewards),
        'actions': np.array(actions[:-1]) if len(actions) > 1 else np.array(actions),
        'steps': step - 1 if step > 0 else 0
    }


# =====================================================================
# 3. 多模型对比器
# =====================================================================

class ModelComparator:
    def __init__(self, args):
        self.args = args
        self.results = {}  # {label: [episode_data, ...]}

        # 解析图中图放大区域
        self.zoom_regions = {
            'a': parse_zoom_region(args.zoom_a),
            'b': parse_zoom_region(args.zoom_b),
            'c': parse_zoom_region(args.zoom_c),
            'd': parse_zoom_region(args.zoom_d),
        }

        os.makedirs(args.save_dir, exist_ok=True)

        # 收集所有非 None 的模型及其对应的帧数
        self.model_configs = []  # 列表元素: (label, path, frames)
        default_frames = args.stack_frames

        if args.transformer:
            frames = args.transformer_frames if args.transformer_frames is not None else default_frames
            self.model_configs.append(('Transformer', args.transformer, frames))
        if args.mlp:
            frames = args.mlp_frames if args.mlp_frames is not None else default_frames
            self.model_configs.append(('MLP', args.mlp, frames))
        if args.lstm:
            frames = args.lstm_frames if args.lstm_frames is not None else default_frames
            self.model_configs.append(('LSTM', args.lstm, frames))
        if args.cnn:
            frames = args.cnn_frames if args.cnn_frames is not None else default_frames
            self.model_configs.append(('CNN', args.cnn, frames))

        if not self.model_configs:
            print("[错误] 请至少指定一个模型路径（--transformer / --mlp / --lstm / --cnn）")
            sys.exit(1)

        # 打印帧数配置
        print("[帧数配置]")
        for label, _, frames in self.model_configs:
            print(f"  {label}: {frames} 帧")
        print()

    def run_all_models(self):
        """加载所有模型并运行评估，每个模型使用各自的堆叠帧数创建环境"""
        target_params = load_target_params(self.args.target_type, self.args.target_params)
        render_mode = 'human' if self.args.render else None

        n_models = len(self.model_configs)
        print(f"将对比 {n_models} 个模型: {', '.join([c[0] for c in self.model_configs])}")

        for idx, (label, path, frames) in enumerate(self.model_configs):
            print(f"\n{'='*50}")
            print(f"[模型 {idx+1}/{n_models}] 加载: {label} ({path})")
            print(f"  堆叠帧数: {frames}")
            try:
                model = PPO.load(path)
                print(f"  策略类型: {type(model.policy).__name__}")
            except Exception as e:
                print(f"  加载失败: {e}")
                continue

            # 为当前模型创建独立环境（使用指定帧数）
            vec_env, raw_env = create_vec_env_with_stack(
                target_type=self.args.target_type,
                target_params=target_params,
                dt=self.args.dt,
                max_steps=self.args.max_steps,
                seed=self.args.seed,
                stack_frames=frames,
                feature_dim=self.args.feature_dim,
            )
            if raw_env is not None:
                raw_env.render_mode = render_mode

            episode_data_list = []
            for ep in range(self.args.episodes):
                print(f"  Episode {ep+1}/{self.args.episodes} 运行中...")
                data = evaluate_episode(vec_env, model, self.args, ep,
                                        deterministic=True)
                episode_data_list.append(data)
                print(f"    步数: {data['steps']}, 最终误差: {abs(data['errors'][-1]):.3f}°, "
                      f"平均误差: {np.mean(np.abs(data['errors'])):.3f}°")

            self.results[label] = episode_data_list
            # 关闭环境
            vec_env.close()
            print(f"[{label}] 评估完成，环境已关闭")

        print("\n[全部模型评估完成]")

    def compute_statistics(self):
        """计算每个模型的统计指标（对所有 episode 取平均），返回原生 Python float"""
        stats = {}
        for label, ep_list in self.results.items():
            mean_errors = []
            min_errors = []
            final_errors = []
            total_rewards = []
            rmse_errors = []
            for data in ep_list:
                err = data['errors']
                abs_err = np.abs(err)
                mean_errors.append(np.mean(abs_err))
                min_errors.append(np.min(abs_err))
                final_errors.append(abs_err[-1])
                total_rewards.append(np.sum(data['rewards']))
                rmse_errors.append(np.sqrt(np.mean(err**2)))
            stats[label] = {
                'mean_error_deg': float(np.mean(mean_errors)),
                'std_error_deg': float(np.std(mean_errors)),
                'min_error_deg': float(np.min(min_errors)),
                'final_error_deg': float(np.mean(final_errors)),
                'total_reward': float(np.mean(total_rewards)),
                'rmse_deg': float(np.mean(rmse_errors)),
            }
        return stats

    def plot_comparison(self):
        """
        绘制 2×2 子图：
        (a) 左上：轨迹跟踪（负载角度 q 与目标）
        (b) 右上：电机侧角度 θ（不绘制目标线）
        (c) 左下：跟踪误差 (q - target)
        (d) 右下：控制输入
        每个子图可选择添加图中图（由参数控制）
        """
        if not self.results:
            print("[警告] 无数据可绘制")
            return

        first_label = list(self.results.keys())[0]
        first_ep_data = self.results[first_label][0]
        time_axis = first_ep_data['times']

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        linestyles = ['-', '--', '-.', ':']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        # 绘制四个子图
        subplot_keys = ['a', 'b', 'c', 'd']
        titles = [
            'Tracking Performance (q)',
            'Motor Side Angle (θ)',
            'Tracking Error',
            'Control Actions'
        ]
        ylabels = ['Angle (deg)', 'Angle (deg)', 'Error (deg)', 'Control Input']
        target_visible = [True, False, False, False]  # 只在 (a) 原图绘制目标线，(b) 不绘制

        for idx, key in enumerate(subplot_keys):
            ax = axes[idx // 2, idx % 2]
            # 绘制各模型曲线
            for i, (label, ep_list) in enumerate(self.results.items()):
                data = ep_list[0]
                if key == 'a':
                    y = data['angles']
                elif key == 'b':
                    y = data['motor_angles']
                elif key == 'c':
                    y = data['errors']
                elif key == 'd':
                    y = data['actions']
                else:
                    continue
                ax.plot(time_axis, y,
                        label=label, color=colors[i], linestyle=linestyles[i % len(linestyles)],
                        linewidth=1.5)

            # 仅在 (a) 绘制目标线
            if target_visible[idx]:
                ax.plot(time_axis, first_ep_data['targets'], 'k--', label='Target', linewidth=2, alpha=0.7)

            ax.set_ylabel(ylabels[idx])
            ax.set_title(titles[idx])
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.text(0.5, -0.15, f'({key})', transform=ax.transAxes, ha='center', fontsize=12, fontweight='bold')

            # 添加图中图（如果指定了放大区域）
            if self.zoom_regions[key] is not None:
                x1, x2, y1, y2 = self.zoom_regions[key]
                mask = (time_axis >= x1) & (time_axis <= x2)
                idx_zoom = np.where(mask)[0]
                if len(idx_zoom) > 0:
                    # 在主图上标记区域
                    from matplotlib.patches import Rectangle
                    rect = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='gray', facecolor='none', alpha=0.5)
                    ax.add_patch(rect)

                    # 创建图中图（位于子图正下方）
                    inset_ax = inset_axes(ax, width="50%", height="50%", loc='lower center',
                                          bbox_to_anchor=(-0.03, 0.07, 1, 1),
                                          bbox_transform=ax.transAxes)

                    # 在图中图中绘制相同数据
                    for i, (label, ep_list) in enumerate(self.results.items()):
                        data = ep_list[0]
                        if key == 'a':
                            y_zoom = data['angles'][idx_zoom]
                        elif key == 'b':
                            y_zoom = data['motor_angles'][idx_zoom]
                        elif key == 'c':
                            y_zoom = data['errors'][idx_zoom]
                        elif key == 'd':
                            y_zoom = data['actions'][idx_zoom]
                        else:
                            continue
                        inset_ax.plot(time_axis[idx_zoom], y_zoom,
                                      label=label, color=colors[i], linestyle=linestyles[i % len(linestyles)],
                                      linewidth=1.2)

                    # 对于 (a) 图中图，额外绘制 target 线
                    if key == 'a':
                        inset_ax.plot(time_axis[idx_zoom], first_ep_data['targets'][idx_zoom],
                                      'k--', linewidth=1.5, alpha=0.7, label='Target' if 'Target' not in inset_ax.get_legend_handles_labels()[1] else '')

                    inset_ax.set_xlim(x1, x2)
                    inset_ax.set_ylim(y1, y2)
                    inset_ax.tick_params(labelsize=7)
                    inset_ax.grid(True, alpha=0.3)
                    # 标记放大区域
                    mark_inset(ax, inset_ax, loc1=1, loc2=3, fc="none", ec="0.5", linewidth=1.0)

        plt.tight_layout()
        save_path = os.path.join(self.args.save_dir, 'comparison.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"[图表] 对比图已保存: {save_path}")

    def save_csv(self):
        """保存每个模型的详细轨迹数据到 CSV"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        for label, ep_list in self.results.items():
            for ep_idx, data in enumerate(ep_list):
                csv_path = os.path.join(self.args.save_dir,
                                        f'{label}_ep{ep_idx+1}_{timestamp}.csv')
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['step', 'time(s)', 'angle(deg)', 'motor_angle(deg)',
                                     'target(deg)', 'error(deg)', 'action', 'reward'])
                    for i in range(len(data['times'])):
                        writer.writerow([
                            i,
                            f"{data['times'][i]:.6f}",
                            f"{data['angles'][i]:.6f}",
                            f"{data['motor_angles'][i]:.6f}",
                            f"{data['targets'][i]:.6f}",
                            f"{data['errors'][i]:.6f}",
                            f"{data['actions'][i]:.6f}",
                            f"{data['rewards'][i]:.6f}",
                        ])
                print(f"[CSV] {label} Episode {ep_idx+1} 已保存: {csv_path}")

    def save_stats(self):
        """保存统计指标为 JSON"""
        stats = self.compute_statistics()
        stats_path = os.path.join(self.args.save_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"[统计] 已保存: {stats_path}")

    def run(self):
        """运行完整对比流程"""
        print("="*70)
        print("  多模型对比评估工具（Transformer/MLP/LSTM/CNN）")
        print(f"  目标轨迹: {self.args.target_type}")
        print(f"  Episode数: {self.args.episodes}  最大步数: {self.args.max_steps}")
        print("="*70)

        self.run_all_models()
        self.plot_comparison()
        self.save_csv()
        self.save_stats()
        print("\n[完成]")


# =====================================================================
# 主入口
# =====================================================================

def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(args.save_dir, exist_ok=True)

    comparator = ModelComparator(args)
    comparator.run()


if __name__ == "__main__":
    main()
