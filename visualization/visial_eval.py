#!/usr/bin/env python3
"""
评估数据可视化脚本：读取 eval_data/*.csv，生成多维度分析图表
用法：
    python visualize_eval.py --csv eval_data/eval_20260519_193400.csv
    python visualize_eval.py --csv eval_data/              # 批量处理目录下所有csv
    python visualize_eval.py --csv eval_data/ --compare   # 多文件对比模式
"""

import os
import sys
import argparse
import glob
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不依赖 Qt
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# Keep project-local packages importable when this file is run from visualization/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description='评估数据可视化')
    parser.add_argument('--csv', type=str, default='/home/y/vscode_ws/xuan/PPTransformer/eval_data/',
                        help='CSV文件路径或包含CSV的目录路径')
    parser.add_argument('--out_dir', type=str, default='visualization_results',
                        help='图表输出目录')
    parser.add_argument('--compare', action='store_true',
                        help='对比模式：若指定目录，则将所有CSV画在同一张对比图中')
    parser.add_argument('--dpi', type=int, default=200,
                        help='输出图片DPI')
    parser.add_argument('--format', type=str, default='png', choices=['png', 'pdf', 'svg'],
                        help='输出图片格式')
    parser.add_argument('--no_show', action='store_true',
                        help='不弹出显示窗口，仅保存文件')
    return parser.parse_args()


def load_csv(csv_path):
    """加载单个CSV文件，按episode切分数据"""
    episodes = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ep = int(row['episode'])
            if ep not in episodes:
                episodes[ep] = {
                    'step': [], 'time': [], 'angle': [], 'target': [],
                    'error': [], 'total_reward': [], 'reward_tracking': [],
                    'reward_precision': [], 'reward_flex': [],
                    'reward_velocity': [], 'reward_boundary': []
                }
            e = episodes[ep]
            e['step'].append(int(row['step']))
            e['time'].append(float(row['time(s)']))
            e['angle'].append(float(row['angle(deg)']))
            e['target'].append(float(row['target(deg)']))
            e['error'].append(float(row['error(deg)']))
            e['total_reward'].append(float(row['total_reward']))
            e['reward_tracking'].append(float(row['reward_tracking']))
            e['reward_precision'].append(float(row['reward_precision']))
            e['reward_flex'].append(float(row['reward_flex']))
            e['reward_velocity'].append(float(row['reward_velocity']))
            e['reward_boundary'].append(float(row['reward_boundary']))
    # 转为numpy数组
    for ep in episodes:
        for k in episodes[ep]:
            episodes[ep][k] = np.array(episodes[ep][k])
    return episodes


def compute_episode_stats(ep_data):
    """计算单个episode的统计指标"""
    abs_err = np.abs(ep_data['error'])
    return {
        'mean_abs_error': np.mean(abs_err),
        'max_abs_error': np.max(abs_err),
        'final_error': ep_data['error'][-1],
        'total_reward': np.sum(ep_data['total_reward']),
        'mean_tracking': np.mean(ep_data['reward_tracking']),
        'mean_precision': np.mean(ep_data['reward_precision']),
        'mean_flex': np.mean(ep_data['reward_flex']),
        'mean_velocity': np.mean(ep_data['reward_velocity']),
        'total_boundary': np.sum(ep_data['reward_boundary']),
        'settling_idx': np.where(abs_err < 2.0)[0][0] if np.any(abs_err < 2.0) else len(abs_err),
        'overshoot': np.max(ep_data['angle']) - np.max(ep_data['target']) if len(ep_data['target']) > 0 else 0,
    }


def plot_tracking_single(episodes, save_path, dpi=200):
    """单文件：跟踪曲线（所有episode拼接或分subplot）"""
    n_eps = len(episodes)
    fig, axes = plt.subplots(n_eps, 1, figsize=(14, 3 * n_eps), sharex=True)
    if n_eps == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for idx, (ep_num, data) in enumerate(sorted(episodes.items())):
        ax = axes[idx]
        t = data['time']
        ax.plot(t, data['angle'], label='Actual', color=colors[0], linewidth=1.2)
        ax.plot(t, data['target'], label='Target', color=colors[1], linewidth=1.2, linestyle='--')
        ax.fill_between(t, data['angle'], data['target'], alpha=0.15, color='gray')
        ax.set_ylabel('Angle (deg)')
        ax.set_title(f'Episode {ep_num} — Tracking Performance')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if not args.no_show:
        plt.show()
    plt.close()


def plot_error_analysis_single(episodes, save_path, dpi=200):
    """单文件：误差分析（误差曲线 + 误差分布直方图 + 误差箱线图）"""
    n_eps = len(episodes)
    fig = plt.figure(figsize=(16, 4 * n_eps))
    gs = GridSpec(n_eps, 3, figure=fig, width_ratios=[3, 1, 1])

    for idx, (ep_num, data) in enumerate(sorted(episodes.items())):
        t = data['time']
        err = data['error']
        abs_err = np.abs(err)

        # 时域误差
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.plot(t, err, color='crimson', linewidth=1.0)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.axhline(y=2, color='green', linestyle='--', alpha=0.5, label='±2°')
        ax1.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
        ax1.fill_between(t, err, 0, where=(abs_err < 2), alpha=0.2, color='green')
        ax1.fill_between(t, err, 0, where=(abs_err >= 2), alpha=0.2, color='red')
        ax1.set_ylabel('Error (deg)')
        ax1.set_title(f'Episode {ep_num} — Tracking Error')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 误差分布
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.hist(err, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
        ax2.axvline(x=0, color='red', linestyle='--')
        ax2.set_xlabel('Error (deg)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution')
        ax2.grid(True, alpha=0.3)

        # 绝对误差CDF
        ax3 = fig.add_subplot(gs[idx, 2])
        sorted_err = np.sort(abs_err)
        cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
        ax3.plot(sorted_err, cdf, color='darkorange', linewidth=1.5)
        ax3.axvline(x=np.mean(abs_err), color='blue', linestyle='--', label=f'Mean={np.mean(abs_err):.2f}°')
        ax3.set_xlabel('|Error| (deg)')
        ax3.set_ylabel('CDF')
        ax3.set_title('Error CDF')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if not args.no_show:
        plt.show()
    plt.close()


def plot_reward_decomposition_single(episodes, save_path, dpi=200):
    """单文件：奖励函数分解堆叠面积图"""
    n_eps = len(episodes)
    fig, axes = plt.subplots(n_eps, 1, figsize=(14, 3 * n_eps), sharex=True)
    if n_eps == 1:
        axes = [axes]

    for idx, (ep_num, data) in enumerate(sorted(episodes.items())):
        ax = axes[idx]
        t = data['time']

        # 堆叠面积图
        ax.stackplot(t,
                     data['reward_tracking'],
                     data['reward_precision'],
                     data['reward_flex'],
                     data['reward_velocity'],
                     data['reward_boundary'],
                     labels=['Tracking', 'Precision', 'Flex', 'Velocity', 'Boundary'],
                     colors=['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6'],
                     alpha=0.8)
        ax.plot(t, data['total_reward'], color='black', linewidth=1.0, label='Total', linestyle='-')
        ax.set_ylabel('Reward')
        ax.set_title(f'Episode {ep_num} — Reward Decomposition')
        ax.legend(loc='upper right', ncol=3, fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if not args.no_show:
        plt.show()
    plt.close()


def plot_phase_portrait_single(episodes, save_path, dpi=200):
    """单文件：相图（角度 vs 角速度，通过差分近似）"""
    n_eps = len(episodes)
    fig, axes = plt.subplots(1, n_eps, figsize=(5 * n_eps, 5))
    if n_eps == 1:
        axes = [axes]

    for idx, (ep_num, data) in enumerate(sorted(episodes.items())):
        ax = axes[idx]
        angle = data['angle']
        dt = np.mean(np.diff(data['time'])) if len(data['time']) > 1 else 0.001
        velocity = np.gradient(angle, dt)

        # 用颜色映射时间
        scatter = ax.scatter(angle, velocity, c=data['time'], cmap='viridis', s=3, alpha=0.7)
        ax.plot(angle, velocity, color='gray', alpha=0.3, linewidth=0.5)
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Velocity (deg/s)')
        ax.set_title(f'Episode {ep_num} — Phase Portrait')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5)
        plt.colorbar(scatter, ax=ax, label='Time (s)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if not args.no_show:
        plt.show()
    plt.close()


def plot_summary_table_single(episodes, save_path, dpi=200):
    """单文件：统计摘要表格图"""
    stats_list = []
    for ep_num in sorted(episodes.keys()):
        stats = compute_episode_stats(episodes[ep_num])
        stats['episode'] = ep_num
        stats_list.append(stats)

    fig, ax = plt.subplots(figsize=(14, 2 + 0.4 * len(stats_list)))
    ax.axis('off')

    columns = ['Episode', 'Mean|Err|°', 'Max|Err|°', 'FinalErr°', 'TotalRwd',
               'TrackRwd', 'PrecRwd', 'FlexPen', 'VelPen', 'BdryPen']

    cell_text = []
    for s in stats_list:
        cell_text.append([
            f"{s['episode']}",
            f"{s['mean_abs_error']:.3f}",
            f"{s['max_abs_error']:.3f}",
            f"{s['final_error']:.3f}",
            f"{s['total_reward']:.2f}",
            f"{s['mean_tracking']:.3f}",
            f"{s['mean_precision']:.3f}",
            f"{s['mean_flex']:.3f}",
            f"{s['mean_velocity']:.3f}",
            f"{s['total_boundary']:.3f}",
        ])

    table = ax.table(cellText=cell_text, colLabels=columns,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # 表头样式
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4a69bd')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # 交替行颜色
    for i in range(1, len(cell_text) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.set_title('Evaluation Statistics Summary', fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if not args.no_show:
        plt.show()
    plt.close()


def plot_multi_file_comparison(csv_files, out_dir, dpi=200):
    """多文件对比模式：对比不同评估的统计指标"""
    all_stats = {}
    for csv_path in csv_files:
        label = os.path.splitext(os.path.basename(csv_path))[0]
        episodes = load_csv(csv_path)
        stats_per_ep = [compute_episode_stats(episodes[ep]) for ep in sorted(episodes.keys())]
        all_stats[label] = {
            'mean_abs_error': [s['mean_abs_error'] for s in stats_per_ep],
            'max_abs_error': [s['max_abs_error'] for s in stats_per_ep],
            'total_reward': [s['total_reward'] for s in stats_per_ep],
            'final_error': [s['final_error'] for s in stats_per_ep],
        }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['mean_abs_error', 'max_abs_error', 'total_reward', 'final_error']
    titles = ['Mean Absolute Error (deg)', 'Max Absolute Error (deg)',
              'Total Reward', 'Final Error (deg)']

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for label, stats in all_stats.items():
            values = stats[metric]
            x = np.arange(len(values)) + 1
            ax.plot(x, values, marker='o', label=label, linewidth=1.5, markersize=6)
        ax.set_xlabel('Episode')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Multi-Evaluation Comparison', fontsize=14, weight='bold')
    plt.tight_layout()
    save_path = os.path.join(out_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if not args.no_show:
        plt.show()
    plt.close()
    print(f"对比图已保存至: {save_path}")


def export_to_pdf(csv_path, out_dir, dpi=150):
    """将所有图表导出为单个PDF文件"""
    episodes = load_csv(csv_path)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    pdf_path = os.path.join(out_dir, f'{base_name}_report.pdf')

    with PdfPages(pdf_path) as pdf:
        # 1. 跟踪曲线
        fig, axes = plt.subplots(len(episodes), 1, figsize=(12, 3 * len(episodes)), sharex=True)
        if len(episodes) == 1:
            axes = [axes]
        for idx, (ep_num, data) in enumerate(sorted(episodes.items())):
            ax = axes[idx]
            ax.plot(data['time'], data['angle'], label='Actual', linewidth=1.2)
            ax.plot(data['time'], data['target'], label='Target', linestyle='--', linewidth=1.2)
            ax.set_ylabel('Angle (deg)')
            ax.set_title(f'Episode {ep_num} — Tracking')
            ax.legend()
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close()

        # 2. 误差分析
        fig, axes = plt.subplots(len(episodes), 1, figsize=(12, 3 * len(episodes)), sharex=True)
        if len(episodes) == 1:
            axes = [axes]
        for idx, (ep_num, data) in enumerate(sorted(episodes.items())):
            ax = axes[idx]
            ax.plot(data['time'], data['error'], color='crimson', linewidth=1.0)
            ax.axhline(y=0, color='black', linewidth=0.5)
            ax.set_ylabel('Error (deg)')
            ax.set_title(f'Episode {ep_num} — Error')
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close()

        # 3. 奖励分解
        fig, axes = plt.subplots(len(episodes), 1, figsize=(12, 3 * len(episodes)), sharex=True)
        if len(episodes) == 1:
            axes = [axes]
        for idx, (ep_num, data) in enumerate(sorted(episodes.items())):
            ax = axes[idx]
            ax.stackplot(data['time'], data['reward_tracking'], data['reward_precision'],
                        data['reward_flex'], data['reward_velocity'], data['reward_boundary'],
                        labels=['Tracking', 'Precision', 'Flex', 'Velocity', 'Boundary'],
                        alpha=0.7)
            ax.plot(data['time'], data['total_reward'], color='black', linewidth=1.0, label='Total')
            ax.set_ylabel('Reward')
            ax.set_title(f'Episode {ep_num} — Reward')
            ax.legend(fontsize=8, ncol=3)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time (s)')
        plt.tight_layout()
        pdf.savefig(fig, dpi=dpi)
        plt.close()

    print(f"PDF报告已保存至: {pdf_path}")


def process_single_file(csv_path, out_dir, args):
    """处理单个CSV文件"""
    print(f"\n正在处理: {csv_path}")
    episodes = load_csv(csv_path)
    print(f"  加载完成，共 {len(episodes)} 个 episode")

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    file_out_dir = os.path.join(out_dir, base_name)
    os.makedirs(file_out_dir, exist_ok=True)

    # 生成各类图表
    plot_tracking_single(episodes, os.path.join(file_out_dir, '01_tracking.png'), args.dpi)
    plot_error_analysis_single(episodes, os.path.join(file_out_dir, '02_error_analysis.png'), args.dpi)
    plot_reward_decomposition_single(episodes, os.path.join(file_out_dir, '03_reward_decomp.png'), args.dpi)
    plot_phase_portrait_single(episodes, os.path.join(file_out_dir, '04_phase_portrait.png'), args.dpi)
    plot_summary_table_single(episodes, os.path.join(file_out_dir, '05_summary_table.png'), args.dpi)

    # 导出PDF
    export_to_pdf(csv_path, file_out_dir, args.dpi)

    print(f"  所有图表已保存至: {file_out_dir}")


def main():
    global args
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 判断输入是文件还是目录
    if os.path.isfile(args.csv):
        csv_files = [args.csv]
    elif os.path.isdir(args.csv):
        csv_files = sorted(glob.glob(os.path.join(args.csv, '*.csv')))
        if not csv_files:
            print(f"错误: 目录 {args.csv} 下未找到任何CSV文件")
            sys.exit(1)
    else:
        print(f"错误: 路径不存在 {args.csv}")
        sys.exit(1)

    print(f"发现 {len(csv_files)} 个CSV文件")

    # 对比模式
    if args.compare and len(csv_files) > 1:
        plot_multi_file_comparison(csv_files, args.out_dir, args.dpi)

    # 逐个处理
    for csv_path in csv_files:
        process_single_file(csv_path, args.out_dir, args)

    print("\n可视化完成。")


if __name__ == "__main__":
    main()
