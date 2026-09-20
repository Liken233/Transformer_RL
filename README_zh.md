# Simulink 柔性关节机械臂 PPO 工程（中文说明）

> English documentation: [README.md](README.md)

本目录包含基于 Stable-Baselines3 PPO 的柔性关节机械臂跟踪控制实验代码，环境通过 UDP 与 MATLAB/Simulink 交互。对应论文：*Transformer-Based Temporal Reinforcement Learning for Series Elastic Actuator Control*。

## 目录结构

~~~text
├── training/                 # 训练入口
│   ├── train_simulink.py             # 基线 PPO/MLP
│   ├── train_simulink_cnn.py         # CNN 策略
│   ├── train_simulink_lstm.py        # LSTM 策略
│   └── train_simulink_transformer.py # Transformer 策略
├── evaluation/               # 模型评估与模型对比
│   ├── eval_model.py                  # 3D 环境评估
│   ├── eval_cnn.py / eval_lstm.py / eval_transformer.py
│   ├── eval_transformer_single.py    # 单模型评估
│   └── eval_single.py                 # 多模型对比
├── environments/             # Gymnasium 环境封装
│   ├── simulink_env.py                # UDP/Simulink 环境
│   └── simulink_3d_env.py            # 3D 渲染环境
├── extractor/                # CNN/LSTM/Transformer 特征提取器
├── policy/                   # 对应 Actor-Critic 策略
├── visualization/            # 评估 CSV 可视化（visial_eval.py）
├── experiment_data/          # 实验数据记录（见下）
├── requirements.txt
└── environment.yml
~~~

### 实验数据（experiment_data/）

~~~text
experiment_data/
├── simulation_step_response/            # 仿真阶跃实验
├── simulation_sinusoidal_tracking/      # 仿真正弦跟踪实验
├── simulation_multi_model_comparison/   # 仿真多模型（MLP/LSTM/CNN/Transformer）对比
├── hardware_multi_target_step/          # 实物多目标阶跃实验
├── hardware_sinusoidal_tracking/        # 实物正弦跟踪实验
├── hardware_payload_experiment/         # 实物负载实验（0/400/800/1200 g）
└── window_length_ablation/              # 历史窗口长度消融（L = 1/2/4/8）
~~~

每个目录包含原始记录（.mat/.csv）、处理后数据、汇总指标、绘图脚本以及论文所用图表。

## 安装环境

建议使用 Python 3.10 的 Conda 环境：

~~~powershell
conda env create -f environment.yml
conda activate bullet_arm
pip install -r requirements.txt
~~~

也可以直接在已有 Python 环境中安装 `requirements.txt`。运行前请确认 `stable-baselines3`、`gymnasium`、`torch` 和 `matplotlib` 可导入。

## 运行约定

请在本目录作为当前工作目录执行命令。所有入口脚本都包含项目根路径引导，移动到二级目录后仍可正确导入 `environments`、`policy` 和 `extractor`。

### 训练

~~~powershell
python training/train_simulink.py --exp_name baseline --target_type random
python training/train_simulink_cnn.py --exp_name cnn --target_type random --stack_frames 8 --feature_dim 5
python training/train_simulink_lstm.py --exp_name lstm --target_type random --stack_frames 8 --feature_dim 5
python training/train_simulink_transformer.py --exp_name transformer --target_type random --stack_frames 8 --feature_dim 5
~~~

训练结果默认写入当前目录下的 `experiments/<实验名>/`，其中包含模型、日志、配置和评估数据。

### 评估

将 `--model_path` 指向训练生成的 `.zip` 模型：

~~~powershell
python evaluation/eval_model.py --model_path experiments/<实验名>/models/final_model.zip
python evaluation/eval_transformer.py --mode evaluate --model_path experiments/<实验名>/models/final_model.zip
python evaluation/eval_lstm.py --mode evaluate --model_path experiments/<实验名>/models/final_model.zip
python evaluation/eval_cnn.py --mode evaluate --model_path experiments/<实验名>/models/final_model.zip
python evaluation/eval_single.py --transformer <transformer.zip> --lstm <lstm.zip> --cnn <cnn.zip>
~~~

`eval_*` 脚本默认将 CSV 和图表写入 `eval_data/` 或指定的 `--save_dir`。`eval_* --mode sync` 需要同时运行 MATLAB/Simulink，并确保 UDP 地址和端口参数匹配。

### 可视化

~~~powershell
python visualization/visial_eval.py --csv eval_data/
python visualization/visial_eval.py --csv eval_data/<file>.csv --out_dir visualization_results
~~~

## 路径与兼容性说明

- 模型路径、CSV 路径和输出目录支持相对路径；相对路径以执行命令时的当前目录为准。
- 原脚本中的 Linux 示例模型路径仅作为历史默认值，实际使用时请显式传入 Windows 可访问的 `--model_path`。
- `simulink_env.py` 使用 UDP 与外部 Simulink 通信；运行同步评估前请先启动对应模型。
- `visial_eval.py` 文件名保留原样，以免影响已有调用；它的功能是评估结果可视化。
