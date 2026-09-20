# Transformer-Based Temporal RL for Series Elastic Actuator Control

This repository contains the experiment code for PPO-based tracking control of a flexible-joint (series elastic) robot arm, built on Stable-Baselines3. The environment communicates with MATLAB/Simulink over UDP. The paper associated with this repository: *Transformer-Based Temporal Reinforcement Learning for Series Elastic Actuator Control*.

> 中文说明见 [README_zh.md](README_zh.md)。

## Repository Structure

~~~text
├── training/                 # Training entry points
│   ├── train_simulink.py             # Baseline PPO/MLP
│   ├── train_simulink_cnn.py         # CNN policy
│   ├── train_simulink_lstm.py        # LSTM policy
│   └── train_simulink_transformer.py # Transformer policy
├── evaluation/               # Model evaluation and multi-model comparison
│   ├── eval_model.py                  # 3D environment evaluation
│   ├── eval_cnn.py / eval_lstm.py / eval_transformer.py
│   ├── eval_transformer_single.py    # Single-model evaluation
│   └── eval_single.py                 # Multi-model comparison
├── environments/             # Gymnasium environment wrappers
│   ├── simulink_env.py                # UDP/Simulink environment
│   └── simulink_3d_env.py            # 3D rendering environment
├── extractor/                # CNN/LSTM/Transformer feature extractors
├── policy/                   # Corresponding Actor-Critic policies
├── visualization/            # Evaluation CSV visualization (visial_eval.py)
├── experiment_data/          # Recorded experiment datasets (see below)
├── requirements.txt
└── environment.yml
~~~

### Experiment Data (`experiment_data/`)

~~~text
experiment_data/
├── simulation_step_response/            # Simulated step-response experiment
├── simulation_sinusoidal_tracking/      # Simulated sinusoidal-tracking experiment
├── simulation_multi_model_comparison/   # Simulated MLP/LSTM/CNN/Transformer comparison
├── hardware_multi_target_step/          # Hardware multi-target step experiment
├── hardware_sinusoidal_tracking/        # Hardware sinusoidal-tracking experiment
├── hardware_payload_experiment/         # Hardware payload experiment (0/400/800/1200 g)
└── window_length_ablation/              # Historical window length ablation (L = 1/2/4/8)
~~~

Each directory contains raw records (`.mat`/`.csv`), processed data, summary metrics, plotting scripts, and the figures used in the paper.

## Installation

A Python 3.10 Conda environment is recommended:

~~~powershell
conda env create -f environment.yml
conda activate bullet_arm
pip install -r requirements.txt
~~~

Alternatively, install `requirements.txt` directly into an existing Python environment. Before running, make sure `stable-baselines3`, `gymnasium`, `torch`, and `matplotlib` can be imported.

## Usage

Run all commands with this directory as the working directory. Every entry script bootstraps the project root path, so imports of `environments`, `policy`, and `extractor` keep working from subdirectories.

### Training

~~~powershell
python training/train_simulink.py --exp_name baseline --target_type random
python training/train_simulink_cnn.py --exp_name cnn --target_type random --stack_frames 8 --feature_dim 5
python training/train_simulink_lstm.py --exp_name lstm --target_type random --stack_frames 8 --feature_dim 5
python training/train_simulink_transformer.py --exp_name transformer --target_type random --stack_frames 8 --feature_dim 5
~~~

Training outputs are written to `experiments/<exp_name>/`, including models, logs, configurations, and evaluation data.

### Evaluation

Point `--model_path` to the `.zip` model produced by training:

~~~powershell
python evaluation/eval_model.py --model_path experiments/<exp_name>/models/final_model.zip
python evaluation/eval_transformer.py --mode evaluate --model_path experiments/<exp_name>/models/final_model.zip
python evaluation/eval_lstm.py --mode evaluate --model_path experiments/<exp_name>/models/final_model.zip
python evaluation/eval_cnn.py --mode evaluate --model_path experiments/<exp_name>/models/final_model.zip
python evaluation/eval_single.py --transformer <transformer.zip> --lstm <lstm.zip> --cnn <cnn.zip>
~~~

The `eval_*` scripts write CSV files and figures to `eval_data/` or the `--save_dir` given on the command line. `eval_* --mode sync` requires MATLAB/Simulink to be running with matching UDP addresses and ports.

### Visualization

~~~powershell
python visualization/visial_eval.py --csv eval_data/
python visualization/visial_eval.py --csv eval_data/<file>.csv --out_dir visualization_results
~~~

## Notes on Paths and Compatibility

- Model paths, CSV paths, and output directories accept relative paths, resolved against the current working directory.
- The Linux sample model paths in the original scripts are historical defaults only; pass an explicit Windows-accessible `--model_path` in practice.
- `simulink_env.py` communicates with external Simulink over UDP; start the corresponding model before running synchronized evaluation.
- The file name `visial_eval.py` is kept as-is for backward compatibility; it performs visualization of evaluation results.
