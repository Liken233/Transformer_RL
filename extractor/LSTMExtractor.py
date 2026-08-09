from typing import Callable, Optional, List, Union, Dict, Type, Tuple
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.utils import get_device


class SeqFeaturesExtractor(BaseFeaturesExtractor):
    """
    配合 VecFrameStack 使用。
    输入: [batch_size, feature_dim * seq_len]
    输出: [batch_size * seq_len, feature_dim]
    """

    def __init__(
        self,
        observation_space: gym.Space,
        seq_len: int,
        feature_dim: int,
    ):
        super().__init__(observation_space, features_dim=feature_dim)
        self.seq_len = seq_len
        self.feature_dim = feature_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.seq_len, self.feature_dim)
        x = x.reshape(batch_size * self.seq_len, self.feature_dim)
        return x


class LSTMExtractor(nn.Module):
    """
    基于 LSTM 的特征编码网络，仅替换 Actor 的特征提取部分。

    输入: [batch_size * seq_len, feature_dim]
    输出: latent_policy [batch_size, latent_dim_pi]
    """

    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        latent_dim_pi: int = 256,
        latent_dim_vf: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation_fn: Type[nn.Module] = nn.ReLU,
        bidirectional: bool = False,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf
        self.bidirectional = bidirectional

        # 输入投影层
        self.input_proj = nn.Linear(feature_dim, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        # LSTM 编码器
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        # 方向因子
        self.num_directions = 2 if bidirectional else 1
        lstm_output_dim = hidden_size * self.num_directions

        # 时序融合：拼接最后一帧隐状态 + 全局平均池化
        self.temporal_fusion = nn.Sequential(
            nn.Linear(lstm_output_dim * 2, latent_dim_pi),
            activation_fn(),
            nn.LayerNorm(latent_dim_pi),
        )

        self.to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        batch_total, feat_dim = features.shape
        batch_size = batch_total // self.seq_len

        # 恢复序列维度: [batch_size, seq_len, feature_dim]
        x = features.view(batch_size, self.seq_len, feat_dim)

        # 输入投影
        x = self.input_proj(x)          # [batch, seq_len, hidden_size]
        x = self.input_norm(x)

        # LSTM 前向传播
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [batch, seq_len, hidden_size * directions]

        # 取最后一帧的隐状态
        last_frame = lstm_out[:, -1, :]       # [batch, hidden_size * directions]

        # 全局平均池化
        global_avg = lstm_out.mean(dim=1)     # [batch, hidden_size * directions]

        # 融合
        fused = th.cat([last_frame, global_avg], dim=-1)  # [batch, hidden_size * directions * 2]
        latent_pi = self.temporal_fusion(fused)           # [batch, latent_dim_pi]

        # Critic 占位（由外部 MLP 处理）
        latent_vf = th.zeros(batch_size, self.latent_dim_vf, device=features.device)

        return latent_pi, latent_vf

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        latent_pi, _ = self.forward(features)
        return latent_pi

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        batch_total = features.shape[0]
        batch_size = batch_total // self.seq_len
        return th.zeros(batch_size, self.latent_dim_vf, device=features.device)
