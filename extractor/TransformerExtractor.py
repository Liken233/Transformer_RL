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



class TransformerExtractor(nn.Module):
    """
    基于 Transformer 的特征编码网络，仅替换 Actor 的特征提取部分。

    输入: [batch_size * seq_len, feature_dim]
    输出: latent_policy [batch_size, latent_dim_pi]
    """

    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        latent_dim_pi: int = 256,
        latent_dim_vf: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation_fn: Type[nn.Module] = nn.ReLU,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = latent_dim_pi
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

        _dim_ff = dim_feedforward or latent_dim_pi * 2
        _act = activation_fn()

        self.input_proj = nn.Linear(feature_dim, self.d_model)
        self.pos_embedding = nn.Parameter(th.randn(1, seq_len, self.d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=_dim_ff,
            dropout=dropout,
            activation=_act,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            activation_fn(),
            nn.LayerNorm(self.d_model),
        )

        self.to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        batch_total, feat_dim = features.shape
        batch_size = batch_total // self.seq_len

        x = features.view(batch_size, self.seq_len, feat_dim)
        x = self.input_proj(x)
        x = x + self.pos_embedding
        x = self.transformer(x)

        last_frame = x[:, -1, :]
        global_avg = x.mean(dim=1)
        fused = th.cat([last_frame, global_avg], dim=-1)

        latent_pi = self.temporal_fusion(fused)
        latent_vf = th.zeros(batch_size, self.latent_dim_vf, device=features.device)

        return latent_pi, latent_vf

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        latent_pi, _ = self.forward(features)
        return latent_pi

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        batch_total = features.shape[0]
        batch_size = batch_total // self.seq_len
        return th.zeros(batch_size, self.latent_dim_vf, device=features.device)