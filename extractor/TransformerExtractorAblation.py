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
    支持 feature_mask
    """
    def __init__(
        self,
        observation_space: gym.Space,
        seq_len: int,
        feature_dim: int,
        feature_mask: Optional[List[int]] = None,
    ):
        actual_dim = len(feature_mask) if feature_mask else feature_dim
        super().__init__(observation_space, features_dim=actual_dim)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.actual_dim = actual_dim
        self.feature_mask = feature_mask

    def forward(self, observations: th.Tensor) -> th.Tensor:
        batch_size = observations.shape[0]
        x = observations.view(batch_size, self.seq_len, self.feature_dim)
        if self.feature_mask is not None:
            x = x[:, :, self.feature_mask]
        x = x.reshape(batch_size * self.seq_len, self.actual_dim)
        return x


class TransformerExtractorAblation(nn.Module):
    """
    支持消融实验的 Transformer 特征编码网络。
    新增：
    - pos_type: "none" | "sinusoidal" | "learnable"
    - causal_mask: bool
    - init_type: "xavier" | "orthogonal" (在外部处理)
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
        # 消融参数
        pos_type: str = "none",           # "none" | "sinusoidal" | "learnable"
        causal_mask: bool = False,
        fusion_mode: str = "concat",
    ):
        super().__init__()
        device = get_device(device)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.d_model = latent_dim_pi
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf
        self.pos_type = pos_type
        self.causal_mask = causal_mask
        self.fusion_mode = fusion_mode

        _dim_ff = dim_feedforward or latent_dim_pi * 2
        _act = activation_fn()

        self.input_proj = nn.Linear(feature_dim, self.d_model)

        # ---- 位置编码 ----
        if pos_type == "none":
            self.pos_embedding = None
        elif pos_type == "sinusoidal":
            # 固定正弦编码，不训练
            pe = th.zeros(seq_len, self.d_model)
            position = th.arange(0, seq_len, dtype=th.float).unsqueeze(1)
            div_term = th.exp(th.arange(0, self.d_model, 2).float() * (-th.log(10000.0) / self.d_model))
            pe[:, 0::2] = th.sin(position * div_term)
            pe[:, 1::2] = th.cos(position * div_term)
            pe = pe.unsqueeze(0)  # [1, seq_len, d_model]
            self.register_buffer('pos_embedding', pe)
        elif pos_type == "learnable":
            self.pos_embedding = nn.Parameter(th.randn(1, seq_len, self.d_model) * 0.02)
        else:
            raise ValueError(f"Unknown pos_type: {pos_type}")

        # ---- Transformer ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=_dim_ff,
            dropout=dropout,
            activation=_act,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 若使用因果掩码，生成下三角掩码 (attn_mask for TransformerEncoder)
        if causal_mask:
            # 生成形状 (seq_len, seq_len) 的掩码，其中上三角为 -inf
            mask = th.triu(th.ones(seq_len, seq_len) * float('-inf'), diagonal=1)
            self.register_buffer('attn_mask', mask)
        else:
            self.attn_mask = None

        # ---- 时序融合 ----
        self._build_fusion_layer()

        self.to(device)

    def _build_fusion_layer(self):
        if self.fusion_mode == "concat":
            self.temporal_fusion = nn.Sequential(
                nn.Linear(self.d_model * 2, self.d_model),
                nn.ReLU(),
                nn.LayerNorm(self.d_model),
            )
        elif self.fusion_mode == "last_only":
            self.temporal_fusion = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.LayerNorm(self.d_model),
            )
        elif self.fusion_mode == "avg_only":
            self.temporal_fusion = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.LayerNorm(self.d_model),
            )
        elif self.fusion_mode == "attention_pool":
            self.attn_pool = nn.Sequential(
                nn.Linear(self.d_model, 1),
            )
            self.temporal_fusion = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.LayerNorm(self.d_model),
            )
        elif self.fusion_mode == "max_pool":
            # 新增最大池化
            self.temporal_fusion = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.LayerNorm(self.d_model),
            )
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        batch_total, feat_dim = features.shape
        batch_size = batch_total // self.seq_len

        x = features.view(batch_size, self.seq_len, feat_dim)
        x = self.input_proj(x)  # [batch, seq, d_model]

        # 位置编码
        if self.pos_type == "none":
            pass
        elif self.pos_type in ["sinusoidal", "learnable"]:
            if self.pos_embedding is not None:
                x = x + self.pos_embedding

        # Transformer 前向，传入因果掩码
        x = self.transformer(x, mask=self.attn_mask)  # mask will be applied to each layer

        # 时序融合
        if self.fusion_mode == "concat":
            last_frame = x[:, -1, :]
            global_avg = x.mean(dim=1)
            fused = th.cat([last_frame, global_avg], dim=-1)
        elif self.fusion_mode == "last_only":
            fused = x[:, -1, :]
        elif self.fusion_mode == "avg_only":
            fused = x.mean(dim=1)
        elif self.fusion_mode == "attention_pool":
            attn_weights = th.softmax(self.attn_pool(x).squeeze(-1), dim=1)
            fused = (x * attn_weights.unsqueeze(-1)).sum(dim=1)
        elif self.fusion_mode == "max_pool":
            fused, _ = x.max(dim=1)  # [batch, d_model]
        else:
            raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")

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