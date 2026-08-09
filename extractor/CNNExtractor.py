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


class CNNExtractor(nn.Module):
    """
    基于 1D-CNN 的特征编码网络，仅替换 Actor 的特征提取部分。

    输入: [batch_size * seq_len, feature_dim]
    输出: latent_policy [batch_size, latent_dim_pi]

    设计思路:
    - 将时间序列视为 1D 信号，用 CNN 提取局部时序特征
    - 通过多层卷积扩大感受野，捕捉不同时间尺度的模式
    - 最后拼接最后一帧特征 + 全局池化特征，送入融合层
    """

    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        latent_dim_pi: int = 256,
        latent_dim_vf: int = 256,
        channels: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.1,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_bn: bool = True,
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        device = get_device(device)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.latent_dim_pi = latent_dim_pi
        self.latent_dim_vf = latent_dim_vf

        # 默认卷积配置: [64, 128, 256] 三层，kernel=3
        _channels = channels or [64, 128, 256]
        _kernel_sizes = kernel_sizes or [3, 3, 3]

        assert len(_channels) == len(_kernel_sizes),             "channels 和 kernel_sizes 长度必须相同"

        # 输入投影: feature_dim -> channels[0]
        self.input_proj = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=_channels[0],
            kernel_size=1,
            padding=0,
        )
        if use_bn:
            self.input_bn = nn.BatchNorm1d(_channels[0])

        # 构建卷积层
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if use_bn else None
        self.dropout_layers = nn.ModuleList()

        in_ch = _channels[0]
        for out_ch, k in zip(_channels, _kernel_sizes):
            # 使用 causal padding（左侧补零，保持因果性）
            padding = k - 1
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=k,
                    padding=padding,
                    padding_mode='zeros',
                )
            )
            if use_bn:
                self.bn_layers.append(nn.BatchNorm1d(out_ch))
            self.dropout_layers.append(nn.Dropout(dropout))
            in_ch = out_ch

        self.use_bn = use_bn
        self.activation = activation_fn()

        # 计算卷积后的序列长度（带 causal padding）
        # 每层 causal padding 后: L_out = L_in + k - 1
        current_len = seq_len
        for k in _kernel_sizes:
            current_len = current_len + k - 1
        self.output_seq_len = current_len

        # 时序融合: 最后一帧特征 + 全局平均池化
        last_feat_dim = _channels[-1]
        self.temporal_fusion = nn.Sequential(
            nn.Linear(last_feat_dim * 2, latent_dim_pi),
            activation_fn(),
            nn.LayerNorm(latent_dim_pi),
        )

        self.to(device)

    def _causal_crop(self, x: th.Tensor, target_len: int) -> th.Tensor:
        """
        因果裁剪: 只保留最后 target_len 个时间步。
        确保输出与输入序列对齐（因果性）。
        """
        if x.shape[-1] > target_len:
            return x[..., -target_len:]
        return x

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        batch_total, feat_dim = features.shape
        batch_size = batch_total // self.seq_len

        # 恢复序列维度: [batch_size, seq_len, feature_dim]
        x = features.view(batch_size, self.seq_len, feat_dim)

        # 转置为 Conv1d 输入格式: [batch, feature_dim, seq_len]
        x = x.permute(0, 2, 1)  # [batch, feature_dim, seq_len]

        # 输入投影
        x = self.input_proj(x)  # [batch, channels[0], seq_len]
        if self.use_bn:
            x = self.input_bn(x)
        x = self.activation(x)

        # 卷积层
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)  # causal padding 后长度增加
            if self.use_bn:
                x = self.bn_layers[i](x)
            x = self.activation(x)
            x = self.dropout_layers[i](x)

        # 因果裁剪回原始序列长度
        x = self._causal_crop(x, self.seq_len)

        # 转回: [batch, seq_len, channels[-1]]
        x = x.permute(0, 2, 1)

        # 最后一帧特征
        last_frame = x[:, -1, :]  # [batch, channels[-1]]

        # 全局平均池化
        global_avg = x.mean(dim=1)  # [batch, channels[-1]]

        # 融合
        fused = th.cat([last_frame, global_avg], dim=-1)  # [batch, channels[-1] * 2]
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
