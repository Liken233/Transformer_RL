from typing import Callable, Optional, List, Union, Dict, Type, Tuple
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.utils import get_device
from extractor.LSTMExtractor import LSTMExtractor, SeqFeaturesExtractor


class LSTMActorCriticPolicy(ActorCriticPolicy):
    """
    Actor 用 LSTM，Critic 用标准 MLP。

    关键设计：
    - Critic 只接收最后一帧（当前状态），避免维度错误
    - 同步 latent_dim_vf 确保 value_net 维度正确
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        lr_schedule: Callable[[float], float],
        seq_len: int = 8,
        feature_dim: int = None,
        latent_dim_pi: int = 256,
        latent_dim_vf: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = False,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        self.seq_len = seq_len
        self.feature_dim = feature_dim or (observation_space.shape[0] // seq_len)

        self.lstm_kwargs = dict(
            feature_dim=self.feature_dim,
            seq_len=seq_len,
            latent_dim_pi=latent_dim_pi,
            latent_dim_vf=latent_dim_vf,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            activation_fn=activation_fn,
        )

        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = SeqFeaturesExtractor
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = dict(
                seq_len=seq_len,
                feature_dim=self.feature_dim,
            )

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

    def _extract_last_frame(self, features: th.Tensor) -> th.Tensor:
        """
        从 [batch_size * seq_len, feature_dim] 中提取最后一帧 [batch_size, feature_dim]。
        Critic 只需要当前状态（最后一帧），不需要历史序列。
        """
        batch_total = features.shape[0]
        batch_size = batch_total // self.seq_len
        return features.view(batch_size, self.seq_len, self.feature_dim)[:, -1, :]

    def _build_mlp_extractor(self) -> None:
        """
        Actor: LSTM
        Critic: 标准 MLP（输入为最后一帧）
        """
        # Actor: LSTM
        self.mlp_extractor = LSTMExtractor(**self.lstm_kwargs)

        # Critic: 标准 MLP
        net_arch = self.net_arch
        if isinstance(net_arch, list):
            if len(net_arch) > 0 and isinstance(net_arch[0], dict):
                net_arch = net_arch[0]

        self.critic_mlp_extractor = MlpExtractor(
            feature_dim=self.features_dim,  # 最后一帧的维度 = feature_dim
            net_arch=net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

        # 同步 latent_dim_vf
        self.mlp_extractor.latent_dim_vf = self.critic_mlp_extractor.latent_dim_vf

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        """前向传播"""
        features = self.extract_features(obs)  # [batch * seq_len, feature_dim]

        # Actor: LSTM（使用完整序列）
        latent_pi = self.mlp_extractor.forward_actor(features)  # [batch, latent_dim_pi]

        # Critic: MLP（只使用最后一帧）
        features_last = self._extract_last_frame(features)  # [batch, feature_dim]
        latent_vf = self.critic_mlp_extractor.forward_critic(features_last)  # [batch, latent_dim_vf]

        values = self.value_net(latent_vf)  # [batch, 1]
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        """训练时评估"""
        features = self.extract_features(obs)

        latent_pi = self.mlp_extractor.forward_actor(features)
        features_last = self._extract_last_frame(features)
        latent_vf = self.critic_mlp_extractor.forward_critic(features_last)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        """预测价值"""
        features = self.extract_features(obs)
        features_last = self._extract_last_frame(features)
        latent_vf = self.critic_mlp_extractor.forward_critic(features_last)
        return self.value_net(latent_vf)
