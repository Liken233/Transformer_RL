from typing import Callable, Optional, List, Union, Dict, Type, Tuple
import gymnasium as gym
import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.utils import get_device
from extractor.TransformerExtractorAblation import TransformerExtractorAblation, SeqFeaturesExtractor


class TransformerAblationPolicy(ActorCriticPolicy):
    """
    支持消融实验的 Transformer Actor-Critic Policy。
    新增参数：
    - pos_type: str
    - causal_mask: bool
    - init_type: str
    - warmup_steps: int (在训练脚本中处理)
    - critic_type 扩展支持 "partial_shared"
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
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.1,
        activation_fn: Type[nn.Module] = nn.ReLU,
        # 消融参数
        pos_type: str = "none",
        causal_mask: bool = False,
        fusion_mode: str = "concat",
        critic_type: str = "mlp",
        feature_mask: Optional[List[int]] = None,
        init_type: str = "xavier",    # "xavier" or "orthogonal"
        *args,
        **kwargs,
    ):
        self.seq_len = seq_len
        self.feature_dim = feature_dim or (observation_space.shape[0] // seq_len)
        self.actual_feature_dim = len(feature_mask) if feature_mask else self.feature_dim
        self.critic_type = critic_type
        self.feature_mask = feature_mask
        self.pos_type = pos_type
        self.causal_mask = causal_mask
        self.fusion_mode = fusion_mode
        self.init_type = init_type

        self.transformer_kwargs = dict(
            feature_dim=self.actual_feature_dim,
            seq_len=seq_len,
            latent_dim_pi=latent_dim_pi,
            latent_dim_vf=latent_dim_vf,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation_fn=activation_fn,
            pos_type=pos_type,
            causal_mask=causal_mask,
            fusion_mode=fusion_mode,
        )

        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = SeqFeaturesExtractor
        if "features_extractor_kwargs" not in kwargs:
            kwargs["features_extractor_kwargs"] = dict(
                seq_len=seq_len,
                feature_dim=self.feature_dim,
                feature_mask=feature_mask,
            )

        super().__init__(observation_space, action_space, lr_schedule, *args, **kwargs)

        # 自定义初始化
        self._init_weights(init_type)

    def _init_weights(self, init_type: str):
        """对网络参数进行初始化"""
        def init_fn(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                if init_type == "orthogonal":
                    nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
                else:  # xavier
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(init_fn)

    def _extract_last_frame(self, features: th.Tensor) -> th.Tensor:
        batch_total = features.shape[0]
        batch_size = batch_total // self.seq_len
        return features.view(batch_size, self.seq_len, self.actual_feature_dim)[:, -1, :]

    def _build_mlp_extractor(self) -> None:
        # Actor
        self.mlp_extractor = TransformerExtractorAblation(**self.transformer_kwargs)

        # Critic
        net_arch = self.net_arch
        if isinstance(net_arch, list) and len(net_arch) > 0 and isinstance(net_arch[0], dict):
            net_arch = net_arch[0]

        if self.critic_type == "mlp":
            self.critic_mlp_extractor = MlpExtractor(
                feature_dim=self.features_dim,
                net_arch=net_arch,
                activation_fn=self.activation_fn,
                device=self.device,
            )
        elif self.critic_type == "shared":
            self.critic_mlp_extractor = None
        elif self.critic_type == "transformer":
            critic_kwargs = self.transformer_kwargs.copy()
            critic_kwargs["latent_dim_pi"] = self.transformer_kwargs["latent_dim_vf"]
            self.critic_transformer = TransformerExtractorAblation(**critic_kwargs)
            self.critic_mlp_extractor = None
        elif self.critic_type == "partial_shared":
            # 部分共享：前2层共享，后2层独立（此处简化，实际可构造两个Transformer）
            # 为简化，我们复用 mlp_extractor 的 transformer，但单独用最后一帧做 value
            # 但为了区分，这里使用与 shared 相同的方式（但可改）
            self.critic_mlp_extractor = None
        else:
            raise ValueError(f"Unknown critic_type: {self.critic_type}")

        if self.critic_type == "mlp":
            self.mlp_extractor.latent_dim_vf = self.critic_mlp_extractor.latent_dim_vf
        else:
            self.mlp_extractor.latent_dim_vf = self.transformer_kwargs["latent_dim_vf"]

    def forward(self, obs: th.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)

        if self.critic_type == "mlp":
            features_last = self._extract_last_frame(features)
            latent_vf = self.critic_mlp_extractor.forward_critic(features_last)
        elif self.critic_type in ["shared", "partial_shared"]:
            _, latent_vf = self.mlp_extractor.forward(features)
        elif self.critic_type == "transformer":
            _, latent_vf = self.critic_transformer.forward(features)
        else:
            raise ValueError(f"Unknown critic_type: {self.critic_type}")

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor):
        features = self.extract_features(obs)
        latent_pi = self.mlp_extractor.forward_actor(features)

        if self.critic_type == "mlp":
            features_last = self._extract_last_frame(features)
            latent_vf = self.critic_mlp_extractor.forward_critic(features_last)
        elif self.critic_type in ["shared", "partial_shared"]:
            _, latent_vf = self.mlp_extractor.forward(features)
        elif self.critic_type == "transformer":
            _, latent_vf = self.critic_transformer.forward(features)
        else:
            raise ValueError(f"Unknown critic_type: {self.critic_type}")

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(latent_vf)
        entropy = distribution.entropy()
        return values, log_prob, entropy

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features = self.extract_features(obs)

        if self.critic_type == "mlp":
            features_last = self._extract_last_frame(features)
            latent_vf = self.critic_mlp_extractor.forward_critic(features_last)
        elif self.critic_type in ["shared", "partial_shared"]:
            _, latent_vf = self.mlp_extractor.forward(features)
        elif self.critic_type == "transformer":
            _, latent_vf = self.critic_transformer.forward(features)
        else:
            raise ValueError(f"Unknown critic_type: {self.critic_type}")

        return self.value_net(latent_vf)