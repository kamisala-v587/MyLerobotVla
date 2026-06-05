from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("my_policy") # 注册策略类型名
@dataclass
class MyPolicyConfig(PreTrainedConfig):
    """Configuration for MyPolicyPolicy."""
    # --- 策略特有超参 ---
    chunk_size: int = 50  # 一次预测多少步 action
    n_action_steps: int = 50 # 实际执行多少步（≤ chunk_size）
    hidden_dim: int = 256 # 网络超参（示例）

    # --- 归一化方式 ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    # 调用 super().__post_init__()，做参数校验 -- 推荐有
    def __post_init__(self):
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than chunk_size ({self.chunk_size})."
            )
    # 校验/补全 input/output features -- 必备
    def validate_features(self) -> None:
        # 补全 input/output features -- 兜底策略
        if OBS_STATE not in self.input_features:
            self.input_features[OBS_STATE] = PolicyFeature(
                type=FeatureType.STATE,
                shape=(7,),
            )
        if ACTION not in self.output_features:
            self.output_features[ACTION] = PolicyFeature(
                type=FeatureType.ACTION,
                shape=(7,),
            )
    # 训练用优化器默认配置 -- 必备
    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=1e-4, weight_decay=1e-4, grad_clip_norm=10.0)


    # 训练用学习率调度器默认配置 -- 必备
    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=1e-4,
            decay_lr=1e-5,
            num_warmup_steps=500,
            num_decay_steps=50_000,
        )

    # 训练时取哪些历史观测帧 -- 必备
    @property
    def observation_delta_indices(self) -> list[int] | None:
        return list(range(1 - self.n_obs_steps, 1)) # openpi通常只取当前帧

    # 训练时取哪些未来 action 帧 -- 必备
    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size)) # openpi通常只取当前帧
        
    # RL 策略用；模仿学习通常返回 `None`
    @property
    def reward_delta_indices(self) -> None: 
        return None
