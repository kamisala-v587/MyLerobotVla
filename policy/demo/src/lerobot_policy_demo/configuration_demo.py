from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("demo_policy") # 注册策略类型名
@dataclass
class DemoConfig(PreTrainedConfig):
    """Configuration for DemoPolicy."""
    # 单步 demo：一次只预测 / 执行 1 步 action
    chunk_size: int = 1
    n_action_steps: int = 1
    hidden_dim: int = 256
    push_to_hub: bool = False # 是否推送到 hub
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

    # 单帧输入：不 stack 历史观测（与 notebook 里 dataset[0] 行为一致）
    @property
    def observation_delta_indices(self) -> None:
        return None

    # 单步 action：不取未来 action 序列
    @property
    def action_delta_indices(self) -> None:
        return None
    # RL 策略用；模仿学习通常返回 `None`
    @property
    def reward_delta_indices(self) -> None: 
        return None
