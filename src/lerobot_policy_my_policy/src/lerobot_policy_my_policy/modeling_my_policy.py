import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_my_policy import MyPolicyConfig


class MyPolicyPolicy(PreTrainedPolicy): # 实现 5 个抽象方法 -- 必备
    config_class = MyPolicyConfig # 指定配置类  PreTrainedPolicy继承基本属性
    name = "my_policy" # 指定策略类型名 PreTrainedPolicy 继承基本属性

    def __init__(self, config: MyPolicyConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        # 获取配置维度,初始化self.model
        action_dim = config.action_feature.shape[0]
        state_dim = config.input_features["observation.state"].shape[0] 
        self.model = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, action_dim),
        )
        self._action_queue: list[Tensor] | None = None

    # 构建 optimizer
    def get_optim_params(self) -> dict:
        return self.parameters()

    # 每个 episode 开始时清空 action 队列
    def reset(self):
        self._action_queue = None

    # 训练：返回 (loss, metrics_dict)
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        pred = self.model(batch["observation.state"])
        loss = F.mse_loss(pred, batch["action"])
        return loss, {"mse": loss.item()}

    # 推理：预测一整段 action chunk，shape [B, chunk_size, action_dim]
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        state = batch["observation.state"]
        single = self.model(state)
        return single.unsqueeze(1).expand(-1, self.config.chunk_size, -1)

    # 推理：返回单步 action（可内部维护 chunk 队列）
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        if self._action_queue is None or len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(batch)
            self._action_queue = [chunk[:, i] for i in range(self.config.n_action_steps)]
        return self._action_queue.pop(0)
