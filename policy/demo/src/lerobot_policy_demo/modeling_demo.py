import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lerobot.policies.pretrained import PreTrainedPolicy

from .configuration_demo import DemoConfig


class DemoPolicy(PreTrainedPolicy):
    config_class = DemoConfig
    name = "demo_policy"

    def __init__(self, config: DemoConfig, **kwargs):
        super().__init__(config)
        action_dim = config.action_feature.shape[0]
        state_dim = config.input_features["observation.state"].shape[0]
        self.model = nn.Sequential(
            nn.Linear(state_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, action_dim),
        )

    def get_optim_params(self) -> dict:
        return self.parameters()

    def reset(self):
        pass

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        pred = self.model(batch["observation.state"])
        loss = F.mse_loss(pred, batch["action"])
        return loss, {"mse": loss.item()}

    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        single = self.model(batch["observation.state"])
        return single.unsqueeze(1)

    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        return self.model(batch["observation.state"])
