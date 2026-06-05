# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""预训练策略基类：权重加载/保存、Hub 上传、PEFT 包装等通用逻辑。"""

import abc
import builtins
import dataclasses
import logging
import os
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, TypeVar

import packaging
import safetensors
from huggingface_hub import HfApi, ModelCard, ModelCardData, hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor, save_model as save_model_as_safetensor
from torch import Tensor, nn
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.utils import log_model_loading_keys
from lerobot.utils.hub import HubMixin

T = TypeVar("T", bound="PreTrainedPolicy")


class ActionSelectKwargs(TypedDict, total=False):
    """推理时可选参数（如 flow matching 的初始噪声）。"""

    noise: Tensor | None


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """所有 LeRobot 策略模型的抽象基类。

    子类必须定义 ``config_class`` 与 ``name``，并实现 forward / select_action 等接口。
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        """保存 config.json 与 model.safetensors 到目录。"""
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: builtins.type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """从本地目录或 Hugging Face Hub 加载策略权重。

        默认以 ``eval()`` 模式返回；训练前需调用 ``policy.train()``。

        流程：解析 config → ``cls(config)`` 构建结构 → 加载 safetensors → ``to(device)``。
        若传入 ``config``，则超参以该对象为准，仅加载权重。
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: str, map_location: str, strict: bool) -> T:
        """将 safetensors 权重载入已实例化的模型，并记录 missing/unexpected keys。"""
        kwargs = {"strict": strict}

        # safetensors >= 0.4.3 支持直接指定 device，避免先 CPU 再拷贝
        if packaging.version.parse(safetensors.__version__) >= packaging.version.parse("0.4.3"):
            kwargs["device"] = map_location

        missing_keys, unexpected_keys = load_model_as_safetensor(model, model_file, **kwargs)
        log_model_loading_keys(missing_keys, unexpected_keys)

        if "device" not in kwargs and map_location != "cpu":
            logging.warning(
                "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                " This means that the model is loaded on 'cpu' first and then copied to the device."
                " This leads to a slower loading time."
                " Please update safetensors to version 0.4.3 or above for improved performance."
            )
            model.to(map_location)
        return model

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """返回应交给优化器的参数（可为全部参数或参数组 dict）。"""
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """环境 reset 时调用，清空 action 队列等内部缓存。"""
        raise NotImplementedError

    # TODO(aliberts, rcadene): 拆分为 forward 与 compute_loss？
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """训练一步的前向，返回 (loss, 可日志化的 output_dict)。"""
        raise NotImplementedError

    @abc.abstractmethod
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """预测一整段 action chunk（action chunking 策略）；``select_action`` 会缓存其中逐步执行。"""
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        """根据观测返回单步 action（可含队列/cache 逻辑）。"""
        raise NotImplementedError

    def push_model_to_hub(
        self,
        cfg: TrainPipelineConfig,
        peft_model=None,
    ):
        """将权重、train_config、README 一并上传到 Hugging Face Hub。"""
        api = HfApi()
        repo_id = api.create_repo(
            repo_id=self.config.repo_id, private=self.config.private, exist_ok=True
        ).repo_id

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            saved_path = Path(tmp) / repo_id

            if peft_model is not None:
                # PEFT 场景：adapter 由 peft_model 保存，policy config 需自行写入
                peft_model.save_pretrained(saved_path)
                self.config.save_pretrained(saved_path)
            else:
                self.save_pretrained(saved_path)

            card = self.generate_model_card(
                cfg.dataset.repo_id, self.config.type, self.config.license, self.config.tags
            )
            card.save(str(saved_path / "README.md"))

            cfg.save_pretrained(saved_path)

            commit_info = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=saved_path,
                commit_message="Upload policy weights, train config and readme",
                allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
                ignore_patterns=["*.tmp", "*.log"],
            )

            logging.info(f"Model pushed to {commit_info.repo_url.url}")

    def generate_model_card(
        self, dataset_repo_id: str, model_type: str, license: str | None, tags: list[str] | None
    ) -> ModelCard:
        """根据模板生成 Model Card。"""
        base_model = "lerobot/smolvla_base" if model_type == "smolvla" else None

        card_data = ModelCardData(
            license=license or "apache-2.0",
            library_name="lerobot",
            pipeline_tag="robotics",
            tags=list(set(tags or []).union({"robotics", "lerobot", model_type})),
            model_name=model_type,
            datasets=dataset_repo_id,
            base_model=base_model,
        )

        template_card = (
            files("lerobot.templates").joinpath("lerobot_modelcard_template.md").read_text(encoding="utf-8")
        )
        card = ModelCard.from_template(card_data, template_str=template_card)
        card.validate()
        return card

    def wrap_with_peft(
        self,
        peft_config=None,
        peft_cli_overrides: dict | None = None,
    ) -> "PreTrainedPolicy":
        """用 PEFT（如 LoRA）包装当前策略，实现参数高效微调。

        子类可覆写 ``_get_default_peft_targets`` 提供默认 target_modules，
        ``_validate_peft_config`` 做策略特有校验。

        Args:
            peft_config: 完整 PEFT 配置；若提供则直接使用（仍可被 CLI overrides 覆盖）。
            peft_cli_overrides: CLI 中的 peft 字段（method_type、target_modules、r 等）。
        """
        from peft import get_peft_model

        if peft_config is not None:
            final_config = peft_config
            if peft_cli_overrides:
                final_config = self._apply_peft_cli_overrides(final_config, peft_cli_overrides)
        else:
            final_config = self._build_peft_config(peft_cli_overrides or {})

        self._validate_peft_config(final_config)

        # 冻结基座，仅训练 adapter
        for p in self.parameters():
            p.requires_grad_(False)

        if self.config.pretrained_path:
            self.name_or_path = str(self.config.pretrained_path)

        peft_model = get_peft_model(self, final_config)
        peft_model.config.use_peft = True

        logging.info(f"Wrapped {self.name} with PEFT ({type(final_config).__name__})")
        return peft_model

    def _get_default_peft_targets(self) -> dict[str, any] | None:
        """子类返回默认 PEFT target_modules 等（与具体 PEFT 方法无关）。"""
        return None

    def _validate_peft_config(self, peft_config) -> None:
        """校验 PEFT 配置；默认要求存在 ``pretrained_path``。"""
        if not self.config.pretrained_path:
            raise ValueError(
                "Training from scratch using PEFT is unlikely to yield good results. "
                "Supply a `policy.pretrained_path` to fine-tune an existing model."
            )

    def _preprocess_peft_cli_overrides(self, cli_overrides: dict, peft_method_type) -> dict:
        """预处理 CLI overrides：字段重命名、按 PEFT 方法映射 init_type。"""
        from peft import PeftType

        cli_overrides = cli_overrides.copy()

        if "full_training_modules" in cli_overrides:
            cli_overrides["modules_to_save"] = cli_overrides.pop("full_training_modules")

        cli_overrides.pop("method_type", None)

        init_type = cli_overrides.pop("init_type", None)
        if init_type is not None:
            if peft_method_type == PeftType.LORA:
                cli_overrides["init_lora_weights"] = init_type
            elif peft_method_type == PeftType.MISS:
                cli_overrides["init_weights"] = init_type
            else:
                raise ValueError(f"Init type '{init_type}' unknown for PEFT method {peft_method_type}.")

        return cli_overrides

    def _build_peft_config(self, cli_overrides: dict):
        """由策略默认 + CLI overrides 构建 PEFT 配置对象。"""
        from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType

        method_type_str = cli_overrides.get("method_type") or "lora"
        peft_method_type = PeftType[method_type_str.upper()]
        peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]

        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        config_dict = dict(self._get_default_peft_targets() or {})
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        if not config_dict.get("target_modules"):
            raise ValueError(
                f"Policy '{self.name}' does not define default target_modules. "
                "Please pass --peft.target_modules explicitly."
            )

        return peft_config_cls(**config_dict)

    def _apply_peft_cli_overrides(self, peft_config, cli_overrides: dict):
        """在已有 PEFT config 上应用 CLI overrides 并重建配置对象。"""
        from peft import PEFT_TYPE_TO_CONFIG_MAPPING, PeftType

        method_type_str = cli_overrides.get("method_type")
        if method_type_str:
            peft_method_type = PeftType[method_type_str.upper()]
            peft_config_cls = PEFT_TYPE_TO_CONFIG_MAPPING[peft_method_type]
        else:
            peft_method_type = PeftType(peft_config.peft_type)
            peft_config_cls = type(peft_config)

        cli_overrides = self._preprocess_peft_cli_overrides(cli_overrides, peft_method_type)

        config_dict = {k: v for k, v in dataclasses.asdict(peft_config).items() if not k.startswith("_")}
        for key, value in cli_overrides.items():
            if value is not None:
                config_dict[key] = value

        return peft_config_cls(**config_dict)
