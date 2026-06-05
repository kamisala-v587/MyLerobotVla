#!/usr/bin/env python

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
"""策略工厂：按名称创建 Policy 类、配置、pre/post processor 及模型实例。"""

from __future__ import annotations

import importlib
import logging
from typing import Any, TypedDict

import torch
from typing_extensions import Unpack

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.envs.configs import EnvConfig
from lerobot.envs.utils import env_to_policy_features
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.groot.configuration_groot import GrootConfig
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.policies.sac.reward_model.configuration_classifier import RewardClassifierConfig
from lerobot.policies.sarm.configuration_sarm import SARMConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.policies.utils import validate_visual_features_consistency
from lerobot.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.policies.wall_x.configuration_wall_x import WallXConfig
from lerobot.policies.xvla.configuration_xvla import XVLAConfig
from lerobot.processor import PolicyAction, PolicyProcessorPipeline
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    ACTION,
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def get_policy_class(name: str) -> type[PreTrainedPolicy]:
    """根据注册名返回 Policy 模型类（延迟 import，减少启动开销）。

    Args:
        name: 策略类型字符串，如 "tdmpc", "diffusion", "act", "vqbet",
              "pi0", "pi05", "sac", "reward_classifier", "smolvla", "wall_x" 等。

    Returns:
        对应的 ``*Policy`` 子类。

    Raises:
        ValueError: 名称未注册且无法通过插件动态解析时。
    """
    if name == "tdmpc":
        from lerobot.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    elif name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    elif name == "pi0_fast":
        from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPolicy

        return PI0FastPolicy
    elif name == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        return PI05Policy
    elif name == "sac":
        from lerobot.policies.sac.modeling_sac import SACPolicy

        return SACPolicy
    elif name == "reward_classifier":
        from lerobot.policies.sac.reward_model.modeling_classifier import Classifier

        return Classifier
    elif name == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        return SmolVLAPolicy
    elif name == "sarm":
        from lerobot.policies.sarm.modeling_sarm import SARMRewardModel

        return SARMRewardModel
    elif name == "groot":
        from lerobot.policies.groot.modeling_groot import GrootPolicy

        return GrootPolicy
    elif name == "xvla":
        from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

        return XVLAPolicy
    elif name == "wall_x":
        from lerobot.policies.wall_x.modeling_wall_x import WallXPolicy

        return WallXPolicy
    else:
        # 尝试从第三方插件动态加载（命名约定见 _get_policy_cls_from_policy_name）
        try:
            return _get_policy_cls_from_policy_name(name=name)
        except Exception as e:
            raise ValueError(f"Policy type '{name}' is not available.") from e


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    """根据类型字符串实例化对应的 ``*Config``（PreTrainedConfig 子类）。

    Args:
        policy_type: 如 "pi05", "act", "diffusion" 等。
        **kwargs: 传给配置类构造函数的字段。

    Returns:
        PreTrainedConfig 子类实例。

    Raises:
        ValueError: 类型未知时。
    """
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi05":
        return PI05Config(**kwargs)
    elif policy_type == "sac":
        return SACConfig(**kwargs)
    elif policy_type == "smolvla":
        return SmolVLAConfig(**kwargs)
    elif policy_type == "reward_classifier":
        return RewardClassifierConfig(**kwargs)
    elif policy_type == "groot":
        return GrootConfig(**kwargs)
    elif policy_type == "xvla":
        return XVLAConfig(**kwargs)
    elif policy_type == "wall_x":
        return WallXConfig(**kwargs)
    else:
        try:
            config_cls = PreTrainedConfig.get_choice_class(policy_type)
            return config_cls(**kwargs)
        except Exception as e:
            raise ValueError(f"Policy type '{policy_type}' is not available.") from e


class ProcessorConfigKwargs(TypedDict, total=False):
    """``make_pre_post_processors`` 可选关键字参数的类型提示。"""

    preprocessor_config_filename: str | None
    postprocessor_config_filename: str | None
    preprocessor_overrides: dict[str, Any] | None
    postprocessor_overrides: dict[str, Any] | None
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None


def make_pre_post_processors(
    policy_cfg: PreTrainedConfig,
    pretrained_path: str | None = None,
    **kwargs: Unpack[ProcessorConfigKwargs],
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """创建或加载策略的 preprocessor / postprocessor 流水线。

    - 提供 ``pretrained_path``：从 checkpoint 目录加载已保存的 pipeline JSON。
    - 否则：按 ``policy_cfg`` 类型调用各策略专属的 ``make_*_pre_post_processors``。

    Args:
        policy_cfg: 策略配置。
        pretrained_path: 预训练目录（含 processor 配置文件时优先加载）。
        **kwargs: 见 ``ProcessorConfigKwargs``（如 dataset_stats、overrides）。

    Returns:
        (preprocessor, postprocessor) 元组。

    Raises:
        ValueError: 该策略类型未实现 processor 工厂时。
    """
    if pretrained_path:
        # TODO(Steven): Groot 的 processor 需进一步完善
        if isinstance(policy_cfg, GrootConfig):
            # Groot 在 groot_pack_inputs_v3 中做归一化；需覆盖 stats 与 normalize_min_max
            preprocessor_overrides = {}
            postprocessor_overrides = {}
            preprocessor_overrides["groot_pack_inputs_v3"] = {
                "stats": kwargs.get("dataset_stats"),
                "normalize_min_max": True,
            }

            # 后处理：按环境 action 维度切片，并用数据集 stats 反归一化
            env_action_dim = policy_cfg.output_features[ACTION].shape[0]
            postprocessor_overrides["groot_action_unpack_unnormalize_v1"] = {
                "stats": kwargs.get("dataset_stats"),
                "normalize_min_max": True,
                "env_action_dim": env_action_dim,
            }
            kwargs["preprocessor_overrides"] = preprocessor_overrides
            kwargs["postprocessor_overrides"] = postprocessor_overrides

        return (
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "preprocessor_config_filename", f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("preprocessor_overrides", {}),
                to_transition=batch_to_transition,
                to_output=transition_to_batch,
            ),
            PolicyProcessorPipeline.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                config_filename=kwargs.get(
                    "postprocessor_config_filename", f"{POLICY_POSTPROCESSOR_DEFAULT_NAME}.json"
                ),
                overrides=kwargs.get("postprocessor_overrides", {}),
                to_transition=policy_action_to_transition,
                to_output=transition_to_policy_action,
            ),
        )

    # 从零构建：按配置类型分发到各策略的 processor 工厂
    if isinstance(policy_cfg, TDMPCConfig):
        from lerobot.policies.tdmpc.processor_tdmpc import make_tdmpc_pre_post_processors

        processors = make_tdmpc_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, DiffusionConfig):
        from lerobot.policies.diffusion.processor_diffusion import make_diffusion_pre_post_processors

        processors = make_diffusion_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, ACTConfig):
        from lerobot.policies.act.processor_act import make_act_pre_post_processors

        processors = make_act_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, VQBeTConfig):
        from lerobot.policies.vqbet.processor_vqbet import make_vqbet_pre_post_processors

        processors = make_vqbet_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI0Config):
        from lerobot.policies.pi0.processor_pi0 import make_pi0_pre_post_processors

        processors = make_pi0_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, PI05Config):
        from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

        processors = make_pi05_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SACConfig):
        from lerobot.policies.sac.processor_sac import make_sac_pre_post_processors

        processors = make_sac_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, RewardClassifierConfig):
        from lerobot.policies.sac.reward_model.processor_classifier import make_classifier_processor

        processors = make_classifier_processor(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SmolVLAConfig):
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        processors = make_smolvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, SARMConfig):
        from lerobot.policies.sarm.processor_sarm import make_sarm_pre_post_processors

        processors = make_sarm_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
            dataset_meta=kwargs.get("dataset_meta"),
        )
    elif isinstance(policy_cfg, GrootConfig):
        from lerobot.policies.groot.processor_groot import make_groot_pre_post_processors

        processors = make_groot_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, XVLAConfig):
        from lerobot.policies.xvla.processor_xvla import (
            make_xvla_pre_post_processors,
        )

        processors = make_xvla_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    elif isinstance(policy_cfg, WallXConfig):
        from lerobot.policies.wall_x.processor_wall_x import make_wall_x_pre_post_processors

        processors = make_wall_x_pre_post_processors(
            config=policy_cfg,
            dataset_stats=kwargs.get("dataset_stats"),
        )

    else:
        try:
            processors = _make_processors_from_policy_config(
                config=policy_cfg,
                dataset_stats=kwargs.get("dataset_stats"),
            )
        except Exception as e:
            raise ValueError(f"Processor for policy type '{policy_cfg.type}' is not implemented.") from e

    return processors


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    rename_map: dict[str, str] | None = None,
) -> PreTrainedPolicy:
    """实例化策略模型并放到 ``cfg.device``。

    流程概要：
    1. 从 ``ds_meta`` 或 ``env_cfg`` 推断 input/output feature 形状；
    2. 写回 ``cfg.input_features`` / ``cfg.output_features``；
    3. 若 ``cfg.pretrained_path`` 设置则 ``from_pretrained``，否则从零构造；
    4. 可选校验视觉特征与配置一致。

    Args:
        cfg: 策略配置；``pretrained_path`` 非空时加载权重。
        ds_meta: 数据集元信息（特征 + stats），训练时常用。
        env_cfg: 仿真环境配置，与 ds_meta 二选一。
        rename_map: 数据集/环境 key → policy 期望 key 的映射。

    Returns:
        已 ``.to(device)`` 的 PreTrainedPolicy 实例。

    Raises:
        ValueError: ds_meta 与 env_cfg 同时提供或均未提供；PEFT 无 checkpoint 等。
        NotImplementedError: 如不支持的 vqbet + mps 组合。
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # vqbet 在 MPS 上缺少 aten::unique_dim 等算子，需使用 cpu/cuda
    # TODO(aliberts, rcadene): 可抽成统一的 backend 兼容性检查
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta is not None:
        features = dataset_to_policy_features(ds_meta.features)
    else:
        if not cfg.pretrained_path:
            logging.warning(
                "从零初始化 policy，且特征来自环境而非数据集；"
                "若无 dataset stats，policy 内归一化层可能为无穷大默认值。"
            )
        if env_cfg is None:
            raise ValueError("env_cfg cannot be None when ds_meta is not provided")
        features = env_to_policy_features(env_cfg)

    cfg.output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    if not cfg.input_features:
        cfg.input_features = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    kwargs["config"] = cfg

    if ds_meta is not None and hasattr(ds_meta, "stats"):
        kwargs["dataset_stats"] = ds_meta.stats

    if ds_meta is not None:
        kwargs["dataset_meta"] = ds_meta

    if not cfg.pretrained_path and cfg.use_peft:
        raise ValueError(
            "Instantiating a policy with `use_peft=True` without a checkpoint is not supported since that requires "
            "the PEFT config parameters to be set. For training with PEFT, see `lerobot_train.py` on how to do that."
        )

    if cfg.pretrained_path and not cfg.use_peft:
        # 加载完整预训练权重；cfg 中推理相关超参仍可覆盖 checkpoint 内配置
        kwargs["pretrained_name_or_path"] = cfg.pretrained_path
        policy = policy_cls.from_pretrained(**kwargs)
    elif cfg.pretrained_path and cfg.use_peft:
        # pretrained_path 指向 adapter 目录；先读 PeftConfig 再加载 base + adapter
        from peft import PeftConfig, PeftModel

        logging.info("正在加载 policy 的 PEFT adapter")

        peft_pretrained_path = cfg.pretrained_path
        peft_config = PeftConfig.from_pretrained(peft_pretrained_path)

        kwargs["pretrained_name_or_path"] = peft_config.base_model_name_or_path
        if not kwargs["pretrained_name_or_path"]:
            raise ValueError(
                "No pretrained model name found in adapter config. Can't instantiate the pre-trained policy on which "
                "the adapter was trained."
            )

        policy = policy_cls.from_pretrained(**kwargs)
        policy = PeftModel.from_pretrained(policy, peft_pretrained_path, config=peft_config)

    else:
        policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, torch.nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    if not rename_map:
        validate_visual_features_consistency(cfg, features)
        # TODO: (jadechoghari) - 增加 check_state / check_action

    return policy


def _get_policy_cls_from_policy_name(name: str) -> type[PreTrainedConfig]:
    """从 ChoiceRegistry 注册名动态 import Policy 类（供第三方插件使用）。

    约定：``FooConfig`` → 模块 ``modeling_foo`` 中的 ``FooPolicy``。

    Args:
        name: ``PreTrainedConfig.register_subclass`` 注册的 type 字符串。

    Returns:
        Policy 模型类。
    """
    if name not in PreTrainedConfig.get_known_choices():
        raise ValueError(
            f"Unknown policy name '{name}'. Available policies: {PreTrainedConfig.get_known_choices()}"
        )

    config_cls = PreTrainedConfig.get_choice_class(name)
    config_cls_name = config_cls.__name__

    model_name = config_cls_name.removesuffix("Config")  # 如 DiffusionConfig -> Diffusion
    if model_name == config_cls_name:
        raise ValueError(
            f"The config class name '{config_cls_name}' does not follow the expected naming convention."
            f"Make sure it ends with 'Config'!"
        )
    cls_name = model_name + "Policy"  # 如 DiffusionPolicy
    module_path = config_cls.__module__.replace(
        "configuration_", "modeling_"
    )  # configuration_diffusion -> modeling_diffusion

    module = importlib.import_module(module_path)
    policy_cls = getattr(module, cls_name)
    return policy_cls


def _make_processors_from_policy_config(
    config: PreTrainedConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[Any, Any]:
    """按命名约定动态 import 第三方策略的 ``make_{type}_pre_post_processors``。

    模块路径：将 configuration_* 替换为 processor_*。

    Args:
        config: 策略配置。
        dataset_stats: 归一化用的数据集统计量。

    Returns:
        (preprocessor, postprocessor)。
    """

    policy_type = config.type
    function_name = f"make_{policy_type}_pre_post_processors"
    module_path = config.__class__.__module__.replace(
        "configuration_", "processor_"
    )
    logging.debug(
        "通过模块 '%s' 中的函数 '%s' 实例化 pre/post processors",
        module_path,
        function_name,
    )
    module = importlib.import_module(module_path)
    function = getattr(module, function_name)
    return function(config, dataset_stats=dataset_stats)
