#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot PI05 Policy Server（定制精简版）
========================================

用途
----
启动一个 WebSocket 推理服务，加载 LeRobot 格式的 PI05 checkpoint，对外提供
``infer(obs_dict) -> action_dict`` 能力。

本脚本刻意保持精简：只支持 ``config.type == "pi05"`` 的 checkpoint，不包含
TBot_SA1 / RTC / PEFT 等扩展逻辑。

LeRobot 中「数据处理」分三层（重要）
------------------------------------
1. **config.json（策略配置）**
   - 定义模型结构与推理超参：``input_features`` / ``output_features``、
     ``chunk_size``、``n_action_steps``、``max_state_dim``、
     ``normalization_mapping``、``image_resolution`` 等。
   - 若客户端观测 key / 维度与训练时不一致，通常需要改这里（并重新训练或
     至少保证 preprocessor 的 feature 定义一致）。

2. **policy_preprocessor.json + policy_postprocessor.json（数据流水线）**
   - 训练结束时与 checkpoint 一起保存，描述「进模型前 / 出模型后」的处理步骤。
   - PI05 典型 preprocessor 顺序：
     rename → batch → normalize(state/action) → state token 拼接 → PaliGemma tokenizer → device
   - postprocessor：unnormalize(action) → 搬回 CPU
   - 归一化统计量保存在同目录的 ``*.safetensors``（如 quantile stats），
     **不是** 写在 config.json 里。
   - 若仅环境 camera 命名不同，优先改 ``policy_preprocessor.json`` 里
     ``rename_observations_processor.rename_map``，或通过本脚本 ``--rename_map`` 覆盖。

3. **PI05Policy 模型内部（modeling_pi05.py）**
   - ``_preprocess_images``：resize/pad 到 ``config.image_resolution``，[0,1]→[-1,1]
   - ``predict_action_chunk``：flow matching 采样，输出 ``[B, chunk_size, action_dim]``
   - 这部分一般不需要改 server；改图像尺寸 / action 维数应回到 config.json。

推荐 checkpoint 目录结构
------------------------
可直接传入以下任一目录（脚本会自动解析）：

- ``.../checkpoints/010000/pretrained_model/``   （含 config.json）
- ``.../checkpoints/010000/``                   （自动下钻到 pretrained_model）

启动示例（离线环境，与训练时一致）
------------------------------------
PI05 的 preprocessor 需要 **PaliGemma tokenizer**（``google/paligemma-3b-pt-224``）。
训练若已设置 ``HF_HUB_OFFLINE=1``，说明 tokenizer 应在本地 HF cache 或
``/vla/.models/paligemma-3b-pt-224`` 中，server 同样需要离线加载。

.. code-block:: bash

   cd /vla/my_vla
   conda activate myvla
   export HF_HUB_OFFLINE=1
   export TRANSFORMERS_OFFLINE=1
   export TOKENIZERS_PARALLELISM=false
   # 若 tokenizer 不在默认 cache，可显式指定：
   # export PALIGEMMA_TOKENIZER_PATH=/path/to/paligemma-3b-pt-224

   python /vla/my_vla/scripts/serve_pi05_policy.py \\
       --ckpt_path /vla/my_vla/policy/checkpoints/.../pretrained_model \\
       --host 0.0.0.0 \\
       --port 8000 \\
       --default_prompt "adjust the bottle"

说明：你 my_tbot 训练用的是 **TBot_SA1 + Qwen3-VL tokenizer**（本地路径
``/vla/.models/Qwen3-VL-2B-Instruct``）；本 PI05 server 用的是 **PaliGemma
tokenizer**，二者不是同一个文件。

客户端请求格式（推荐：LeRobot frame，等同 ``ds[i]``）
------------------------------------------------
与本地 notebook 推理一致：**键名对齐即可**，normalize / batch / tokenize 全由 server 的
``preprocess(frame)`` 完成。客户端只需把 ``LeRobotDataset[i]`` 序列化后发来（tensor→numpy）。

.. code-block:: python

   # 等价于 preprocess(ds[0]) 的输入
   payload = dict(ds[0])          # 含 observation.* / task / action 等
   payload["timestep"] = 0
   payload["reset"] = True
   # msgpack 发送；server 收到后 numpy→tensor，再 preprocess(frame)

若数据集 key 与 checkpoint 不一致，在 server 启动时用 ``--rename_map`` 配置
（与 ``RenameObservationsProcessorStep`` 相同，old_key → new_key）。

仍兼容旧版紧凑格式 ``{task, state, images:{cam_high:...}}``，但不推荐。

服务端响应（action_dict）
-------------------------
.. code-block:: python

   {
       "actions": np.ndarray([T, A]),        # T = infer_horizon, A = action 维（如 14）
       "action": np.ndarray([A]),            # 第一步 action
       "model_actions": np.ndarray([T, A]),  # 反归一化前的模型输出（调试用）
   }

依赖
----
- lerobot（本仓库 ``my_vla/src/lerobot``）
- torch, websockets, msgpack, numpy
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import json
import logging
import os
import socket
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import torch
import websockets
import websockets.asyncio.server as websocket_server
import websockets.frames

# ---------------------------------------------------------------------------
# 路径：把本地 lerobot 源码加入 PYTHONPATH（无需 pip install -e）
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MY_VLA_ROOT = SCRIPT_DIR.parent
LEROBOT_SRC = MY_VLA_ROOT / "src" / "lerobot" / "src"
for path in (SCRIPT_DIR, LEROBOT_SRC, MY_VLA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# 与训练脚本保持一致：默认离线，避免启动时访问 huggingface.co
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# PI05 preprocessor 里写死的 tokenizer repo id（见 policy_preprocessor.json）
DEFAULT_PALIGEMMA_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
DEFAULT_LOCAL_TOKENIZER_CANDIDATES = (
    Path("/vla/.models/paligemma-3b-pt-224"),
    Path("/vla/.models/google/paligemma-3b-pt-224"),
)

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402
from lerobot.processor import PolicyAction  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

# WebSocket 控制字段，不属于 LeRobot frame
CONTROL_KEYS = frozenset({"reset", "timestep", "prompt"})


# ---------------------------------------------------------------------------
# msgpack + numpy 序列化（WebSocket 传输用，与 openpi / my_tbot 示例兼容）
# ---------------------------------------------------------------------------
def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype for msgpack: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


MsgpackPacker = functools.partial(msgpack.Packer, default=_pack_array)
msgpack_pack = functools.partial(msgpack.packb, default=_pack_array)
msgpack_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# CLI 参数
# ---------------------------------------------------------------------------
@dataclass
class ServeArgs:
    ckpt_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str = "Execute the task."
    device: str = "auto"
    dtype: str = "auto"  # auto | float32 | bfloat16
    infer_horizon: int | None = None  # 覆盖 config.n_action_steps，控制每次返回多少步 action
    rename_map: dict[str, str] | None = None  # 覆盖 policy_preprocessor 中的 rename_map
    tokenizer_path: str | None = None  # PaliGemma tokenizer 本地目录（离线必需）


def parse_args() -> ServeArgs:
    parser = argparse.ArgumentParser(
        description="启动 LeRobot PI05 checkpoint 的 WebSocket 推理服务（定制精简版）。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt_path",
        required=True,
        help="checkpoint 目录：pretrained_model 目录，或其上一级 step 目录。",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--default_prompt",
        default="Execute the task.",
        help="客户端未提供 task/prompt 时的默认语言指令。",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备：auto / cpu / cuda / cuda:0 等。",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16"],
        default="auto",
        help="推理 dtype；auto 时使用 checkpoint config.dtype。",
    )
    parser.add_argument(
        "--infer_horizon",
        type=int,
        default=None,
        help="每次 infer 返回的 action 步数；默认读取 config.n_action_steps。",
    )
    parser.add_argument(
        "--rename_map",
        default=None,
        help='JSON 字符串，环境 key → policy key，例如 \'{"head":"observation.images.cam_high"}\'。',
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help=(
            "PaliGemma tokenizer 本地目录。"
            "也可设环境变量 PALIGEMMA_TOKENIZER_PATH / TOKENIZER_PATH。"
            "未指定时自动查找 /vla/.models/... 与 HF cache。"
        ),
    )
    parsed = parser.parse_args()
    rename_map = None
    if parsed.rename_map:
        rename_map = json.loads(parsed.rename_map)
        if not isinstance(rename_map, dict):
            raise ValueError("--rename_map 必须是 JSON object")
    tokenizer_path = parsed.tokenizer_path or os.environ.get("PALIGEMMA_TOKENIZER_PATH")
    tokenizer_path = tokenizer_path or os.environ.get("TOKENIZER_PATH")
    return ServeArgs(
        ckpt_path=parsed.ckpt_path,
        host=parsed.host,
        port=parsed.port,
        default_prompt=parsed.default_prompt,
        device=parsed.device,
        dtype=parsed.dtype,
        infer_horizon=parsed.infer_horizon,
        rename_map=rename_map,
        tokenizer_path=tokenizer_path,
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def resolve_ckpt_dir(ckpt_path: str | Path) -> Path:
    """解析 checkpoint 根目录（必须含 config.json）。"""
    ckpt_dir = Path(ckpt_path).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint 路径不存在: {ckpt_dir}")
    if (ckpt_dir / "config.json").is_file():
        return ckpt_dir
    pretrained_dir = ckpt_dir / "pretrained_model"
    if (pretrained_dir / "config.json").is_file():
        return pretrained_dir
    raise FileNotFoundError(
        f"在 {ckpt_dir} 或 {pretrained_dir} 下未找到 config.json。"
        "请传入包含 pretrained_model 的 step 目录，或直接传入 pretrained_model 目录。"
    )


def _looks_like_tokenizer_dir(path: Path) -> bool:
    """判断目录是否像 HuggingFace tokenizer 快照。"""
    return path.is_dir() and (
        (path / "tokenizer_config.json").is_file() or (path / "tokenizer.json").is_file()
    )


def _iter_hf_hub_roots() -> list[Path]:
    """收集可能的 HuggingFace Hub cache 根目录。"""
    roots: list[Path] = []
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    # 部分机器训练 cache 不在 $HOME 下，补充常见路径
    roots.append(Path("/mnt/workspace/luyi/.cache/huggingface/hub"))
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _find_tokenizer_in_hf_cache(repo_id: str) -> Path | None:
    """在 HF cache 中查找 repo 对应的 tokenizer 快照目录。"""
    repo_folder = "models--" + repo_id.replace("/", "--")
    for hub_root in _iter_hf_hub_roots():
        snapshots_dir = hub_root / repo_folder / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snapshot_dir in sorted(snapshots_dir.iterdir()):
            if _looks_like_tokenizer_dir(snapshot_dir):
                return snapshot_dir.resolve()
    return None


def resolve_tokenizer_path(explicit_path: str | None, repo_id: str = DEFAULT_PALIGEMMA_TOKENIZER_REPO) -> Path:
    """
    解析 PaliGemma tokenizer 本地路径（离线部署必需）。

    优先级：
    1. --tokenizer_path / PALIGEMMA_TOKENIZER_PATH / TOKENIZER_PATH
    2. /vla/.models/paligemma-3b-pt-224 等固定候选
    3. HuggingFace cache 中的 snapshots（训练时 ``HF_HUB_OFFLINE=1`` 通常走这里）
    """
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    candidates.extend(DEFAULT_LOCAL_TOKENIZER_CANDIDATES)

    checked: list[str] = []
    for candidate in candidates:
        checked.append(str(candidate))
        if _looks_like_tokenizer_dir(candidate):
            return candidate.resolve()

    cached = _find_tokenizer_in_hf_cache(repo_id)
    if cached is not None:
        logging.info("从 HF cache 找到 tokenizer: %s", cached)
        return cached

    raise FileNotFoundError(
        "离线模式下找不到 PaliGemma tokenizer。\n"
        f"  repo_id: {repo_id}\n"
        f"  已检查: {checked}\n"
        f"  HF cache 根目录: {[str(p) for p in _iter_hf_hub_roots()]}\n"
        "解决办法（任选其一）：\n"
        "  1) 把 tokenizer 目录放到 /vla/.models/paligemma-3b-pt-224\n"
        "  2) export PALIGEMMA_TOKENIZER_PATH=/path/to/tokenizer_dir\n"
        "  3) export HF_HOME=你的训练机 cache 路径（含 models--google--paligemma-3b-pt-224）"
    )


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_runtime_dtype(dtype_arg: str, config_dtype: str | None, device: str) -> torch.dtype:
    if dtype_arg == "auto":
        name = (config_dtype or "float32").lower()
    else:
        name = dtype_arg
    if name in {"bf16", "bfloat16"}:
        if device == "cpu":
            logging.warning("bfloat16 在 CPU 上可能不支持，回退到 float32")
            return torch.float32
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"不支持的 dtype: {name}")


# ---------------------------------------------------------------------------
# PI05 推理封装
# ---------------------------------------------------------------------------
class LeRobotPI05Policy:
    """
    将 LeRobot 官方组件（Policy + Pre/Post Processor）包装成 ``infer(obs)`` 接口。

    数据流（与本地 ``preprocess(ds[0]); policy.select_action(batch)`` 一致）
    --------------------------------------------------------------------
    client payload（LeRobot frame，msgpack 传输后 numpy）
      -> _payload_to_frame()      # 仅 numpy→tensor，保留 ds[i] 键名
      -> preprocessor(frame)      # rename / batch / normalize / tokenize / device
      -> policy.predict_action_chunk
      -> postprocessor            # unnormalize
      -> numpy actions
    """

    def __init__(self, args: ServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)

        # 1) 读取 config.json —— 策略结构与 feature 定义的来源
        self.config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        if self.config.type != "pi05":
            raise ValueError(
                f"本 server 仅支持 pi05，当前 checkpoint type={self.config.type!r}。"
            )

        self.device = resolve_device(args.device)
        self.config.device = self.device
        # 推理默认 float32（与本地 notebook 一致）；checkpoint 可为 bf16 权重，fp32 推理更稳
        self.runtime_dtype = resolve_runtime_dtype(
            args.dtype if args.dtype != "auto" else "float32",
            getattr(self.config, "dtype", None),
            self.device,
        )
        if args.dtype == "auto" and str(getattr(self.config, "dtype", "")).lower() in {"bfloat16", "bf16"}:
            logging.info(
                "推理 dtype 使用 float32（checkpoint 训练为 %s；可通过 --dtype bfloat16 强制）",
                self.config.dtype,
            )

        # infer_horizon：控制每次返回多少步 action（<= chunk_size）
        chunk_size = int(getattr(self.config, "chunk_size", 50))
        default_horizon = int(getattr(self.config, "n_action_steps", chunk_size))
        self.infer_horizon = int(args.infer_horizon or default_horizon)
        self.infer_horizon = max(1, min(self.infer_horizon, chunk_size))

        # 2) 加载 PI05Policy 权重（config.json + model.safetensors）
        logging.info("正在加载 PI05Policy: %s", self.ckpt_dir)
        self.policy = PI05Policy.from_pretrained(self.ckpt_dir, config=self.config)
        self.policy.config.device = self.device
        self.policy.to(device=self.device, dtype=self.runtime_dtype)
        self.policy.eval()

        # 3) 加载 preprocessor / postprocessor
        #    优先从 checkpoint 读取 JSON + safetensors（训练时保存的 stats）
        #    rename_map 可通过 CLI 覆盖 policy_preprocessor.json 中的配置
        #    tokenizer 必须本地加载（policy_preprocessor.json 里写的是 HF repo id）
        rename_override = args.rename_map or {}
        self.tokenizer_path = resolve_tokenizer_path(args.tokenizer_path)
        logging.info("使用本地 PaliGemma tokenizer: %s", self.tokenizer_path)
        runtime_dtype_name = self._runtime_dtype_for_processor(self.runtime_dtype)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.ckpt_dir),
            preprocessor_overrides={
                "device_processor": {
                    "device": self.device,
                    "float_dtype": runtime_dtype_name,
                },
                "rename_observations_processor": {"rename_map": rename_override},
                "tokenizer_processor": {"tokenizer_name": str(self.tokenizer_path)},
            },
            postprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )

        # 从 config.json 读取期望的输入 key（部署时应与客户端对齐）
        self.expected_image_keys = list(self.policy.config.image_features.keys())
        self.expected_state_dim = int(self.policy.config.input_features[OBS_STATE].shape[0])
        self.expected_action_dim = int(self.policy.config.output_features[ACTION].shape[0])

        self._metadata = {
            "model_type": "pi05",
            "deployment": "lerobot_pi05_custom_server",
            "checkpoint_dir": str(self.ckpt_dir),
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "chunk_size": chunk_size,
            "default_prompt": args.default_prompt,
            "expected_image_keys": self.expected_image_keys,
            "expected_state_dim": self.expected_state_dim,
            "expected_action_dim": self.expected_action_dim,
            "rename_map_override": rename_override,
            "tokenizer_path": str(self.tokenizer_path),
            "notes": {
                "config_json": "模型结构 / feature 形状 / 推理超参",
                "policy_preprocessor_json": "进模型前的 normalize/tokenize 流水线",
                "policy_postprocessor_json": "出模型后的 action 反归一化",
            },
        }

        logging.info(
            "PI05 server 就绪 | device=%s | dtype=%s | infer_horizon=%d | action_dim=%d",
            self.device,
            self.runtime_dtype,
            self.infer_horizon,
            self.expected_action_dim,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _runtime_dtype_for_processor(dtype: torch.dtype) -> str | None:
        mapping = {
            torch.bfloat16: "bfloat16",
            torch.float32: "float32",
            torch.float16: "float16",
        }
        return mapping.get(dtype)

    @staticmethod
    def _value_to_tensor(value: Any) -> Any:
        """msgpack 反序列化后的 numpy → torch.Tensor（与 LeRobotDataset  dtype 对齐）。"""
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            arr = np.array(value, copy=True)  # msgpack buffer 可能只读
            # 兼容 uint8 HWC 旧客户端：转成 CHW float32 [0,1]
            if arr.ndim == 3 and arr.dtype == np.uint8 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
            elif np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            return torch.from_numpy(np.ascontiguousarray(arr))
        return value

    def _payload_to_frame(self, obs: dict[str, Any]) -> dict[str, Any]:
        """
        WebSocket payload → LeRobot frame，供 ``preprocess(frame)`` 直接消费。

        推荐客户端发送 ``dict(ds[i])``（tensor 经 msgpack 变为 numpy 也没关系）。
        键名应与 checkpoint 的 ``policy.config.input_features`` 一致，
        或由 ``--rename_map`` 在 preprocessor 内对齐。
        """
        frame: dict[str, Any] = {}

        for key, value in obs.items():
            if key in CONTROL_KEYS or isinstance(value, dict):
                continue
            frame[key] = self._value_to_tensor(value)

        if "task" not in frame:
            prompt = obs.get("prompt") or self.args.default_prompt
            frame["task"] = str(prompt)

        # --- 旧版紧凑协议 fallback（不推荐）---
        if OBS_STATE not in frame:
            for state_key in (OBS_STATE, "state", "qpos"):
                if state_key in obs:
                    frame[OBS_STATE] = self._value_to_tensor(
                        np.asarray(obs[state_key], dtype=np.float32)
                    )
                    break

        images = obs.get("images")
        if isinstance(images, dict) and not any(k.startswith("observation.images.") for k in frame):
            for policy_key in self.expected_image_keys:
                short_name = policy_key.split(".")[-1]
                if short_name in images:
                    frame[policy_key] = self._value_to_tensor(images[short_name])

        missing_images = [k for k in self.expected_image_keys if k not in frame]
        if missing_images:
            raise KeyError(
                f"frame 缺少图像 key: {missing_images}。"
                f"请发送 LeRobot frame（如 dict(ds[i])），或配置 --rename_map。"
            )
        if OBS_STATE not in frame:
            raise KeyError(f"frame 缺少 {OBS_STATE!r}，请发送 LeRobot frame 或 state/qpos。")

        return frame

    def _postprocess_action_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        """
        对 [B, T, A] chunk 逐步做 postprocessor（unnormalize）。

        这与 lerobot 官方 async policy_server 的处理方式一致：
        postprocessor 接口面向单步 action (B, A)。
        PolicyAction 在 lerobot 中就是 torch.Tensor 的类型别名。
        """
        _, chunk_size, _ = action_chunk.shape
        processed: list[PolicyAction] = []
        for step_idx in range(chunk_size):
            single: PolicyAction = action_chunk[:, step_idx, :]
            processed.append(self.postprocessor(single))
        return torch.stack(processed, dim=1).squeeze(0)

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        """单次推理入口（WebSocket 每个请求调用一次）。"""
        if obs.get("reset") or obs.get("timestep") == 0:
            self.policy.reset()

        # Step A: payload -> LeRobot frame（等同 ds[i]）
        frame = self._payload_to_frame(obs)

        # Step B: preprocess(frame) —— 与本地 notebook 完全一致
        batch = self.preprocessor(frame)

        # Step C: 模型推理
        with torch.inference_mode():
            action_chunk = self.policy.predict_action_chunk(batch)
        if action_chunk.ndim != 3:
            raise RuntimeError(f"PI05 输出维度异常: {tuple(action_chunk.shape)}")

        model_chunk = action_chunk[0, : self.infer_horizon, : self.expected_action_dim]

        # Step D: postprocessor（读 policy_postprocessor.json）
        action_chunk_processed = self._postprocess_action_chunk(
            action_chunk[:, : self.infer_horizon, :]
        )

        model_np = model_chunk.detach().cpu().numpy().astype(np.float32)
        action_np = action_chunk_processed.detach().cpu().numpy().astype(np.float32)

        return {
            "actions": action_np,
            "action": action_np[0],
            "model_actions": model_np,
            "model_action": model_np[0],
        }


# ---------------------------------------------------------------------------
# WebSocket Server（精简版，协议与 openpi / my_tbot 示例兼容）
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    """
    协议说明
    --------
    1. 连接建立后，服务端先发送 metadata（msgpack）
    2. 客户端循环发送 obs_dict（msgpack）
    3. 服务端返回 action_dict（msgpack），附带 server_timing
    """

    def __init__(
        self,
        policy: LeRobotPI05Policy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with websocket_server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: websocket_server.ServerConnection) -> None:
        logging.info("客户端已连接: %s", websocket.remote_address)
        packer = MsgpackPacker()
        await websocket.send(packer.pack(self._metadata))

        prev_total_time: float | None = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_unpack(await websocket.recv())

                infer_start = time.monotonic()
                action = self._policy.infer(obs)
                infer_ms = (time.monotonic() - infer_start) * 1000.0

                action["server_timing"] = {"infer_ms": infer_ms}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000.0

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logging.info("客户端断开: %s", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(
    connection: websocket_server.ServerConnection,
    request: websocket_server.Request,
) -> websocket_server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(args: ServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))

    policy = LeRobotPI05Policy(args)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError as exc:
        local_ip = "unknown"
        logging.warning("无法解析本机 IP (%s): %s", hostname, exc)

    logging.info(
        "PI05 WebSocket 服务启动 | host=%s ip=%s port=%d ckpt=%s",
        args.host,
        local_ip,
        args.port,
        policy.ckpt_dir,
    )
    logging.info("Server metadata:\n%s", json.dumps(policy.metadata, indent=2, ensure_ascii=False))

    server = WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    main(parse_args())
