#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LeRobot 通用 Policy WebSocket Server
====================================

用途
----
加载任意 LeRobot checkpoint（由 ``config.json`` 的 ``type`` 字段自动识别策略），
通过 WebSocket + msgpack 对外提供 ``infer(obs_dict) -> action_dict``。

与 ``serve_pi05_policy.py`` 使用**相同客户端协议**（连接后收 metadata，循环发 frame）。
区别：本脚本通过 ``lerobot.policies.factory`` 自动选择 Policy 类与 pre/post processor，
无需为每种策略单独写 server。

黑盒使用（通常只需）
--------------------
1. 指向训练产物的 ``pretrained_model`` 目录（含 config / model / processor JSON）
2. 可选 ``--rename_map`` 对齐相机 key
3. VLA 类策略（pi0/pi05/smolvla 等）离线部署时准备 tokenizer 本地路径

启动示例
--------

.. code-block:: bash

   cd /vla/my_vla
   conda activate myvla
   export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

   python /vla/my_vla/scripts/serve_lerobot_policy.py \\
       --ckpt_path /path/to/pretrained_model \\
       --host 0.0.0.0 --port 8000 \\
       --default_prompt "adjust the bottle" \\
       --infer_horizon 16

客户端推荐发送 ``dict(LeRobotDataset[i])``，与本地 ``preprocessor(frame)`` 输入一致。

不支持作为 action server 的策略
--------------------------------
``sac``（单步 RL）、``sarm`` / ``reward_classifier``（奖励模型，不输出 action）。
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
# 路径：本地 lerobot 源码
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MY_VLA_ROOT = SCRIPT_DIR.parent
LEROBOT_SRC = MY_VLA_ROOT / "src" / "lerobot" / "src"
for path in (SCRIPT_DIR, LEROBOT_SRC, MY_VLA_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_PALIGEMMA_TOKENIZER_REPO = "google/paligemma-3b-pt-224"
DEFAULT_LOCAL_TOKENIZER_CANDIDATES = (
    Path("/vla/.models/paligemma-3b-pt-224"),
    Path("/vla/.models/google/paligemma-3b-pt-224"),
)

# 无法作为 action chunk server 使用的策略类型
NON_ACTION_SERVER_TYPES = frozenset({"sac", "sarm", "reward_classifier"})

# 默认尝试离线解析 PaliGemma tokenizer 的策略
PALIGEMMA_POLICY_TYPES = frozenset({"pi0", "pi05", "pi0_fast"})

from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.configs.types import FeatureType  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.processor import PolicyAction  # noqa: E402
from lerobot.utils.constants import (  # noqa: E402
    ACTION,
    OBS_STATE,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)

CONTROL_KEYS = frozenset({"reset", "timestep", "prompt"})


# ---------------------------------------------------------------------------
# msgpack + numpy
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
msgpack_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@dataclass
class ServeArgs:
    ckpt_path: str
    host: str = "0.0.0.0"
    port: int = 8000
    default_prompt: str = "Execute the task."
    device: str = "auto"
    dtype: str = "auto"  # auto | float32 | bfloat16
    infer_horizon: int | None = None
    rename_map: dict[str, str] | None = None
    tokenizer_path: str | None = None
    preprocessor_overrides: dict[str, Any] | None = None
    postprocessor_overrides: dict[str, Any] | None = None


def parse_args() -> ServeArgs:
    parser = argparse.ArgumentParser(
        description="启动任意 LeRobot checkpoint 的 WebSocket 推理服务。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt_path", required=True, help="pretrained_model 或其上一级 step 目录。")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default_prompt", default="Execute the task.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "bfloat16"], default="auto")
    parser.add_argument(
        "--infer_horizon",
        type=int,
        default=None,
        help="每次返回的 action 步数；默认读 config.n_action_steps 或 chunk_size，单步策略为 1。",
    )
    parser.add_argument(
        "--rename_map",
        default=None,
        help='JSON：环境 key → policy key，如 \'{"head":"observation.images.cam_high"}\'。',
    )
    parser.add_argument(
        "--tokenizer_path",
        default=None,
        help="本地 tokenizer 目录（VLA 策略离线部署）。也可设 PALIGEMMA_TOKENIZER_PATH / TOKENIZER_PATH。",
    )
    parser.add_argument(
        "--preprocessor_overrides",
        default=None,
        help="JSON：覆盖 preprocessor 某 step 的配置（高级用法）。",
    )
    parser.add_argument(
        "--postprocessor_overrides",
        default=None,
        help="JSON：覆盖 postprocessor 某 step 的配置（高级用法）。",
    )
    parsed = parser.parse_args()

    def _load_json_obj(raw: str | None, flag: str) -> dict[str, Any] | None:
        if not raw:
            return None
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError(f"{flag} 必须是 JSON object")
        return obj

    rename_map = _load_json_obj(parsed.rename_map, "--rename_map")
    preprocessor_overrides = _load_json_obj(parsed.preprocessor_overrides, "--preprocessor_overrides")
    postprocessor_overrides = _load_json_obj(parsed.postprocessor_overrides, "--postprocessor_overrides")

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
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def resolve_ckpt_dir(ckpt_path: str | Path) -> Path:
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
    )


def _looks_like_tokenizer_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "tokenizer_config.json").is_file() or (path / "tokenizer.json").is_file()
    )


def _iter_hf_hub_roots() -> list[Path]:
    roots: list[Path] = []
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    roots.append(Path("/mnt/workspace/luyi/.cache/huggingface/hub"))
    seen: set[str] = set()
    deduped: list[Path] = []
    for root in roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def _find_tokenizer_in_hf_cache(repo_id: str) -> Path | None:
    repo_folder = "models--" + repo_id.replace("/", "--")
    for hub_root in _iter_hf_hub_roots():
        snapshots_dir = hub_root / repo_folder / "snapshots"
        if not snapshots_dir.is_dir():
            continue
        for snapshot_dir in sorted(snapshots_dir.iterdir()):
            if _looks_like_tokenizer_dir(snapshot_dir):
                return snapshot_dir.resolve()
    return None


def resolve_tokenizer_path(
    explicit_path: str | None,
    repo_id: str = DEFAULT_PALIGEMMA_TOKENIZER_REPO,
) -> Path:
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
        "离线模式下找不到 tokenizer。\n"
        f"  repo_id: {repo_id}\n"
        f"  已检查: {checked}\n"
        "请使用 --tokenizer_path 或 PALIGEMMA_TOKENIZER_PATH 指定本地目录。"
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


def _runtime_dtype_for_processor(dtype: torch.dtype) -> str | None:
    return {
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float16: "float16",
    }.get(dtype)


def _read_preprocessor_steps(ckpt_dir: Path) -> list[dict[str, Any]]:
    path = ckpt_dir / f"{POLICY_PREPROCESSOR_DEFAULT_NAME}.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    steps = data.get("steps", [])
    return steps if isinstance(steps, list) else []


def _has_processor_step(ckpt_dir: Path, registry_name: str) -> bool:
    return any(step.get("registry_name") == registry_name for step in _read_preprocessor_steps(ckpt_dir))


def _read_normalizer_stat_dim(ckpt_dir: Path, feature_key: str) -> int | None:
    """从 normalizer safetensors 读取 stats 实际维数（比 config 更准确）。"""
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    stat_key = f"{feature_key}.q01"
    for path in sorted(ckpt_dir.glob("policy_preprocessor_step_*_normalizer_processor.safetensors")):
        with safe_open(str(path), framework="pt") as sf:
            if stat_key in sf.keys():
                return int(sf.get_tensor(stat_key).shape[0])
    return None


def _deep_merge_dict(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in extra.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# 通用 Policy 推理封装
# ---------------------------------------------------------------------------
class LeRobotPolicyService:
    """
    任意 LeRobot Policy + checkpoint 内 pre/post processor → ``infer(obs)``。

    数据流
    ------
    client frame → preprocessor → policy → postprocessor → numpy actions
    """

    def __init__(self, args: ServeArgs):
        self.args = args
        self.ckpt_dir = resolve_ckpt_dir(args.ckpt_path)

        self.config = PreTrainedConfig.from_pretrained(self.ckpt_dir)
        self.policy_type = str(self.config.type)
        if self.policy_type in NON_ACTION_SERVER_TYPES:
            raise ValueError(
                f"策略 type={self.policy_type!r} 不适合作为 action server。"
                f"不支持: {sorted(NON_ACTION_SERVER_TYPES)}"
            )

        self.device = resolve_device(args.device)
        self.config.device = self.device
        self.runtime_dtype = resolve_runtime_dtype(
            args.dtype if args.dtype != "auto" else "float32",
            getattr(self.config, "dtype", None),
            self.device,
        )
        if args.dtype == "auto" and str(getattr(self.config, "dtype", "")).lower() in {"bfloat16", "bf16"}:
            logging.info(
                "推理 dtype 使用 float32（checkpoint 训练为 %s；可用 --dtype bfloat16 覆盖）",
                self.config.dtype,
            )

        self.supports_chunking = self.policy_type != "sac"
        chunk_size = getattr(self.config, "chunk_size", None)
        n_action_steps = getattr(self.config, "n_action_steps", None)
        if self.supports_chunking and chunk_size is not None:
            default_horizon = int(n_action_steps or chunk_size)
            self.infer_horizon = int(args.infer_horizon or default_horizon)
            self.infer_horizon = max(1, min(self.infer_horizon, int(chunk_size)))
        else:
            self.infer_horizon = 1
            if args.infer_horizon not in (None, 1):
                logging.warning("策略 %s 仅支持单步 action，infer_horizon 强制为 1", self.policy_type)

        policy_cls = get_policy_class(self.policy_type)
        logging.info("正在加载 %s: %s", policy_cls.__name__, self.ckpt_dir)
        self.policy = policy_cls.from_pretrained(self.ckpt_dir, config=self.config)
        self.policy.config.device = self.device
        self.policy.to(device=self.device, dtype=self.runtime_dtype)
        self.policy.eval()

        preprocessor_overrides, postprocessor_overrides = self._build_processor_overrides()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.ckpt_dir),
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )

        self.expected_image_keys = list(self.policy.config.image_features.keys())
        self.expected_input_keys = list(self.policy.config.input_features.keys())
        self.expected_action_dim = int(self.policy.config.output_features[ACTION].shape[0])
        self.norm_state_dim = _read_normalizer_stat_dim(self.ckpt_dir, OBS_STATE)
        config_state_dim = None
        if OBS_STATE in self.policy.config.input_features:
            config_state_dim = int(self.policy.config.input_features[OBS_STATE].shape[0])

        self._metadata = {
            "model_type": self.policy_type,
            "deployment": "lerobot_generic_websocket_server",
            "checkpoint_dir": str(self.ckpt_dir),
            "device": self.device,
            "dtype": str(self.runtime_dtype),
            "infer_horizon": self.infer_horizon,
            "chunk_size": chunk_size,
            "supports_chunking": self.supports_chunking,
            "default_prompt": args.default_prompt,
            "expected_image_keys": self.expected_image_keys,
            "expected_input_keys": self.expected_input_keys,
            "config_state_dim": config_state_dim,
            "norm_state_dim": self.norm_state_dim,
            "expected_action_dim": self.expected_action_dim,
            "rename_map_override": args.rename_map or {},
            "tokenizer_path": getattr(self, "resolved_tokenizer_path", None),
            "notes": {
                "client_payload": "推荐发送 dict(LeRobotDataset[i])",
                "norm_state_dim": "preprocessor stats 中 state 维数，client 应对齐此值做 normalize",
                "config_state_dim": "config.input_features 声明，PI05 等可能与 norm_state_dim 不同",
            },
        }

        logging.info(
            "Server 就绪 | type=%s | device=%s | dtype=%s | horizon=%d | action_dim=%d",
            self.policy_type,
            self.device,
            self.runtime_dtype,
            self.infer_horizon,
            self.expected_action_dim,
        )

    def _build_processor_overrides(self) -> tuple[dict[str, Any], dict[str, Any]]:
        rename_override = self.args.rename_map or {}
        runtime_dtype_name = _runtime_dtype_for_processor(self.runtime_dtype)

        preprocessor_overrides: dict[str, Any] = {
            "device_processor": {
                "device": self.device,
                "float_dtype": runtime_dtype_name,
            },
            "rename_observations_processor": {"rename_map": rename_override},
        }
        postprocessor_overrides: dict[str, Any] = {
            "device_processor": {"device": self.device},
        }

        if _has_processor_step(self.ckpt_dir, "tokenizer_processor"):
            tokenizer_override = self._resolve_tokenizer_override()
            if tokenizer_override is not None:
                preprocessor_overrides["tokenizer_processor"] = {
                    "tokenizer_name": str(tokenizer_override),
                }
                self.resolved_tokenizer_path = str(tokenizer_override)
                logging.info("tokenizer_processor 使用本地路径: %s", tokenizer_override)

        if self.args.preprocessor_overrides:
            preprocessor_overrides = _deep_merge_dict(
                preprocessor_overrides, self.args.preprocessor_overrides
            )
        if self.args.postprocessor_overrides:
            postprocessor_overrides = _deep_merge_dict(
                postprocessor_overrides, self.args.postprocessor_overrides
            )
        return preprocessor_overrides, postprocessor_overrides

    def _resolve_tokenizer_override(self) -> Path | None:
        if self.args.tokenizer_path:
            path = Path(self.args.tokenizer_path).expanduser()
            if not _looks_like_tokenizer_dir(path):
                raise FileNotFoundError(f"--tokenizer_path 不是有效 tokenizer 目录: {path}")
            return path.resolve()

        if self.policy_type in PALIGEMMA_POLICY_TYPES:
            return resolve_tokenizer_path(None)

        logging.warning(
            "checkpoint 含 tokenizer_processor，但未指定 --tokenizer_path；"
            "将使用 policy_preprocessor.json 中保存的 tokenizer_name（离线可能失败）。"
        )
        return None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    @staticmethod
    def _value_to_tensor(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, np.ndarray):
            arr = np.array(value, copy=True)
            if arr.ndim == 3 and arr.dtype == np.uint8 and arr.shape[-1] == 3:
                arr = np.transpose(arr, (2, 0, 1)).astype(np.float32) / 255.0
            elif np.issubdtype(arr.dtype, np.floating):
                arr = arr.astype(np.float32, copy=False)
            return torch.from_numpy(np.ascontiguousarray(arr))
        return value

    def _payload_to_frame(self, obs: dict[str, Any]) -> dict[str, Any]:
        frame: dict[str, Any] = {}
        for key, value in obs.items():
            if key in CONTROL_KEYS or isinstance(value, dict):
                continue
            frame[key] = self._value_to_tensor(value)

        if "task" not in frame:
            frame["task"] = str(obs.get("prompt") or self.args.default_prompt)

        # 紧凑协议 fallback
        if OBS_STATE not in frame and OBS_STATE in self.policy.config.input_features:
            for state_key in (OBS_STATE, "state", "qpos"):
                if state_key in obs:
                    frame[OBS_STATE] = self._value_to_tensor(
                        np.asarray(obs[state_key], dtype=np.float32)
                    )
                    break

        images = obs.get("images")
        if isinstance(images, dict) and self.expected_image_keys:
            if not any(k.startswith("observation.images.") for k in frame):
                for policy_key in self.expected_image_keys:
                    short_name = policy_key.split(".")[-1]
                    if short_name in images:
                        frame[policy_key] = self._value_to_tensor(images[short_name])

        missing = [k for k in self.expected_input_keys if k not in frame]
        if missing:
            raise KeyError(
                f"frame 缺少 input_features 中的 key: {missing}。"
                f"请发送 LeRobot frame（dict(ds[i])），或配置 --rename_map。"
            )
        return frame

    def _postprocess_action_chunk(self, action_chunk: torch.Tensor) -> torch.Tensor:
        _, chunk_size, _ = action_chunk.shape
        processed: list[PolicyAction] = []
        for step_idx in range(chunk_size):
            single: PolicyAction = action_chunk[:, step_idx, :]
            processed.append(self.postprocessor(single))
        return torch.stack(processed, dim=1).squeeze(0)

    def _predict_raw_actions(self, batch: dict[str, Any]) -> torch.Tensor:
        if not self.supports_chunking:
            action = self.policy.select_action(batch)
            if action.ndim == 1:
                action = action.unsqueeze(0)
            return action.unsqueeze(1)  # (B, 1, A)

        chunk = self.policy.predict_action_chunk(batch)
        if chunk.ndim == 2:
            chunk = chunk.unsqueeze(0)
        if chunk.ndim != 3:
            raise RuntimeError(f"策略输出维度异常: {tuple(chunk.shape)}，期望 (B, T, A)")
        return chunk

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if obs.get("reset") or obs.get("timestep") == 0:
            self.policy.reset()

        frame = self._payload_to_frame(obs) # 预处理 提取特征 numpy -> tensor 特征校验等
        batch = self.preprocessor(frame)

        with torch.inference_mode():
            action_chunk = self._predict_raw_actions(batch) # 调用 select_action 进行推理

        horizon = min(self.infer_horizon, action_chunk.shape[1])
        action_chunk = action_chunk[:, :horizon, : self.expected_action_dim]

        action_chunk_processed = self._postprocess_action_chunk(action_chunk)

        model_np = action_chunk.detach().cpu().numpy().astype(np.float32)
        action_np = action_chunk_processed.detach().cpu().numpy().astype(np.float32)

        return {
            "actions": action_np,
            "action": action_np[0],
            "model_actions": model_np,
            "model_action": model_np[0],
        }


# ---------------------------------------------------------------------------
# WebSocket Server（协议与 serve_pi05_policy.py / client_sim.ipynb 兼容）
# ---------------------------------------------------------------------------
class WebsocketPolicyServer:
    def __init__(
        self,
        policy: LeRobotPolicyService,
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
                obs = msgpack_unpack(await websocket.recv()) # 反序列化 提取obs

                infer_start = time.monotonic()
                action = self._policy.infer(obs) # 推理
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


def main(args: ServeArgs) -> None:
    logging.info("启动参数:\n%s", json.dumps(asdict(args), indent=2, ensure_ascii=False))

    policy = LeRobotPolicyService(args)

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except OSError as exc:
        local_ip = "unknown"
        logging.warning("无法解析本机 IP (%s): %s", hostname, exc)

    logging.info(
        "LeRobot WebSocket 服务 | type=%s | host=%s ip=%s port=%d",
        policy.policy_type,
        args.host,
        local_ip,
        args.port,
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
