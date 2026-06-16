# -*- coding: utf-8 -*-
"""
RoboTwin processed 数据 → LeRobot v3.0

输入目录由 ``my_process_data.py`` 生成（``*_processed``），结构示例::

    demo_clean_processed/
        episode_0/
            episode_0.hdf5
            instructions.json
        episode_1/
            ...

用法::

    cd /vla/my_vla
    python data_processors/convert_robotwin_to_lerobot.py
    python data_processors/convert_robotwin_to_lerobot.py --file_path /path/to/demo_clean_processed
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import tqdm

# ---------------------------------------------------------------------------
# 本地 lerobot 源码（与 scripts/serve_lerobot_policy.py 一致）
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
MY_VLA_ROOT = SCRIPT_DIR.parent
LEROBOT_SRC = MY_VLA_ROOT / "src" / "lerobot" / "src"
for _path in (SCRIPT_DIR, LEROBOT_SRC, MY_VLA_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

# =============================================================================
# 路径设定
# =============================================================================
ORIGINAL_DATA_PATH = "/vla/my_robotwin/data/adjust_bottle/demo_clean_processed"
REPO_ID = "adjust_bottle"
CONVERTED_DATA_PATH = "/vla/.data/lerobot_v3/" + REPO_ID

# 数据集特征前缀
IMAGE_KEY_PREFIX = "observation.images."
ROBOT_KEYS = ["observation.state", "action"]
MAX_EPISODES = None  # 调试时可设为 2

# =============================================================================
# 机器人 / 相机配置（RoboTwin 双臂，与 pi05 process_data 输出对齐）
# =============================================================================
FPS = 50
ROBOT_TYPE = "aloha"
IMAGE_SHAPE_CHW = (3, 480, 640)

MOTORS = [
    "left_waist",
    "left_shoulder",
    "left_elbow",
    "left_forearm_roll",
    "left_wrist_angle",
    "left_wrist_rotate",
    "left_gripper",
    "right_waist",
    "right_shoulder",
    "right_elbow",
    "right_forearm_roll",
    "right_wrist_angle",
    "right_wrist_rotate",
    "right_gripper",
]

# HDF5 内 observations/images 下的 key；若与首条 episode 不一致会在运行时校验
CAMERAS = ["cam_high", "cam_left_wrist", "cam_right_wrist"]

# 语言指令：instructions.json 里通常是列表，取第一条或随机一条
INSTRUCTION_MODE = "first"  # "first" | "random"


# =============================================================================
# 特征配置 FEATURE_CONFIG
# =============================================================================
FEATURE_CONFIG = {
    "observation.state": {
        "dtype": "float32",
        "shape": (len(MOTORS),),
        "names": [MOTORS],
    },
    "action": {
        "dtype": "float32",
        "shape": (len(MOTORS),),
        "names": [MOTORS],
    },
}
for _cam in CAMERAS:
    FEATURE_CONFIG[f"{IMAGE_KEY_PREFIX}{_cam}"] = {
        "dtype": "video",
        "shape": IMAGE_SHAPE_CHW,
        "names": ["channels", "height", "width"],
    }

# =============================================================================
# LeRobot v3.0 创建参数 CONVERT_CONFIG
# =============================================================================
CONVERT_CONFIG = {
    "repo_id": REPO_ID,
    "root": CONVERTED_DATA_PATH,
    "robot_type": ROBOT_TYPE,
    "features": FEATURE_CONFIG,
    "fps": FPS,
    # 常见配置（与飞升-神功.ipynb 保持一致）
    "use_videos": True,
    "tolerance_s": 0.0001,
    "video_backend": "torchcodec",
    "image_writer_processes": 4,
    "image_writer_threads": 16,
    "batch_encoding_size": 1,
    "vcodec": "libsvtav1",
}


# =============================================================================
# 数据读取
# =============================================================================
def list_episode_dirs(data_root: Path) -> list[Path]:
    """按 episode 序号排序，返回 episode_{i} 目录列表。"""
    episode_dirs = []
    for child in data_root.iterdir():
        if child.is_dir() and child.name.startswith("episode_"):
            episode_dirs.append(child)
    episode_dirs.sort(key=lambda p: int(p.name.split("_", 1)[1]))
    if not episode_dirs:
        raise FileNotFoundError(f"未在 {data_root} 下找到 episode_* 目录")
    return episode_dirs


def find_episode_hdf5(episode_dir: Path) -> Path:
    """episode 目录内 hdf5 命名约定: episode_{i}/episode_{i}.hdf5"""
    idx = episode_dir.name.split("_", 1)[1]
    hdf5_path = episode_dir / f"episode_{idx}.hdf5"
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"缺少 HDF5: {hdf5_path}")
    return hdf5_path


def load_episode_instruction(episode_dir: Path) -> str:
    instr_path = episode_dir / "instructions.json"
    with instr_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    instructions = payload["instructions"]
    if isinstance(instructions, str):
        return instructions
    if INSTRUCTION_MODE == "random":
        return str(np.random.choice(instructions))
    return str(instructions[0])


def load_compressed_images(ep: h5py.File, camera: str) -> np.ndarray:
    """从 JPEG 字节串解码图像，形状 (T, H, W, C)，BGR uint8。"""
    dataset = ep[f"observations/images/{camera}"]
    if dataset.ndim == 4:
        return dataset[:]
    frames = []
    for data in dataset:
        buf = np.frombuffer(data, np.uint8)
        frames.append(cv2.imdecode(buf, cv2.IMREAD_COLOR))
    return np.stack(frames, axis=0)


def load_episode_data(ep_path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """读取整条轨迹: 多相机图像 + qpos + action。"""
    with h5py.File(ep_path, "r") as ep:
        state = ep["observations/qpos"][:].astype(np.float32)
        action = ep["action"][:].astype(np.float32)

        imgs_per_cam = {}
        for camera in CAMERAS:
            imgs_per_cam[camera] = load_compressed_images(ep, camera)

    return imgs_per_cam, state, action


def load_frame_data(
    imgs_per_cam: dict[str, np.ndarray],
    state: np.ndarray,
    action: np.ndarray,
    frame_index: int,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """单帧数据: 返回带前缀的图像 dict、state、action（与飞升-神功.ipynb 同名函数对齐）。"""
    frame_imgs = {
        f"{IMAGE_KEY_PREFIX}{cam}": imgs_per_cam[cam][frame_index] for cam in CAMERAS
    }
    return frame_imgs, state, action


def inspect_cameras_from_first_episode(data_root: Path) -> list[str]:
    """用首条 episode 校验 CAMERAS 是否与 HDF5 一致。"""
    first_hdf5 = find_episode_hdf5(list_episode_dirs(data_root)[0])
    with h5py.File(first_hdf5, "r") as ep:
        found = list(ep["observations/images"].keys())
    for cam in CAMERAS:
        if cam not in found:
            raise KeyError(f"配置相机 {cam} 不在 HDF5 中，实际 keys: {found}")
    return CAMERAS


# =============================================================================
# 写入 LeRobot v3.0
# =============================================================================
def populate_dataset(
    data_root: Path,
    dataset: LeRobotDataset,
    max_episodes: int | None = None,
) -> LeRobotDataset:
    """
    逐 episode 读 HDF5、组帧、写入；每个 episode 结束时 save_episode。

    1. 图像 — cv2 解码后直接 add_frame（HWC BGR）
    2. 轨迹 — 按 length 遍历帧，带 task 字段
    """
    episode_dirs = list_episode_dirs(data_root)
    if max_episodes is not None:
        episode_dirs = episode_dirs[:max_episodes]

    total_frames = 0
    for episode_index, episode_dir in enumerate(episode_dirs):
        ep_path = find_episode_hdf5(episode_dir)
        task_name = load_episode_instruction(episode_dir)
        imgs_per_cam, state, action = load_episode_data(ep_path)
        num_frames = state.shape[0]

        print(
            f"episode {episode_index} / {len(episode_dirs)}, "
            f"frames={num_frames}, task={task_name}"
        )

        for i in tqdm.tqdm(range(num_frames), desc=f"episode_{episode_index}", leave=False):
            frame_imgs, frame_state, frame_action = load_frame_data(
                imgs_per_cam,
                state[i],
                action[i],
                frame_index=i,
            )
            frame = {
                ROBOT_KEYS[0]: frame_state,
                ROBOT_KEYS[1]: frame_action,
                "task": task_name,
            }
            frame.update(frame_imgs)
            dataset.add_frame(frame)
            total_frames += 1

        dataset.save_episode()

    print(f"共写入 {len(episode_dirs)} 条轨迹, {total_frames} 帧")
    dataset.finalize()
    return dataset


def run_convert(
    original_data_path: str | Path,
    converted_data_path: str | Path,
    repo_id: str,
    max_episodes: int | None = None,
) -> LeRobotDataset:
    data_root = Path(original_data_path).resolve()
    output_root = Path(converted_data_path).resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {data_root}")

    inspect_cameras_from_first_episode(data_root)

    convert_config = dict(CONVERT_CONFIG)
    convert_config["repo_id"] = repo_id
    convert_config["root"] = str(output_root)

    if output_root.exists():
        print(f"删除已有输出目录: {output_root}")
        shutil.rmtree(output_root)

    print(f"输入: {data_root}")
    print(f"输出: {output_root}")

    dataset = LeRobotDataset.create(**convert_config)
    dataset = populate_dataset(data_root, dataset, max_episodes=max_episodes)
    return dataset


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="将 RoboTwin processed 数据转换为 LeRobot v3.0 格式"
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=None,
        help="覆盖 ORIGINAL_DATA_PATH（processed 数据根目录）",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="覆盖 CONVERTED_DATA_PATH（LeRobot v3.0 输出目录）",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="覆盖 REPO_ID",
    )
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=None,
        help="最多转换的 episode 数量（调试用）",
    )
    args = parser.parse_args()

    original_path = args.file_path or ORIGINAL_DATA_PATH
    repo_id = args.repo_id or REPO_ID
    if args.output_path is not None:
        output_path = args.output_path
    elif args.repo_id is not None:
        output_path = "/vla/.data/lerobot_v3/" + repo_id
    else:
        output_path = CONVERTED_DATA_PATH
    max_episodes = args.max_episodes if args.max_episodes is not None else MAX_EPISODES

    dataset = run_convert(
        original_data_path=original_path,
        converted_data_path=output_path,
        repo_id=repo_id,
        max_episodes=max_episodes,
    )
    print(dataset)
