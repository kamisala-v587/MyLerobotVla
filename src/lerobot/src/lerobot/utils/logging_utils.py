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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from lerobot.utils.utils import format_big_number

LOSS_LOG_HEADER = "step,timestamp,loss,grad_norm,lr,update_s,data_s,epochs\n"


class AverageMeter:
    """
    Computes and stores the average and current value
    Adapted from https://github.com/pytorch/examples/blob/main/imagenet/main.py
    """

    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name}:{avg" + self.fmt + "}"
        return fmtstr.format(**self.__dict__)


class MetricsTracker:
    """
    A helper class to track and log metrics over time.

    Usage pattern:

    ```python
    # initialize, potentially with non-zero initial step (e.g. if resuming run)
    metrics = {"loss": AverageMeter("loss", ":.3f")}
    train_metrics = MetricsTracker(cfg, dataset, metrics, initial_step=step)

    # update metrics derived from step (samples, episodes, epochs) at each training step
    train_metrics.step()

    # update various metrics
    loss = policy.forward(batch)
    train_metrics.loss = loss

    # display current metrics
    logging.info(train_metrics)

    # export for wandb
    wandb.log(train_metrics.to_dict())

    # reset averages after logging
    train_metrics.reset_averages()
    ```
    """

    __keys__ = [
        "_batch_size",
        "_num_frames",
        "_avg_samples_per_ep",
        "metrics",
        "steps",
        "samples",
        "episodes",
        "epochs",
        "accelerator",
    ]

    def __init__(
        self,
        batch_size: int,
        num_frames: int,
        num_episodes: int,
        metrics: dict[str, AverageMeter],
        initial_step: int = 0,
        accelerator: Callable | None = None,
    ):
        self.__dict__.update(dict.fromkeys(self.__keys__))
        self._batch_size = batch_size
        self._num_frames = num_frames
        self._avg_samples_per_ep = num_frames / num_episodes
        self.metrics = metrics

        self.steps = initial_step
        # A sample is an (observation,action) pair, where observation and action
        # can be on multiple timestamps. In a batch, we have `batch_size` number of samples.
        self.samples = self.steps * self._batch_size
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames
        self.accelerator = accelerator

    def __getattr__(self, name: str) -> int | dict[str, AverageMeter] | AverageMeter | Any:
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self.metrics:
            return self.metrics[name]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.__dict__:
            super().__setattr__(name, value)
        elif name in self.metrics:
            self.metrics[name].update(value)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def step(self) -> None:
        """
        Updates metrics that depend on 'step' for one step.
        """
        self.steps += 1
        self.samples += self._batch_size * (self.accelerator.num_processes if self.accelerator else 1)
        self.episodes = self.samples / self._avg_samples_per_ep
        self.epochs = self.samples / self._num_frames

    def __str__(self) -> str:
        display_list = [
            f"step:{format_big_number(self.steps)}",
            # number of samples seen during training
            f"smpl:{format_big_number(self.samples)}",
            # number of episodes seen during training
            f"ep:{format_big_number(self.episodes)}",
            # number of time all unique samples are seen
            f"epch:{self.epochs:.2f}",
            *[str(m) for m in self.metrics.values()],
        ]
        return " ".join(display_list)

    def to_postfix(self, use_avg: bool = False) -> dict[str, str]:
        """供 tqdm 等进度条显示的紧凑指标（短键名）。"""
        value = lambda m: m.avg if use_avg else m.val
        postfix = {
            "epch": f"{self.epochs:.2f}",
        }
        for _name, meter in self.metrics.items():
            fmt_spec = meter.fmt[1:] if meter.fmt.startswith(":") else meter.fmt
            postfix[meter.name] = format(value(meter), fmt_spec)
        return postfix

    def to_dict(self, use_avg: bool = True) -> dict[str, int | float]:
        """
        Returns the current metric values (or averages if `use_avg=True`) as a dict.
        """
        return {
            "steps": self.steps,
            "samples": self.samples,
            "episodes": self.episodes,
            "epochs": self.epochs,
            **{k: m.avg if use_avg else m.val for k, m in self.metrics.items()},
        }

    def reset_averages(self) -> None:
        """Resets average meters."""
        for m in self.metrics.values():
            m.reset()


def append_loss_log(log_file: Path, tracker: MetricsTracker) -> None:
    """按 log_freq 追加一行 CSV 到 loss.log，便于画 loss 曲线。"""
    stats = tracker.to_dict(use_avg=True)
    row = (
        f"{stats['steps']},"
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},"
        f"{stats['loss']:.6f},"
        f"{stats['grad_norm']:.6f},"
        f"{stats['lr']:.6e},"
        f"{stats['update_s']:.6f},"
        f"{stats['dataloading_s']:.6f},"
        f"{stats['epochs']:.6f}\n"
    )
    if not log_file.exists():
        log_file.write_text(LOSS_LOG_HEADER, encoding="utf-8")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(row)
