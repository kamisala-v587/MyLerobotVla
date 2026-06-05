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
"""LeRobot 离线策略训练入口（支持 Accelerate 多卡）。"""

# 6.2 修改：将训练日志（含 log_freq 触发的 loss）写入 output_dir/train.log
import dataclasses
import logging
import time
from contextlib import nullcontext
from datetime import datetime
from pprint import pformat
from typing import Any

import torch
from accelerate import Accelerator
from termcolor import colored
from tqdm import tqdm
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker, append_loss_log
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    rabc_weights_provider=None,
) -> tuple[MetricsTracker, dict]:
    """执行单步训练：前向、反向、梯度裁剪、优化器与学习率调度。

    混合精度由 Accelerator 自动处理。

    Args:
        train_metrics: 记录本步训练指标的跟踪器。
        policy: 待训练策略模型。
        batch: 一个训练 batch。
        optimizer: 优化器。
        grad_clip_norm: 梯度裁剪的最大范数；<=0 表示仅做无穷范数检查。
        accelerator: Accelerate 实例（分布式与 AMP）。
        lr_scheduler: 可选学习率调度器。
        lock: 可选线程锁，用于保护 optimizer.step()。
        rabc_weights_provider: 可选 RA-BC 样本权重计算器。

    Returns:
        (更新后的 train_metrics, policy 前向返回的 output_dict，供日志使用)
    """
    start_time = time.perf_counter()
    policy.train()

    # 若启用 RA-BC，先计算本 batch 的样本权重
    rabc_batch_weights = None
    rabc_batch_stats = None
    if rabc_weights_provider is not None:
        rabc_batch_weights, rabc_batch_stats = rabc_weights_provider.compute_batch_weights(batch)

    # 混合精度前向
    with accelerator.autocast():
        if rabc_batch_weights is not None:
            # 逐样本 loss，再按 RA-BC 权重聚合：L = Σ(w_i * l_i) / (Σw_i + ε)
            per_sample_loss, output_dict = policy.forward(batch, reduction="none")

            # rabc_batch_weights 已归一化，使权重和等于 batch_size
            epsilon = 1e-6
            loss = (per_sample_loss * rabc_batch_weights).sum() / (rabc_batch_weights.sum() + epsilon)
            # 记录归一化前的平均权重等统计，便于监控
            output_dict["rabc_mean_weight"] = rabc_batch_stats["raw_mean_weight"]
            output_dict["rabc_num_zero_weight"] = rabc_batch_stats["num_zero_weight"]
            output_dict["rabc_num_full_weight"] = rabc_batch_stats["num_full_weight"]
        else:
            loss, output_dict = policy.forward(batch)

        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    accelerator.backward(loss)

    # 梯度裁剪
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # 按 batch 步进 scheduler（非按 epoch）
    if lr_scheduler is not None:
        lr_scheduler.step()

    # 部分 policy 带有内部状态，需在每步后 update
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """策略训练主流程。

    负责：日志与随机种子、构建数据集/环境/policy/优化器、断点恢复、
    训练循环（取 batch → update_policy）、按频率打日志/存 checkpoint/评估、
    可选推送到 Hugging Face Hub。

    Args:
        cfg: 训练流水线配置（TrainPipelineConfig）。
        accelerator: 可选；为 None 时自动创建。
    """
    cfg.validate()

    # 创建 Accelerate：自动识别单卡/分布式
    # step_scheduler_with_optimizer=False：避免按进程数缩放 scheduler 步数
    # find_unused_parameters=True：兼容条件分支导致部分参数未参与前向的模型
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        # 默认按硬件选设备；若 policy.device=cpu 则强制 CPU
        force_cpu = cfg.policy.device == "cpu"
        accelerator = Accelerator(
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    # 主进程将 loss 等日志写入 train.log（output_dir 由 validate() 确定）
    assert cfg.output_dir is not None
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    train_log_file = cfg.output_dir / "train.log"
    loss_log_file = cfg.output_dir / "loss.log"
    init_logging(
        log_file=train_log_file if accelerator.is_main_process else None,
        accelerator=accelerator,
    )
    if accelerator.is_main_process:
        logging.info(
            "训练日志（含 log_freq 触发时的 loss）同时写入: %s",
            train_log_file,
        )
        logging.info("Loss 曲线 CSV 写入: %s", loss_log_file)

    is_main_process = accelerator.is_main_process

    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # 仅主进程初始化 wandb
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("日志仅保存在本地。", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # 主进程先下载/构建数据集，避免多进程竞态
    if is_main_process:
        logging.info("正在创建数据集")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        dataset = make_dataset(cfg)

    # 仿真数据可在训练中途 eval；真机数据通常用独立 eval 脚本
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process:
        logging.info("正在创建评估环境")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("正在创建 policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    if cfg.peft is not None:
        logging.info("启用 PEFT，正在包装模型")
        peft_cli_overrides = dataclasses.asdict(cfg.peft)
        policy = policy.wrap_with_peft(peft_cli_overrides=peft_cli_overrides)

    accelerator.wait_for_everyone()

    # 构建 pre/post processor；resume 时优先用 checkpoint 里已保存的 processor
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.type == "sarm":
        processor_kwargs["dataset_meta"] = dataset.meta

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("正在创建 optimizer 与 scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    # RA-BC：加载预计算的 SARM progress（见 policies/sarm/compute_rabc_weights.py）
    rabc_weights = None
    if cfg.use_rabc:
        from lerobot.utils.rabc import RABCWeights

        chunk_size = getattr(policy.config, "chunk_size", None)
        if chunk_size is None:
            raise ValueError("Chunk size is not found in policy config")

        head_mode = getattr(cfg, "rabc_head_mode", "sparse")
        logging.info("从 %s 加载 RA-BC 用的 SARM progress", cfg.rabc_progress_path)
        logging.info("chunk_size=%s（来自 policy 配置）, head_mode=%s", chunk_size, head_mode)
        rabc_weights = RABCWeights(
            progress_path=cfg.rabc_progress_path,
            chunk_size=chunk_size,
            head_mode=head_mode,
            kappa=getattr(cfg, "rabc_kappa", 0.01),
            epsilon=getattr(cfg, "rabc_epsilon", 1e-6),
            device=device,
        )

    step = 0  # 已完成的 policy 更新步数（forward + backward + optim）

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("输出目录:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("正在创建环境侧 pre/post processor")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(
                env_cfg=cfg.env, policy_cfg=cfg.policy
            )
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"有效 batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # 离线训练 DataLoader
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, dataloader, lr_scheduler
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("开始在固定数据集上离线训练，有效 batch size: %s", effective_batch_size)

    train_pbar = None
    if is_main_process:
        train_pbar = tqdm(
            total=cfg.steps,
            initial=step,
            desc="训练",
            unit="step",
            dynamic_ncols=True,
        )

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            rabc_weights_provider=rabc_weights,
        )

        # 评估与存盘发生在第 step 次更新完成之后，故此处先 step += 1
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_main_process and train_pbar is not None:
            train_pbar.update(1)
            train_pbar.set_postfix(train_tracker.to_postfix(use_avg=is_log_step), refresh=False)

        if is_log_step:
            # 进度条负责终端展示；按 log_freq 写入 train.log / loss.log 便于事后查看
            dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(train_log_file, "a", encoding="utf-8") as log_f:
                log_f.write(f"INFO {dt} ot_train.py:step {train_tracker}\n")
            append_loss_log(loss_log_file, train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                if rabc_weights is not None:
                    rabc_stats = rabc_weights.get_stats()
                    wandb_log_dict.update(
                        {
                            "rabc_delta_mean": rabc_stats["delta_mean"],
                            "rabc_delta_std": rabc_stats["delta_std"],
                            "rabc_num_frames": rabc_stats["num_frames"],
                        }
                    )
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                msg = f"在第 {step} 步保存 checkpoint"
                tqdm.write(msg) if train_pbar is not None else logging.info(msg)
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                msg = f"在第 {step} 步评估 policy"
                tqdm.write(msg) if train_pbar is not None else logging.info(msg)
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                aggregated = eval_info["overall"]

                for suite, suite_info in eval_info.items():
                    logging.info("套件 %s 汇总: %s", suite, suite_info)

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process and train_pbar is not None:
        train_pbar.close()

    if is_main_process:
        logging.info("训练结束")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            if cfg.policy.use_peft:
                unwrapped_policy.push_model_to_hub(cfg, peft_model=unwrapped_policy)
            else:
                unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    register_third_party_plugins()
    train()


if __name__ == "__main__":
    main()
