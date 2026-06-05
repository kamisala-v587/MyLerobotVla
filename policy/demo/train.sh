#!/usr/bin/env bash
set -euo pipefail

# 安装 demo policy 插件（已安装可跳过）
pip install -e /vla/my_vla/policy/demo

# 从零训练 demo_policy（无需 --policy.path）
accelerate launch -m lerobot.scripts.lerobot_train \
  --config_path=/vla/my_vla/policy/demo/train_config.json
