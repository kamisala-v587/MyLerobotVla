# 运行命令

cd /vla/my_vla/src/lerobot
conda activate my_lerobot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

**显卡使用**
CUDA_VISIBLE_DEVICES=0,1

accelerate launch -m lerobot.scripts.lerobot_train \
  --config_path=/vla/my_vla/src/lerobot/.配置/train_config.json \
  --dataset.repo_id=/vla/.data/test \
  --policy.path=/vla/.models/lerobot-pi05_base
