# 运行命令

cd /vla/my_vla/src/lerobot
conda activate myvla 
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

**显卡使用**
CUDA_VISIBLE_DEVICES=0,1

accelerate launch -m lerobot.scripts.lerobot_train \
  --config_path=/vla/my_vla/policy/config.json \
  --dataset.repo_id=/vla/.data/lerobot_v3/adjust_bottle \
  --policy.path=/vla/.models/lerobot-pi05_base




# 只训练AE 并且是随机化的AE 需要修改 /vla/my_vla/policy/config.json 中的  job_name 
accelerate launch -m lerobot.scripts.lerobot_train \
  --config_path=/vla/my_vla/policy/config.json \
  --dataset.repo_id=/vla/.data/lerobot_v3/adjust_bottle \
  --policy.path=/vla/.models/pi05_AE_random

# 只训练AE 但使用pi05的AE  --- 需要修改 train_expert_only  /vla/my_vla/policy/config.json 中的 job_name
accelerate launch -m lerobot.scripts.lerobot_train \
  --config_path=/vla/my_vla/policy/config.json \
  --dataset.repo_id=/vla/.data/lerobot_v3/adjust_bottle \
  --policy.path=/vla/.models/lerobot-pi05_base



# 来自openpi的torch 版本 --- 不启动 train_expert_only 需要修改 /vla/my_vla/policy/config.json 中的 job_name
accelerate launch -m lerobot.scripts.lerobot_train \
  --config_path=/vla/my_vla/policy/config.json \
  --dataset.repo_id=/vla/.data/lerobot_v3/adjust_bottle \
  --policy.path=/vla/.models/pi05-base/torch

