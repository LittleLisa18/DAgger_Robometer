source .venv/bin/activate

export HF_HOME="/home/ma-user/work/hf"
export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_OFFLINE=1
# export TRANSFORMERS_OFFLINE=1
# export WANDB_API_KEY=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

# Distributed training on 2 GPUs via HuggingFace accelerate.
# NOTE: --batch_size is PER-GPU. Effective batch size = batch_size x num_processes.
#       32 x 2 = 64 here (matches the single-GPU train.sh). Bump to 64 for an effective 128.
accelerate launch --num_processes=2 --multi_gpu \
  -m lerobot.scripts.lerobot_train \
  --policy.path=lerobot/smolvla_base \
  --policy.load_vlm_weights=true \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/home/ma-user/work/dataset/lerobot/libero_lerobot30 \
  --policy.chunk_size=10 \
  --policy.n_action_steps=5 \
  --output_dir=./outputs/smolvla_libero \
  --job_name=smolvla_libero \
  --batch_size=32 \
  --steps=100000 \
  --wandb.enable=false \
  --wandb.mode=offline \
  --save_freq=10000 \
  --num_workers=8 \
  --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'


  # --eval.batch_size=1 \
  # --eval.n_episodes=1 \
  # --eval_freq=10000 \
  # --env.type=libero \
  # --env.task=libero_10 \
