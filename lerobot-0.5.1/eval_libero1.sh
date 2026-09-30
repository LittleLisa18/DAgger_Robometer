source .venv/bin/activate

export HF_HOME="/home/ma-user/work/hf"
export HF_LEROBOT_HOME="/home/ma-user/work/dataset/lerobot"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl 

POLICY_PATH=outputs/train/smolvla_dist_libero_ad/checkpoints/050000/pretrained_model
POLICY_DIR="${POLICY_PATH%/pretrained_model}"
N_ACTION_STEPS=(10 5)

SUITE=$1

for N_ACTION_STEP in "${N_ACTION_STEPS[@]}"; do
  OUTPUT_DIR="${POLICY_DIR}/eval_naction${N_ACTION_STEP}_${SUITE}"

  lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --env.type=libero \
    --env.task="libero_${SUITE}" \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --env.max_parallel_tasks=1 \
    --output_dir="${OUTPUT_DIR}" \
    --policy.n_action_steps="${N_ACTION_STEP}" \
    # --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
done
