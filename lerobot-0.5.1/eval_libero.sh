source .venv/bin/activate

export HF_HOME="/home/ma-user/work/hf"
export HF_LEROBOT_HOME="/home/ma-user/work/dataset/lerobot"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
export MUJOCO_GL=egl 

# POLICY_PATH=outputs/smolvla_dist_libero_8gpu/checkpoints/100000/pretrained_model
# POLICY_DIR="${POLICY_PATH%/pretrained_model}"
POLICY_PATH="/home/ma-user/work/model/smolvla_libero"
N_ACTION_STEPS=(5)

for N_ACTION_STEP in "${N_ACTION_STEPS[@]}"; do
  OUTPUT_DIR="outputs/official/eval_naction${N_ACTION_STEP}"

  lerobot-eval \
    --policy.path="${POLICY_PATH}" \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.batch_size=1 \
    --eval.n_episodes=50 \
    --env.max_parallel_tasks=1 \
    --output_dir="${OUTPUT_DIR}" \
    --policy.n_action_steps="${N_ACTION_STEP}" \
    # --rename_map='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'
done
