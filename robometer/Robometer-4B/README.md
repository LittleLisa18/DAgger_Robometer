---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
tags:
  - reward model
  - robot learning
  - foundation models
library_name: transformers
---

# Robometer 4B

**Paper:** [arXiv](https://arxiv.org/abs/2603.02115)

**Robometer** is a general-purpose vision-language reward model for robotics. It is trained on [RBM-1M](https://huggingface.co/datasets/) with **Qwen3-VL-4B** to predict **per-frame progress**, **per-frame success**, and **trajectory preferences** from rollout videos. The model combines (1) frame-level progress supervision on expert data and (2) trajectory-comparison preference supervision, so it can learn from both successful and failed rollouts and generalize across diverse robot embodiments and tasks.

Given a **task instruction** and a **rollout video** (or frame sequence), the model predicts:

- **Per-frame progress** — continuous progress values over time (e.g. 0–1 or binned).
- **Per-frame success** — success probability (or binary) at each timestep.
- **Preference / ranking** — which of two trajectories is better for the task.

### Usage

For full setup, example scripts, and configs, see the **GitHub repo**: [github.com/robometer/robometer](https://github.com/robometer/robometer).

**Option 1 — Run the model locally** (loads this checkpoint from Hugging Face):

```bash
uv run python scripts/example_inference_local.py \
  --model-path robometer/Robometer-4B \
  --video /path/to/video.mp4 \
  --task "your task description"
```

**Option 2 — Use the evaluation server** (start server, then run client):

```bash
# Start server
uv run python robometer/evals/eval_server.py \
  --config-path=robometer/configs \
  --config-name=eval_config_server \
  model_path=robometer/Robometer-4B \
  server_url=0.0.0.0 \
  server_port=8000

# Client (no robometer dependency)
uv run python scripts/example_inference.py \
  --eval-server-url http://localhost:8000 \
  --video /path/to/video.mp4 \
  --task "your task description"
```

## Citation

If you use this model, please cite:

```bibtex
@article{liang2026robometer,
  title     = {Robometer: Scaling General-Purpose Robotic Reward Models via Trajectory Comparisons},
  author={Anthony Liang and Yigit Korkmaz and Jiahui Zhang and Minyoung Hwang and Abrar Anwar and Sidhant Kaushik and Aditya Shah and Alex S. Huang and Luke Zettlemoyer and Dieter Fox and Yu Xiang and Anqi Li and Andreea Bobu and Abhishek Gupta and Stephen Tu and Erdem Biyik and Jesse Zhang},
  year={2026},
  journal   = {arXiv preprint arXiv:2603.02115}
}
```
