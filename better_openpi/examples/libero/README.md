# LIBERO Benchmark

This directory contains the client-side evaluation entry point for the [LIBERO benchmark](https://github.com/Lifelong-Robot-Learning/LIBERO), following the [openpi LIBERO example](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md).

## Setup

Run the following commands from the repository root.

```bash
# Initialize LIBERO repo
git submodule update --init --recursive

# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

# Install dependencies (Add Tsinghua mirror)
uv pip sync \
  examples/libero/requirements.txt \
  third_party/libero/requirements.txt \
  --extra-index-url "https://download.pytorch.org/whl/cu113 https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple" \
  --index-strategy=unsafe-best-match

uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
```

## Evaluation

Evaluation uses two processes:

1. A policy server running in the main openpi environment.
2. A LIBERO client running in `examples/libero/.venv`.

### Start the policy server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_faster_libero \
  --policy.dir=checkpoints/pi05_faster_libero/my_experiment/29999
```

### Run evaluation

```bash
source examples/libero/.venv/bin/activate
export LIBERO_CONFIG_PATH=$PWD/third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.save-name pi05_faster \
  --args.host 0.0.0.0 \
  --args.port 8000
```

Useful options:

- `--args.task-suite-name`: Name of the task suite to evaluate (`libero_spatial`, `libero_object`, `libero_goal`, `libero_10`)
- `--args.out-path`: Output directory.
- `--args.save-name`: Identifier for this evaluation run. Results are appended to `{out_path}/results/{save_name}_results.txt`
- `--args.save-videos`: Save rollout videos under `{out_path}/videos/{save_name}/{task_suite_name}/`


See [main.py](main.py) for the full list of CLI options.

## Results

The following checkpoints are fine-tuned jointly on the four task suites for 30k steps.

| Model       | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
| ----------- | -------------- | ------------- | ----------- | --------- | ------- |
| π0.5        | 98.8           | 98.2          | 98.0        | 92.4      | 96.9    |
| π0.5+FASTER | 98.6           | 97.8          | 97.8        | 91.6      | 96.5    |
