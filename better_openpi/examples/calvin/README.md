# CALVIN Benchmark

This directory contains the client-side evaluation entry point for the [CALVIN benchmark](https://github.com/mees/calvin).

## Setup

Clone the CALVIN repo and update the path in [main.py](main.py):

```python
CALVIN_ROOT = "/path/to/calvin"
```

Create conda environment `calvin_venv` and install dependencies following [official instruction](https://github.com/mees/calvin#computer--quick-start). Then install the client package from the FASTER repo root:

```bash
cd /path/to/FASTER
pip install -e packages/openpi-client
pip install draccus
```

Make sure you have downloaded the dataset in `dataset/task_ABC_D/validation/`.

## Evaluation

Evaluation uses two processes:

1. A policy server running in the main openpi environment.
2. A CALVIN client running in the `calvin_venv` environment.

### Start the policy server

```bash
uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config=pi05_faster_calvin \
  --policy.dir=checkpoints/pi05_faster_calvin/my_experiment/29999
```

### Run evaluation

```bash
conda activate calvin_venv

export PYTHONPATH=$PYTHONPATH:$PWD

python examples/calvin/main.py \
  --save_name pi05_faster \
  --host 0.0.0.0 \
  --port 8000
```

Useful options:

- `--out_path`: Output directory.
- `--save_name`: Identifier for this evaluation run. Logs are written under `{out_path}/{save_name}/`.

See [main.py](main.py) for the full list of CLI options.

Results are written to:

```text
{out_path}/{save_name}/result.json
{out_path}/{save_name}/success_rate.txt
```

`result.json` contain:

- `avg_seq_len`: average successful sequence length in each 5-instruction chain.
- `chain_sr`: success rates for completing 1, 2, 3, 4, and 5 instructions in a row.
- `task_info`: per-task success and total counts.


## Results

The following results are evaluated with the standard CALVIN ABC->D benchmark:

- 1,000 evaluation sequences
- 5 chained instructions per sequence
- 720 environment steps per instruction

| Model       | 1    | 2    | 3    | 4    | 5    | Average Len |
| ----------- | ---- | ---- | ---- | ---- | ---- | ----------- |
| π0.5        | 94.2 | 88.7 | 85.7 | 83.2 | 79.5 | 4.313       |
| π0.5+FASTER | 95.1 | 89.1 | 85.0 | 81.9 | 78.1 | 4.292       |
