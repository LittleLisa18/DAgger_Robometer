# Collect-aware distillation server test plan

Local tests were not run because the local workspace does not have the project runtime. Run this plan on a server with
the LeRobot datasets, teacher checkpoint, and JAX accelerator environment available.

## Automated tests

```bash
uv sync --group dev
uv run pytest -q src/openpi/training/data_loader_test.py
uv run pytest -q src/openpi/models/pi0_test.py
uv run pytest -q scripts/train_distill_test.py
uv run pytest -q src/openpi/training scripts/train_test.py
uv run ruff check scripts/train_distill.py scripts/train_distill_test.py src/openpi/models/pi0.py \
  src/openpi/policies/agilex_policy.py \
  src/openpi/models/pi0_test.py src/openpi/training/config.py src/openpi/training/data_loader.py \
  src/openpi/training/data_loader_test.py
```

The tests must verify:

- `teleop`, `teacher`, and `dagger` samples receive distillation and GT supervision; `rollout` receives distillation
  supervision only.
- `use_rollout_data=False` filters rollout indices before sampling, while `True` retains all four labels.
- Missing, uppercase, or unknown collect labels fail during dataset construction.
- The scalar collect label belongs to the current observation. A later label inside the action chunk must not change the
  current sample's GT mask.
- Single-dataset, multi-dataset, episode-end filtering, and weighted sampling use the filtered lengths.
- A regular loader returns `(observation, actions)`, while the distillation loader returns
  `(observation, actions, gt_mask)`.
- GT loss is reduced over the full batch, and an all-rollout batch has exactly zero GT loss while retaining nonzero
  distillation loss.
- `LeRobotAgilexDataConfig.arm_mode` defaults to `dual`. In `left` mode, Agilex preprocessing preserves dimensions
  `0:7` and zeros `7:14`; in `right` mode it zeros `0:7` and preserves `7:14`; `dual` leaves both arms unchanged.
  State, action chunks, and action prefixes follow the same rule without mutating the source arrays.
- `AgilexOutputs` keeps the 14-dimensional output shape and zeros the inactive arm in `left`/`right` mode. Invalid arm
  modes and combining a single-arm mode with `use_ee6d=True` raise clear errors.

## Real-data smoke test

1. Create the distillation loader with the default `use_rollout_data=False`. Inspect several batches and confirm every
   `gt_mask` value is true and no sampled source frame has `collect=rollout`.
2. Repeat with `use_rollout_data=True`. Confirm rollout frames occur and map to `gt_mask=false`; the other three labels
   map to true.
3. Inspect samples around a collect-label transition and confirm the mask follows only the current observation frame.
4. Run at least one mixed-label train step and one all-rollout train step. Confirm all losses are finite and the
   all-rollout step reports `gt_loss=0`.
5. Confirm W&B/console `rollout_fraction` matches the observed mask ratio and checkpoints can still be saved/resumed.
6. Recompute normalization statistics separately for any single-arm mode in use. In `left` mode, confirm dimensions
   `7:14` have zero state/action means and standard deviations; in `right` mode, confirm the same for dimensions `0:7`.
   Confirm transformed teacher/student batches follow the selected mask and inference still returns 14 dimensions with
   the inactive arm exactly zero. Confirm `dual` preserves the previous behavior.

Record the server commit, dataset IDs, config override, accelerator topology, command output, and any failures when
returning results.
