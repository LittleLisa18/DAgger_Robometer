import functools

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as _model
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import utils as training_utils

from . import train_distill


class _FakeDistillModel(_model.BaseModel):
    def __init__(self, rngs: nnx.Rngs):
        super().__init__(action_dim=1, action_horizon=1, max_token_len=1)
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (1, 1)))
        self.deterministic = True

    def compute_distill_loss(self, rng, observation, actions, teacher, *, gt_mask, **kwargs):
        del rng, observation, actions, teacher, kwargs
        loss = jnp.mean(jnp.square(self.weight.value - gt_mask[:, None].astype(self.weight.value.dtype)))
        rollout_fraction = 1.0 - jnp.mean(gt_mask.astype(jnp.float32))
        return loss, {
            "vit_loss": loss,
            "llm_loss": loss,
            "flow_loss": loss,
            "gt_loss": loss,
            "rollout_fraction": rollout_fraction,
            "gt_mask_sum": jnp.sum(gt_mask),
        }

    def compute_loss(self, rng, observation, actions, *, train=False):
        del rng, observation, train
        return jnp.zeros(actions.shape[:-1])

    def sample_actions(self, rng, observation, **kwargs):
        del rng, kwargs
        return jnp.zeros((observation.state.shape[0], 1, 1))


def test_distill_train_step_passes_jitted_gt_mask_and_logs_rollout_fraction():
    config = _config.TrainConfig(
        name="distill_test",
        exp_name="distill_test",
        model=pi0_config.Pi0Config(
            action_dim=1,
            action_horizon=1,
            max_token_len=1,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
        ),
        distill_config=_config.DistillConfig(teacher_checkpoint="unused"),
        ema_decay=None,
    )
    assert config.distill_config is not None
    assert not config.distill_config.use_rollout_data
    student = _FakeDistillModel(nnx.Rngs(0))
    student_def, student_params = nnx.split(student)
    tx = optax.sgd(1e-2)
    trainable_params = student_params.filter(config.trainable_filter)
    state = training_utils.TrainState(
        step=0,
        params=student_params,
        model_def=student_def,
        tx=tx,
        opt_state=tx.init(trainable_params),
        ema_decay=None,
        ema_params=None,
    )
    teacher = _FakeDistillModel(nnx.Rngs(1))
    teacher_def, teacher_params = nnx.split(teacher)
    observation = _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((2, 1), dtype=jnp.float32),
    )
    batch = (
        observation,
        jnp.zeros((2, 1, 1), dtype=jnp.float32),
        jnp.asarray([True, False]),
    )

    step_fn = jax.jit(functools.partial(train_distill.distill_train_step, config, teacher_def))
    new_state, info = step_fn(jax.random.key(0), state, batch, teacher_params)

    assert int(new_state.step) == 1
    np.testing.assert_allclose(info["gt_mask_sum"], 1.0)
    np.testing.assert_allclose(info["rollout_fraction"], 0.5)
    assert {"vit_loss", "llm_loss", "flow_loss", "gt_loss"}.issubset(info)
