import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def _get_frozen_state_vit(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter_vit()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_freeze_vit():
    config = _pi0_config.Pi0Config(pi05=True)
    state = _get_frozen_state_vit(config)
    print(state.keys())
    assert all("img" in p for p in state)
    assert all("llm" not in p for p in state)


def test_masked_gt_loss_uses_full_batch_denominator():
    prediction = jnp.asarray([[[1.0], [1.0]], [[3.0], [3.0]]])
    target = jnp.zeros_like(prediction)

    all_supervised = _pi0._masked_gt_loss(prediction, target, jnp.asarray([True, True]))
    mixed = _pi0._masked_gt_loss(prediction, target, jnp.asarray([True, False]))
    all_rollout = jax.jit(_pi0._masked_gt_loss)(prediction, target, jnp.asarray([False, False]))

    np.testing.assert_allclose(all_supervised, 5.0)
    # The supervised sample has loss 1.0 and is divided by the complete batch size of 2.
    np.testing.assert_allclose(mixed, 0.5)
    np.testing.assert_allclose(all_rollout, 0.0)


def test_all_rollout_batch_keeps_distillation_loss_but_zeroes_gt_loss():
    class FakeModel:
        def __init__(self, velocity):
            self.velocity = velocity

        def _forward_with_features(self, observation, x_t, time):
            del observation, time
            features = jnp.zeros((x_t.shape[0], 1, 1), dtype=x_t.dtype)
            return {"camera": features}, features, features, jnp.full_like(x_t, self.velocity)

    images = {
        key: jnp.zeros((2, 224, 224, 3), dtype=jnp.float32)
        for key in _model.IMAGE_KEYS
    }
    observation = _model.Observation(
        images=images,
        image_masks={key: jnp.ones((2,), dtype=jnp.bool_) for key in images},
        state=jnp.zeros((2, 1), dtype=jnp.float32),
    )
    actions = jnp.zeros((2, 2, 1), dtype=jnp.float32)

    total_loss, info = _pi0.Pi0.compute_distill_loss(
        FakeModel(1.0),
        jax.random.key(0),
        observation,
        actions,
        FakeModel(2.0),
        gt_mask=jnp.asarray([False, False]),
    )

    np.testing.assert_allclose(info["flow_loss"], 1.0)
    np.testing.assert_allclose(info["gt_loss"], 0.0)
    np.testing.assert_allclose(info["rollout_fraction"], 1.0)
    np.testing.assert_allclose(total_loss, 1.0)
