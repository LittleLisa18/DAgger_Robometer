from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from openpi.models import model as _model
from openpi.models import pi0
from openpi.models import pi0_config
from openpi.models import pi0_fast
from openpi.shared import download
from openpi.shared import nnx_utils


def test_pi0_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

    noise = jax.random.normal(key, actions.shape)
    sample_with_solver = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps", "solver"))
    dpm_actions = sample_with_solver(key, obs, noise=noise, num_steps=3, solver="dpmpp_2m")
    repeated_dpm_actions = sample_with_solver(key, obs, noise=noise, num_steps=3, solver="dpmpp_2m")
    assert dpm_actions.shape == actions.shape
    assert jnp.all(jnp.isfinite(dpm_actions))
    assert jnp.array_equal(dpm_actions, repeated_dpm_actions)

    one_step_dpm_actions = sample_with_solver(key, obs, noise=noise, num_steps=1, solver="dpmpp_2m")
    assert one_step_dpm_actions.shape == actions.shape
    assert jnp.all(jnp.isfinite(one_step_dpm_actions))

    with pytest.raises(ValueError, match="num_steps must be at least 1"):
        model.sample_actions(key, obs, num_steps=0)
    with pytest.raises(ValueError, match="Unsupported solver"):
        model.sample_actions(key, obs, solver="unknown")


def test_dpmpp_updates_follow_linear_flow_path():
    target = jnp.array([[1.0, -2.0]], dtype=jnp.float32)
    source = jnp.array([[-0.5, 3.0]], dtype=jnp.float32)
    previous_time = jnp.asarray(0.9, dtype=jnp.float32)
    time = jnp.asarray(0.7, dtype=jnp.float32)
    next_time = jnp.asarray(0.4, dtype=jnp.float32)
    sample = (1.0 - time) * target + time * source
    expected = (1.0 - next_time) * target + next_time * source

    first_order = pi0._dpmpp_first_order_update(sample, target, time, next_time)  # noqa: SLF001
    second_order = pi0._dpmpp_2m_update(sample, target, target, previous_time, time, next_time)  # noqa: SLF001

    heun = pi0._dpmpp_2m_update(  # noqa: SLF001
        sample, target, target, previous_time, time, next_time, solver_type="heun"
    )

    assert jnp.allclose(first_order, expected, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(second_order, expected, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(heun, expected, rtol=1e-5, atol=1e-6)


def test_ode_solver_steps_are_distinct():
    sample = jnp.asarray([1.0], dtype=jnp.float32)
    time = jnp.asarray(1.0, dtype=jnp.float32)
    dt = jnp.asarray(-1.0, dtype=jnp.float32)

    def velocity_fn(x, t):
        return jnp.square(x) + t

    euler = pi0._ode_solver_step(velocity_fn, sample, time, dt, solver="euler")  # noqa: SLF001
    midpoint = pi0._ode_solver_step(velocity_fn, sample, time, dt, solver="midpoint")  # noqa: SLF001
    heun = pi0._ode_solver_step(velocity_fn, sample, time, dt, solver="heun")  # noqa: SLF001

    assert jnp.allclose(euler, jnp.asarray([-1.0]))
    assert jnp.allclose(midpoint, jnp.asarray([0.5]))
    assert jnp.allclose(heun, jnp.asarray([-0.5]))


def test_pi0_ode_solver_types():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(key)

    obs = config.fake_obs(batch_size=1)
    noise = jax.random.normal(key, (1, model.action_horizon, model.action_dim))
    sample_with_solver = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps", "solver"))

    actions_by_solver = {}
    for solver in ("dpmpp_2m", "midpoint", "heun"):
        actions = sample_with_solver(key, obs, noise=noise, num_steps=1, solver=solver)
        assert actions.shape == noise.shape
        assert jnp.all(jnp.isfinite(actions))
        actions_by_solver[solver] = actions

    assert not jnp.array_equal(actions_by_solver["dpmpp_2m"], actions_by_solver["midpoint"])
    assert not jnp.array_equal(actions_by_solver["midpoint"], actions_by_solver["heun"])


def test_pi0_lora_model():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)


def test_pi0_fast_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig()
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)


def test_pi0_fast_lora_model():
    key = jax.random.key(0)
    config = pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora")
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size,)

    actions = nnx_utils.module_jit(model.sample_actions)(key, obs)
    assert actions.shape == (batch_size, 256)

    lora_filter = nnx_utils.PathRegex(".*lora.*")
    model_state = nnx.state(model)

    lora_state_elems = list(model_state.filter(lora_filter))
    assert len(lora_state_elems) > 0


@pytest.mark.manual
def test_model_restore():
    key = jax.random.key(0)
    config = pi0_config.Pi0Config()

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    model = config.load(
        _model.restore_params(download.maybe_download("gs://openpi-assets/checkpoints/pi0_base/params"))
    )

    loss = model.compute_loss(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = model.sample_actions(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)
