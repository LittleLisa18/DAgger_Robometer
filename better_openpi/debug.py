import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import pi0_config
from openpi.models import model as _model
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import config as _config


def test_pi05_model():
    key = jax.random.key(1)
    config = pi0_config.Pi0Config(pi05=True, action_horizon=10)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    assert loss.shape == (batch_size, config.action_horizon)

    actions = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

    print("pi05 model test passed")


def test_pi05_rtc_model():
    key = jax.random.key(1)

    print("== training without delay ==")
    config = pi0_config.Pi0FasterConfig(pi05=True, action_horizon=10)
    model = config.create(key)

    batch_size = 2
    obs, act = config.fake_obs(batch_size), config.fake_act(batch_size)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    print(loss)

    print("== training with delay ==")
    config = pi0_config.Pi0FasterConfig(pi05=True, action_horizon=10, max_delay=5, mix_prob=1.0, alpha=0.6, u0=0.9)
    model = config.create(key)

    loss = nnx_utils.module_jit(model.compute_loss)(key, obs, act)
    print(loss)

    print("== sampling without delay ==")
    actions, _, _ = nnx_utils.module_jit(model.sample_actions, static_argnames=("num_steps",))(key, obs, num_steps=10)
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

    print("== sampling with delay ==")
    action_prefix = jax.random.normal(key, (batch_size, model.action_horizon, model.action_dim))
    delay = jnp.array([3, 5])
    actions, _, _ = nnx_utils.module_jit(
        model.sample_actions, static_argnames=("num_steps", "infer_time_schedule", "alpha", "u0")
    )(
        key,
        obs,
        num_steps=10,
        delay=delay,
        action_prefix=action_prefix,
        infer_time_schedule="HAS",
        alpha=0.6,
        u0=0.9,
    )
    assert actions.shape == (batch_size, model.action_horizon, model.action_dim)

    print("pi05 rtc model test passed")


def test_pi05_rtc_model_infer():
    config = _config.get_config("pi05_agilex")
    checkpoint_dir = "checkpoints/pi05_torch_throw_rubbish_1222/50000"

    # Create a trained policy.
    policy = policy_config.create_trained_policy(
        config, checkpoint_dir, sample_kwargs={"num_steps": 1}
    )

    actions = np.random.uniform(low=-0.01, high=0.01, size=(50, 14))

    # print("== infer without delay ==")
    example = {
        "state": np.random.uniform(low=-0.01, high=0.01, size=(14)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

    for _ in range(10):
        time_start = time.time()
        actions = policy.infer(example)["actions"]
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")
    # print(actions.shape)

    # print("== infer with delay ==")
    # example = {
    #     "state": np.random.uniform(low=-0.01, high=0.01, size=(14)),
    #     "images": {
    #         "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
    #         "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
    #         "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
    #     },
    #     "prompt": "do something",
    #     "action_prefix": actions[:5],
    #     "delay": np.array(5),  # scalar, becomes (1,) after batch transform
    # }
    # actions = policy.infer(example)["actions"]
    # print(actions.shape)

    # for _ in range(10):
    #     time_start = time.time()
    #     policy.infer(example)
    #     time_end = time.time()
    #     print(f"Time taken: {time_end - time_start} seconds")

    # print("== infer streaming ==")

    # def on_actions_ready(actions: np.ndarray):
    #     print(actions.shape)

    # actions = policy.infer_streaming(example, on_actions_ready=on_actions_ready)["actions"]
    # print(actions.shape)

    # for _ in range(10):
    #     time_start = time.time()
    #     policy.infer_streaming(example, on_actions_ready=on_actions_ready)
    #     time_end = time.time()
    #     print(f"Time taken: {time_end - time_start} seconds")


def test_speed():
    config = _config.get_config("pi05_faster_agilex")
    checkpoint_dir = "checkpoints/pi05_ada_const_pick_pp_roll/49999"

    # Create a trained policy.
    policy = policy_config.create_trained_policy(
        config, checkpoint_dir, sample_kwargs={"infer_time_schedule": "HAS", "alpha": 0.6, "u0": 0.9, "num_steps": 1}
    )

    example = {
        "state": np.random.uniform(low=-0.01, high=0.01, size=(14)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }

    num_warmup = 3
    num_trials = 20
    vlm_times = []
    first_step_times = []
    sampling_times = []

    for trial in range(num_warmup + num_trials):
        inputs = jax.tree.map(lambda x: x, example)
        inputs = policy._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)

        policy._rng, sample_rng = jax.random.split(policy._rng)
        sample_kwargs = {k: v for k, v in policy._sample_kwargs.items() if k != "infer_time_schedule"}
        num_steps = sample_kwargs.get("num_steps", 10)

        vlm_start = time.time()
        (
            x_t, already_output, t_starts, dt_schedule,
            is_ready_after_step, kv_cache, prefix_mask,
            prefix_action_mask, action_prefix_init, observation_preprocessed,
        ) = policy._sample_actions_streaming_init(sample_rng, observation, **sample_kwargs)
        jax.block_until_ready(kv_cache)
        vlm_time = time.time() - vlm_start

        sampling_start = time.time()
        for i in range(num_steps):
            x_next, already_output, newly_ready = policy._sample_actions_streaming_step(
                x_t, already_output, t_starts[i], dt_schedule[i],
                is_ready_after_step[i], kv_cache, prefix_mask,
                prefix_action_mask, action_prefix_init, observation_preprocessed,
            )
            x_t = x_next
            # jax.block_until_ready(x_t)

            if i == 0:
                # jax.block_until_ready(x_t)
                first_step_time = time.time() - sampling_start

        jax.block_until_ready(x_t)
        sampling_time = time.time() - sampling_start
        # first_step_time = 0

        if trial < num_warmup:
            print(f"[warmup {trial+1}/{num_warmup}] VLM: {vlm_time*1000:.1f}ms | 1st step: {first_step_time*1000:.1f}ms | All sampling: {sampling_time*1000:.1f}ms")
        else:
            vlm_times.append(vlm_time)
            first_step_times.append(first_step_time)
            sampling_times.append(sampling_time)
            print(f"[{trial - num_warmup + 1}/{num_trials}] VLM: {vlm_time*1000:.1f}ms | 1st step: {first_step_time*1000:.1f}ms | All sampling: {sampling_time*1000:.1f}ms")

    print(f"\n=== Average over {num_trials} trials ===")
    print(f"VLM:          {np.mean(vlm_times)*1000:.1f} ± {np.std(vlm_times)*1000:.1f} ms")
    print(f"1st step:     {np.mean(first_step_times)*1000:.1f} ± {np.std(first_step_times)*1000:.1f} ms")
    print(f"All sampling: {np.mean(sampling_times)*1000:.1f} ± {np.std(sampling_times)*1000:.1f} ms")
    print(f"Total:        {(np.mean(vlm_times) + np.mean(sampling_times))*1000:.1f} ± {(np.std(vlm_times) + np.std(sampling_times))*1000:.1f} ms")

if __name__ == "__main__":
    # test_pi05_model()
    # test_pi05_rtc_model()
    test_pi05_rtc_model_infer()
    # test_speed()
