from collections.abc import Callable
import logging
from typing import Literal

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")

_DPM_T_MAX = 0.999


def _ode_solver_step(
    velocity_fn: Callable[[at.Float[at.Array, "*b"], at.Float[at.Array, ""]], at.Float[at.Array, "*b"]],
    sample: at.Float[at.Array, "*b"],
    time: at.Float[at.Array, ""],
    dt: at.Float[at.Array, ""],
    *,
    solver: Literal["euler", "midpoint", "heun"],
) -> at.Float[at.Array, "*b"]:
    """Advance one ODE interval with an explicit Euler or second-order Runge--Kutta method."""
    initial_velocity = velocity_fn(sample, time)
    if solver == "euler":
        return sample + dt * initial_velocity
    if solver == "midpoint":
        midpoint_sample = sample + 0.5 * dt * initial_velocity
        midpoint_velocity = velocity_fn(midpoint_sample, time + 0.5 * dt)
        return sample + dt * midpoint_velocity
    if solver == "heun":
        euler_sample = sample + dt * initial_velocity
        final_velocity = velocity_fn(euler_sample, time + dt)
        return sample + 0.5 * dt * (initial_velocity + final_velocity)
    raise ValueError(f"Unsupported ODE solver type: {solver!r}")


def _flow_log_snr(time: at.Float[at.Array, ""]) -> at.Float[at.Array, ""]:
    """Return log(alpha_t / sigma_t) for the linear flow schedule."""
    return jnp.log1p(-time) - jnp.log(time)


def _dpmpp_first_order_update(
    sample: at.Float[at.Array, "*b"],
    x0_prediction: at.Float[at.Array, "*b"],
    time: at.Float[at.Array, ""],
    next_time: at.Float[at.Array, ""],
) -> at.Float[at.Array, "*b"]:
    """Apply a first-order DPM-Solver++ update for the linear flow schedule."""
    alpha_next = 1.0 - next_time
    sigma, sigma_next = time, next_time
    h = _flow_log_snr(next_time) - _flow_log_snr(time)
    return (sigma_next / sigma) * sample - alpha_next * jnp.expm1(-h) * x0_prediction


def _dpmpp_2m_update(
    sample: at.Float[at.Array, "*b"],
    x0_prediction: at.Float[at.Array, "*b"],
    previous_x0_prediction: at.Float[at.Array, "*b"],
    previous_time: at.Float[at.Array, ""],
    time: at.Float[at.Array, ""],
    next_time: at.Float[at.Array, ""],
    solver_type: Literal["midpoint", "heun"] = "midpoint",
) -> at.Float[at.Array, "*b"]:
    """Apply a second-order multistep DPM-Solver++ midpoint or Heun update."""
    lambda_previous = _flow_log_snr(previous_time)
    lambda_current = _flow_log_snr(time)
    lambda_next = _flow_log_snr(next_time)
    h = lambda_next - lambda_current
    h_previous = lambda_current - lambda_previous
    r = h_previous / h
    first_derivative = (x0_prediction - previous_x0_prediction) / r

    alpha_next = 1.0 - next_time
    sigma, sigma_next = time, next_time
    phi_1 = jnp.expm1(-h)
    if solver_type == "midpoint":
        correction = -0.5 * alpha_next * phi_1 * first_derivative
    elif solver_type == "heun":
        correction = alpha_next * (phi_1 / h + 1.0) * first_derivative
    else:
        raise ValueError(f"Unsupported DPM-Solver++ 2M solver type: {solver_type!r}")

    return (sigma_next / sigma) * sample - alpha_next * phi_1 * x0_prediction + correction


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


@at.typecheck
def _masked_gt_loss(
    prediction: at.Float[at.Array, "*b ah ad"],
    target: at.Float[at.Array, "*b ah ad"],
    gt_mask: at.Bool[at.Array, "*b"],
) -> at.Float[at.Array, ""]:
    """Average per-sample flow-matching loss over the full batch after applying the GT mask."""
    per_sample_loss = jnp.mean(jnp.square(prediction - target), axis=(-2, -1))
    return jnp.mean(per_sample_loss * gt_mask.astype(per_sample_loss.dtype))


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b a emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb[:, None, :]
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    def _forward_with_features(
        self,
        observation: _model.Observation,
        x_t: _model.Actions,
        time: at.Float[at.Array, " b"],
    ) -> tuple[
        dict[str, at.Float[at.Array, "b s emb"]],
        at.Float[at.Array, "b _prefix emb"],
        at.Float[at.Array, "b _suffix emb"],
        at.Float[at.Array, "b ah ad"],
    ]:
        """Forward pass that also returns intermediate ViT and LLM features for distillation."""
        tokens = []
        input_mask = []
        ar_mask = []
        vit_features = {}
        for name in observation.images:
            image_tokens, _ = self.PaliGemma.img(observation.images[name], train=False)
            vit_features[name] = image_tokens
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(observation.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            )
            ar_mask += [False] * image_tokens.shape[1]
        if observation.tokenized_prompt is not None:
            lang = self.PaliGemma.llm(observation.tokenized_prompt, method="embed")
            tokens.append(lang)
            input_mask.append(observation.tokenized_prompt_mask)
            ar_mask += [False] * lang.shape[1]
        prefix_tokens = jnp.concatenate(tokens, axis=1)
        prefix_mask = jnp.concatenate(input_mask, axis=1)
        prefix_ar = jnp.array(ar_mask)

        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = self.embed_suffix(observation, x_t, time)
        full_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        full_ar = jnp.concatenate([prefix_ar, suffix_ar], axis=0)
        attn_mask = make_attn_mask(full_mask, full_ar)
        positions = jnp.cumsum(full_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        return vit_features, prefix_out, suffix_out, v_t

    def compute_distill_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        teacher: "Pi0",
        *,
        gt_mask: at.Bool[at.Array, "*b"],
        vit_weight: float = 1.0,
        llm_weight: float = 1.0,
        flow_weight: float = 1.0,
        gt_weight: float = 1.0,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, ""], dict[str, at.Float[at.Array, ""]]]:
        """Distillation loss at ViT, LLM, and flow matching levels plus ground-truth loss."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        s_vit, s_prefix_out, s_suffix_out, s_v_t = self._forward_with_features(observation, x_t, time)
        t_vit, t_prefix_out, t_suffix_out, t_v_t = jax.lax.stop_gradient(
            teacher._forward_with_features(observation, x_t, time)
        )

        vit_losses = [jnp.mean(jnp.square(s_vit[k] - t_vit[k])) for k in s_vit]
        vit_loss = sum(vit_losses) / len(vit_losses)

        llm_loss = (
            jnp.mean(jnp.square(s_prefix_out - t_prefix_out))
            + jnp.mean(jnp.square(s_suffix_out - t_suffix_out))
        ) / 2.0

        flow_loss = jnp.mean(jnp.square(s_v_t - t_v_t), axis=-1).mean()

        # Rollout samples are masked out, but the reduction still divides by the full batch size.
        gt_loss = _masked_gt_loss(s_v_t, u_t, gt_mask)

        total_loss = (
            vit_weight * vit_loss
            + llm_weight * llm_loss
            + flow_weight * flow_loss
            + gt_weight * gt_loss
        )
        loss_dict = {
            "vit_loss": vit_loss,
            "llm_loss": llm_loss,
            "flow_loss": flow_loss,
            "gt_loss": gt_loss,
            "rollout_fraction": 1.0 - jnp.mean(gt_mask.astype(jnp.float32)),
        }
        return total_loss, loss_dict

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        solver: Literal["euler", "dpmpp_2m", "midpoint", "heun"] = "euler",
    ) -> _model.Actions:
        """Sample actions with an ODE solver or multistep DPM-Solver++.

        ``num_steps`` is the number of integration intervals. Euler and DPM-Solver++ 2M use one model
        evaluation per interval, while midpoint and Heun are two-stage RK2 methods and use two.
        """
        if isinstance(num_steps, int) and num_steps < 1:
            raise ValueError(f"num_steps must be at least 1, got {num_steps}")
        if solver not in ("euler", "dpmpp_2m", "midpoint", "heun"):
            raise ValueError(
                f"Unsupported solver {solver!r}; expected 'euler', 'dpmpp_2m', 'midpoint', or 'heun'"
            )

        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def denoise_step(x_t, time):
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            return self.action_out_proj(suffix_out[:, -self.action_horizon :])

        if solver in ("euler", "midpoint", "heun"):

            def ode_step(carry):
                x_t, step_index = carry
                step_fraction = step_index.astype(jnp.float32) / num_steps
                next_step_fraction = (step_index + 1).astype(jnp.float32) / num_steps
                time = 1.0 - step_fraction
                next_time = 1.0 - next_step_fraction
                x_next = _ode_solver_step(
                    denoise_step,
                    x_t,
                    time,
                    next_time - time,
                    solver=solver,
                )
                return x_next, step_index + 1

            def ode_cond(carry):
                return carry[1] < num_steps

            initial_carry = (noise, jnp.asarray(0, dtype=jnp.int32))
            x_0, _ = jax.lax.while_loop(ode_cond, ode_step, initial_carry)
            return x_0

        dpm_t_max = jnp.asarray(_DPM_T_MAX, dtype=jnp.float32)

        def dpm_step(carry):
            x_t, previous_x0_prediction, previous_time, step_index = carry
            step_fraction = step_index.astype(jnp.float32) / num_steps
            next_step_fraction = (step_index + 1).astype(jnp.float32) / num_steps
            time = dpm_t_max * (1.0 - step_fraction)
            next_time = dpm_t_max * (1.0 - next_step_fraction)

            v_t = denoise_step(x_t, time)
            x0_prediction = x_t - time * v_t

            is_first_step = step_index == 0
            is_last_step = step_index == num_steps - 1

            def nonfinal_update(_):
                return jax.lax.cond(
                    is_first_step,
                    lambda: _dpmpp_first_order_update(x_t, x0_prediction, time, next_time),
                    lambda: _dpmpp_2m_update(
                        x_t,
                        x0_prediction,
                        previous_x0_prediction,
                        previous_time,
                        time,
                        next_time,
                        solver_type="midpoint",
                    ),
                )

            # At t=0, the DPM-Solver++ first-order limit is exactly the current data prediction.
            x_next = jax.lax.cond(is_last_step, lambda _: x0_prediction, nonfinal_update, operand=None)
            return x_next, x0_prediction, time, step_index + 1

        def dpm_cond(carry):
            return carry[3] < num_steps

        initial_carry = (noise, jnp.zeros_like(noise), dpm_t_max, jnp.asarray(0, dtype=jnp.int32))
        x_0, _, _, _ = jax.lax.while_loop(dpm_cond, dpm_step, initial_carry)
        return x_0
