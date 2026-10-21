from typing import Optional, Protocol, Tuple
from dataclasses import dataclass

import numpy as np
import jax, jax.numpy as jnp
import optax

class DiffusionModel(Protocol):
    def __call__(self, t: jax.Array, x: jax.Array) -> jax.Array:
        ...

@dataclass(frozen=True)
class BetaScheduleCoefficients:
    betas: jax.Array
    alphas: jax.Array
    alphas_cumprod: jax.Array
    alphas_cumprod_prev: jax.Array
    sqrt_alphas_cumprod: jax.Array
    sqrt_one_minus_alphas_cumprod: jax.Array
    log_one_minus_alphas_cumprod: jax.Array
    sqrt_recip_alphas_cumprod: jax.Array
    sqrt_recipm1_alphas_cumprod: jax.Array
    posterior_variance: jax.Array
    posterior_log_variance_clipped: jax.Array
    posterior_mean_coef1: jax.Array
    posterior_mean_coef2: jax.Array

    @staticmethod
    def from_beta(betas: np.ndarray):
        # 确保betas是float32
        betas = betas.astype(np.float32)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        # calculations for diffusion q(x_t | x_{t-1}) and others
        sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = np.sqrt(1. - alphas_cumprod)
        log_one_minus_alphas_cumprod = np.log(1. - alphas_cumprod)
        sqrt_recip_alphas_cumprod = np.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = np.sqrt(1. / alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = np.log(np.maximum(posterior_variance, 1e-20))
        posterior_mean_coef1 = betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)

        return BetaScheduleCoefficients(
            *jax.device_put((
                betas, alphas, alphas_cumprod, alphas_cumprod_prev,
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, log_one_minus_alphas_cumprod,
                sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod,
                posterior_variance, posterior_log_variance_clipped, posterior_mean_coef1, posterior_mean_coef2
            ))
        )

    @staticmethod
    def vp_beta_schedule(timesteps: int):
        t = np.arange(1, timesteps + 1)
        T = timesteps
        b_max = 10.
        b_min = 0.1
        alpha = np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
        betas = 1 - alpha
        return betas

    @staticmethod
    def cosine_beta_schedule(timesteps: int):
        s = 0.008
        t = np.arange(0, timesteps + 1) / timesteps
        alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
        alphas_cumprod /= alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = np.clip(betas, 0, 0.999)
        return betas

@dataclass(frozen=True)
class GaussianDiffusion:
    num_timesteps: int

    def beta_schedule(self):
        with jax.ensure_compile_time_eval():
            betas = BetaScheduleCoefficients.vp_beta_schedule(self.num_timesteps)
            return BetaScheduleCoefficients.from_beta(betas)

    def p_mean_variance(self, t: int, x: jax.Array, noise_pred: jax.Array):
        B = self.beta_schedule()
        x_recon = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        x_recon = jnp.clip(x_recon, -1, 1)
        model_mean = x_recon * B.posterior_mean_coef1[t] + x * B.posterior_mean_coef2[t]
        model_log_variance = B.posterior_log_variance_clipped[t]
        return model_mean, model_log_variance

    def p_sample(self, key: jax.Array, model: DiffusionModel, shape: Tuple[int, ...]) -> jax.Array:
        """Sample from diffusion model"""
        assert self.num_timesteps > 0, "num_timesteps must be positive"
        assert len(shape) >= 1, "shape must be at least 1D"
        
        x_key, noise_key = jax.random.split(key)
        x = jax.random.normal(x_key, shape)
        noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))

        def body_fn(x, input):
            t, noise = input
            noise_pred = model(t, x)
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            x = model_mean + (t > 0) * jnp.exp(0.5 * model_log_variance) * noise
            return x, None

        t = jnp.arange(self.num_timesteps)[::-1]
        x, _ = jax.lax.scan(body_fn, x, (t, noise))
        return x

    def p_sample_deterministic(self, key: jax.Array, model: DiffusionModel, shape: Tuple[int, ...]) -> jax.Array:
        """Deterministic reverse diffusion with an explicit latent seed.

        We still start from a Gaussian latent so evaluation stays on the same
        initialization family as training-time sampling, but the caller now
        chooses the seed. This avoids coupling every state to one global fixed
        latent while keeping evaluation repeatable.
        """
        assert self.num_timesteps > 0, "num_timesteps must be positive"
        assert len(shape) >= 1, "shape must be at least 1D"

        x = jax.random.normal(key, shape)

        def body_fn(x, t):
            noise_pred = model(t, x)
            model_mean, _ = self.p_mean_variance(t, x, noise_pred)
            return model_mean, None

        t = jnp.arange(self.num_timesteps)[::-1]
        x, _ = jax.lax.scan(body_fn, x, t)
        return x

    def predict_x0(self, t: int, x: jax.Array, noise_pred: jax.Array) -> jax.Array:
        """Estimate the clean action from x_t via Tweedie's formula."""
        B = self.beta_schedule()
        x0 = x * B.sqrt_recip_alphas_cumprod[t] - noise_pred * B.sqrt_recipm1_alphas_cumprod[t]
        return jnp.clip(x0, -1.0, 1.0)

    def posterior_mean_from_x0(self, t: int, x_t: jax.Array, x0: jax.Array) -> jax.Array:
        """Compute q(x_{t-1} | x_t, x0) mean for an externally modified clean action.

        This is the path used by DOVE/UPV-GDS when the proxy-value gradient is
        injected in clean-action space: x_t -> x0_hat -> x0_guided -> x_{t-1}.
        """
        B = self.beta_schedule()
        x0 = jnp.clip(x0, -1.0, 1.0)
        return x0 * B.posterior_mean_coef1[t] + x_t * B.posterior_mean_coef2[t]

    def p_sample_value_guided(
        self,
        key: jax.Array,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        q_grad_value_fn,
        q_uncertainty_fn,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: Optional[jax.Array] = None,
        max_action_shift: float = -1.0,
    ) -> jax.Array:
        action, _ = self.p_sample_value_guided_with_metrics(
            key,
            model,
            shape,
            q_grad_value_fn,
            q_uncertainty_fn,
            sigma_ref,
            lambda_0=lambda_0,
            beta_unc=beta_unc,
            p_decay=p_decay,
            grad_clip=grad_clip,
            guidance_mode=guidance_mode,
            guidance_target=guidance_target,
            guidance_step_interval=guidance_step_interval,
            guidance_injection=guidance_injection,
            guidance_schedule=guidance_schedule,
            critic_objective=critic_objective,
            anchor_action=anchor_action,
            max_action_shift=max_action_shift,
        )
        return action

    def p_sample_value_guided_deterministic(
        self,
        key: jax.Array,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        q_grad_value_fn,
        q_uncertainty_fn,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: Optional[jax.Array] = None,
        max_action_shift: float = -1.0,
    ) -> jax.Array:
        action, _ = self.p_sample_value_guided_deterministic_with_metrics(
            key,
            model,
            shape,
            q_grad_value_fn,
            q_uncertainty_fn,
            sigma_ref,
            lambda_0=lambda_0,
            beta_unc=beta_unc,
            p_decay=p_decay,
            grad_clip=grad_clip,
            guidance_mode=guidance_mode,
            guidance_target=guidance_target,
            guidance_step_interval=guidance_step_interval,
            guidance_injection=guidance_injection,
            guidance_schedule=guidance_schedule,
            critic_objective=critic_objective,
            anchor_action=anchor_action,
            max_action_shift=max_action_shift,
        )
        return action

    def p_sample_value_guided_with_metrics(
        self,
        key: jax.Array,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        q_grad_value_fn,
        q_uncertainty_fn,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: Optional[jax.Array] = None,
        max_action_shift: float = -1.0,
    ) -> Tuple[jax.Array, dict]:
        return self._p_sample_value_guided_impl(
            key,
            model,
            shape,
            q_grad_value_fn,
            q_uncertainty_fn,
            sigma_ref,
            lambda_0=lambda_0,
            beta_unc=beta_unc,
            p_decay=p_decay,
            grad_clip=grad_clip,
            guidance_mode=guidance_mode,
            guidance_target=guidance_target,
            guidance_step_interval=guidance_step_interval,
            guidance_injection=guidance_injection,
            guidance_schedule=guidance_schedule,
            critic_objective=critic_objective,
            anchor_action=anchor_action,
            max_action_shift=max_action_shift,
            deterministic=False,
        )

    def p_sample_value_guided_deterministic_with_metrics(
        self,
        key: jax.Array,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        q_grad_value_fn,
        q_uncertainty_fn,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: Optional[jax.Array] = None,
        max_action_shift: float = -1.0,
    ) -> Tuple[jax.Array, dict]:
        return self._p_sample_value_guided_impl(
            key,
            model,
            shape,
            q_grad_value_fn,
            q_uncertainty_fn,
            sigma_ref,
            lambda_0=lambda_0,
            beta_unc=beta_unc,
            p_decay=p_decay,
            grad_clip=grad_clip,
            guidance_mode=guidance_mode,
            guidance_target=guidance_target,
            guidance_step_interval=guidance_step_interval,
            guidance_injection=guidance_injection,
            guidance_schedule=guidance_schedule,
            critic_objective=critic_objective,
            anchor_action=anchor_action,
            max_action_shift=max_action_shift,
            deterministic=True,
        )

    def _p_sample_value_guided_impl(
        self,
        key: jax.Array,
        model: DiffusionModel,
        shape: Tuple[int, ...],
        q_grad_value_fn,
        q_uncertainty_fn,
        sigma_ref: jax.Array,
        lambda_0: float,
        beta_unc: float,
        p_decay: float,
        grad_clip: float,
        guidance_mode: int,
        guidance_target: int,
        guidance_step_interval: int,
        guidance_injection: int,
        guidance_schedule: int,
        critic_objective: int,
        anchor_action: Optional[jax.Array],
        max_action_shift: float,
        deterministic: bool,
    ) -> Tuple[jax.Array, dict]:
        """Reverse diffusion with DOVE/UPV-GDS value guidance.

        Two injection paths are supported:
        - guidance_injection == 0 (clean_x0): the main paper path. The critic
          gradient is taken with respect to the Tweedie clean-action estimate,
          the clean action is shifted/clipped, then x_{t-1} is reconstructed from
          q(x_{t-1}|x_t, x0_guided).
        - guidance_injection == 1 (latent_mean): the older implementation. The
          critic gradient is backpropagated to x_t and added to the reverse mean.

        guidance_schedule == 0 uses the paper-aligned noise-level schedule
        ((t+1)/T)^p; guidance_schedule == 1 keeps the previous alpha_cumprod^p
        schedule for ablation/backward compatibility.
        """
        assert self.num_timesteps > 0, "num_timesteps must be positive"
        assert len(shape) >= 1, "shape must be at least 1D"
        B = self.beta_schedule()

        x_key, noise_key = jax.random.split(key)
        x = jax.random.normal(x_key, shape)
        noise = jax.random.normal(noise_key, (self.num_timesteps, *shape))
        grad_noise = jax.random.normal(jax.random.fold_in(noise_key, 17), (self.num_timesteps, *shape))
        sigma_ref = jnp.maximum(jnp.asarray(sigma_ref, dtype=jnp.float32), 1e-6)
        lambda_0 = jnp.asarray(lambda_0, dtype=jnp.float32)
        beta_unc = jnp.asarray(beta_unc, dtype=jnp.float32)
        p_decay = jnp.asarray(p_decay, dtype=jnp.float32)
        grad_clip = jnp.maximum(jnp.asarray(grad_clip, dtype=jnp.float32), 1e-6)
        guidance_mode = jnp.asarray(guidance_mode, dtype=jnp.int32)
        guidance_target = jnp.asarray(guidance_target, dtype=jnp.int32)
        guidance_step_interval = jnp.asarray(guidance_step_interval, dtype=jnp.int32)
        guidance_injection = jnp.asarray(guidance_injection, dtype=jnp.int32)
        guidance_schedule = jnp.asarray(guidance_schedule, dtype=jnp.int32)
        critic_objective = jnp.asarray(critic_objective, dtype=jnp.float32)
        objective_sign = jnp.where(critic_objective > 0.5, jnp.float32(1.0), jnp.float32(-1.0))
        max_action_shift = jnp.asarray(max_action_shift, dtype=jnp.float32)
        use_anchor_projection = bool(anchor_action is not None)
        if use_anchor_projection:
            anchor_action = jnp.asarray(anchor_action, dtype=jnp.float32)

        def _flatten_action_like(g):
            return g.reshape((-1, g.shape[-1]))

        def _clip_by_norm(g):
            nan_frac = jnp.mean((~jnp.isfinite(g)).astype(jnp.float32))
            g = jnp.where(jnp.isfinite(g), g, jnp.zeros_like(g))
            flat = _flatten_action_like(g)
            norms = jnp.linalg.norm(flat, axis=-1, keepdims=True)
            scales = jnp.minimum(1.0, grad_clip / (norms + 1e-6))
            clipped = (flat * scales).reshape(g.shape)
            clip_frac = jnp.mean((norms.squeeze(-1) > grad_clip).astype(jnp.float32))
            post_norm = jnp.mean(jnp.linalg.norm(_flatten_action_like(clipped), axis=-1))
            pre_norm = jnp.mean(norms.squeeze(-1))
            return clipped, pre_norm, post_norm, clip_frac, nan_frac

        def _mode_select_grad(real_grad, rand_dir):
            real_grad, pre_norm, post_norm, clip_frac, nan_frac = _clip_by_norm(real_grad)
            rand_dir = jnp.where(jnp.isfinite(rand_dir), rand_dir, jnp.zeros_like(rand_dir))
            rand_flat = _flatten_action_like(rand_dir)
            rand_norm = jnp.linalg.norm(rand_flat, axis=-1, keepdims=True)
            real_flat = _flatten_action_like(real_grad)
            real_norm = jnp.linalg.norm(real_flat, axis=-1, keepdims=True)
            random_grad = (rand_flat * (real_norm / (rand_norm + 1e-6))).reshape(real_grad.shape)
            # The incoming gradient is taken on an objective-normalized proxy
            # score that should be minimized. For cost critics this is C; for
            # value critics this is -Q, so proxy_value increases Q.
            proxy_grad = -real_grad
            opposite_grad = real_grad
            grad = jnp.where(guidance_mode == 1, random_grad, proxy_grad)
            grad = jnp.where(guidance_mode == 2, jnp.zeros_like(grad), grad)
            grad = jnp.where(guidance_mode == 3, opposite_grad, grad)
            return grad, pre_norm, post_norm, clip_frac, nan_frac

        def _expand_like(v, ref):
            while v.ndim < ref.ndim:
                v = jnp.expand_dims(v, -1)
            return v

        def _project_to_anchor(a_candidate):
            # Surrogate prior-compatible trust region: ||a-a0||_2 <= d_max.
            # This projection keeps the UPV-GDS proposal near the vanilla
            # diffusion anchor; Final Gate later realizes the empirical
            # acceptance set with prior, proxy-risk, shift, and uncertainty checks.
            a_candidate = jnp.clip(a_candidate, -1.0, 1.0)
            if not use_anchor_projection:
                return a_candidate, jnp.float32(0.0), jnp.float32(0.0)

            radius = jnp.maximum(max_action_shift, jnp.float32(0.0))
            delta = a_candidate - anchor_action
            flat_delta = _flatten_action_like(delta)
            norms = jnp.linalg.norm(flat_delta, axis=-1, keepdims=True)
            scales = jnp.minimum(jnp.float32(1.0), radius / (norms + 1e-6))
            projected = anchor_action + (flat_delta * scales).reshape(delta.shape)
            projected = jnp.clip(projected, -1.0, 1.0)
            projected_delta = projected - anchor_action
            projected_norm = jnp.mean(jnp.linalg.norm(_flatten_action_like(projected_delta), axis=-1))
            projection_rate = jnp.mean((norms.squeeze(-1) > radius + 1e-6).astype(jnp.float32))
            return projected, projection_rate, projected_norm

        def _eta_t(t, apply_step):
            # t is scanned in descending order, so large t corresponds to early
            # high-noise reverse steps. This implements phi(t)=((t+1)/T)^p.
            time_level = (t.astype(jnp.float32) + 1.0) / jnp.maximum(float(self.num_timesteps), 1.0)
            eta_noise = time_level ** p_decay
            eta_alpha = B.alphas_cumprod[t] ** p_decay
            eta = jnp.where(guidance_schedule == 1, eta_alpha, eta_noise)
            return jnp.where(apply_step, eta, jnp.float32(0.0))


        def guided_step(x_t, t, rand_dir, eps_for_x0):
            a_hat = self.predict_x0(t, x_t, eps_for_x0)
            q_input_for_info = jnp.where(guidance_target == 1, x_t, a_hat)
            q_val = q_grad_value_fn(q_input_for_info)
            proxy_val = objective_sign * q_val
            sig = jax.lax.stop_gradient(q_uncertainty_fn(q_input_for_info))
            lam = lambda_0 * jnp.exp(-beta_unc * sig / sigma_ref)
            lam_mean = jnp.mean(lam)

            def scalar_q_action(a_inner):
                return jnp.sum(objective_sign * q_grad_value_fn(a_inner))

            def scalar_q_latent(x_inner):
                eps_inner = model(t, x_inner)
                a_inner = self.predict_x0(t, x_inner, eps_inner)
                q_input = jnp.where(guidance_target == 1, x_inner, a_inner)
                return jnp.sum(objective_sign * q_grad_value_fn(q_input))

            action_grad = jax.grad(scalar_q_action)(a_hat)
            latent_grad = jax.grad(scalar_q_latent)(x_t)
            action_grad, action_pre_norm, action_post_norm, action_clip_frac, action_nan_frac = _mode_select_grad(action_grad, rand_dir)
            latent_grad, latent_pre_norm, latent_post_norm, latent_clip_frac, latent_nan_frac = _mode_select_grad(latent_grad, rand_dir)

            apply_step = jnp.where(
                guidance_step_interval == 0,
                t == 0,
                (t % jnp.maximum(guidance_step_interval, 1)) == 0,
            )
            eta = _eta_t(t, apply_step)
            action_lam = _expand_like(lam, action_grad)
            latent_lam = _expand_like(lam, latent_grad)
            action_delta = eta * action_lam * action_grad
            latent_delta = eta * latent_lam * latent_grad

            use_clean = jnp.logical_and(guidance_injection == 0, guidance_target == 0)
            a_guided_raw = a_hat + action_delta
            a_guided_projected, anchor_projection_rate, anchor_shift = _project_to_anchor(a_guided_raw)
            # Only the paper-aligned clean-x0 path can be projected around a0;
            # latent-mean guidance keeps its original latent-space update.
            a_guided = jnp.where(use_clean, a_guided_projected, jnp.clip(a_guided_raw, -1.0, 1.0))
            clean_mean = self.posterior_mean_from_x0(t, x_t, a_guided)
            latent_mean_delta = jnp.exp(0.5 * B.posterior_log_variance_clipped[t]) * latent_delta
            delta_for_shift = jnp.where(use_clean, a_guided - a_hat, latent_delta)
            grad_norm = jnp.where(use_clean, action_post_norm, latent_post_norm)
            grad_pre_norm = jnp.where(use_clean, action_pre_norm, latent_pre_norm)
            clip_frac = jnp.where(use_clean, action_clip_frac, latent_clip_frac)
            nan_frac = jnp.where(use_clean, action_nan_frac, latent_nan_frac)
            x0_clip_frac = jnp.mean((jnp.abs(a_hat) >= 0.999).astype(jnp.float32))
            shift = jnp.mean(jnp.linalg.norm(_flatten_action_like(delta_for_shift), axis=-1))

            return (
                clean_mean,
                latent_mean_delta,
                use_clean.astype(jnp.float32),
                grad_norm,
                grad_pre_norm,
                clip_frac,
                nan_frac,
                lam_mean,
                jnp.mean(sig),
                jnp.mean(q_val),
                jnp.mean(proxy_val),
                eta,
                x0_clip_frac,
                shift,
                anchor_projection_rate,
                anchor_shift,
                apply_step.astype(jnp.float32),
            )

        def body_fn(carry, inp):
            (x, grad_norm_sum, grad_pre_norm_sum, clip_sum, nan_sum, lam_sum, sig_sum,
             q_sum, proxy_sum, eta_sum, x0_clip_sum, shift_sum, anchor_projection_sum,
             anchor_shift_sum, applied_sum, clean_injection_sum) = carry
            t, z, rand_dir = inp
            noise_pred = model(t, x)
            model_mean, model_log_variance = self.p_mean_variance(t, x, noise_pred)
            sigma = jnp.exp(0.5 * model_log_variance)
            (clean_mean, latent_mean_delta, use_clean_f, grad_norm, grad_pre_norm, clip_frac, nan_frac,
             lam_mean, sig_mean, q_mean, proxy_mean, eta, x0_clip_frac, shift,
             anchor_projection_rate, anchor_shift, applied) = guided_step(x, t, rand_dir, noise_pred)
            stochastic_term = 0.0 if deterministic else (t > 0) * sigma * z
            guided_mean = jnp.where(use_clean_f > 0.5, clean_mean, model_mean + latent_mean_delta)
            x = guided_mean + stochastic_term
            return (
                x,
                grad_norm_sum + grad_norm,
                grad_pre_norm_sum + grad_pre_norm,
                clip_sum + clip_frac,
                nan_sum + nan_frac,
                lam_sum + lam_mean,
                sig_sum + sig_mean,
                q_sum + q_mean,
                proxy_sum + proxy_mean,
                eta_sum + eta,
                x0_clip_sum + x0_clip_frac,
                shift_sum + shift,
                anchor_projection_sum + anchor_projection_rate,
                anchor_shift_sum + anchor_shift,
                applied_sum + applied,
                clean_injection_sum + use_clean_f,
            ), None

        t_arr = jnp.arange(self.num_timesteps)[::-1]
        zero = jnp.float32(0.0)
        init = (x, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero)
        (x, grad_norm_sum, grad_pre_norm_sum, clip_sum, nan_sum, lam_sum, sig_sum,
         q_sum, proxy_sum, eta_sum, x0_clip_sum, shift_sum, anchor_projection_sum,
         anchor_shift_sum, applied_sum, clean_injection_sum), _ = jax.lax.scan(
            body_fn, init, (t_arr, noise, grad_noise)
        )
        denom = jnp.maximum(jnp.asarray(self.num_timesteps, dtype=jnp.float32), 1.0)
        applied_denom = jnp.maximum(applied_sum, 1.0)
        info = {
            "guidance/q_grad_norm": grad_norm_sum / denom,
            "guidance/q_grad_norm_preclip": grad_pre_norm_sum / denom,
            "guidance/grad_clip_frac": clip_sum / denom,
            "guidance/grad_nan_frac": nan_sum / denom,
            "guidance/lambda": lam_sum / denom,
            "guidance/sigma_q": sig_sum / denom,
            "guidance/sigma_ref": jnp.asarray(sigma_ref, dtype=jnp.float32),
            "guidance/eta_t": eta_sum / denom,
            "guidance/eta_t_applied": eta_sum / applied_denom,
            "guidance/applied_step_rate": applied_sum / denom,
            "guidance/clean_injection_rate": clean_injection_sum / denom,
            "guidance/x0_clip_frac": x0_clip_sum / denom,
            "guidance/q_mean": q_sum / denom,
            "guidance/proxy_score_mean": proxy_sum / denom,
            "guidance/objective_sign": objective_sign,
            "guidance/action_shift": shift_sum / denom,
            "guidance/action_delta_norm": shift_sum / denom,
            "guidance/anchor_project_rate": anchor_projection_sum / denom,
            "guidance/anchor_shift": anchor_shift_sum / denom,
            "guidance/max_action_shift": jnp.where(
                use_anchor_projection, max_action_shift, jnp.float32(-1.0)
            ),
            "guidance/objective_is_cost": (critic_objective > 0.5).astype(jnp.float32),
        }
        return x, info

    def q_sample(self, t: int, x_start: jax.Array, noise: jax.Array):
        B = self.beta_schedule()
        return B.sqrt_alphas_cumprod[t] * x_start + B.sqrt_one_minus_alphas_cumprod[t] * noise

    def p_loss(self, key: jax.Array, model: DiffusionModel, t: jax.Array, x_start: jax.Array):
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]

        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)
        loss = optax.squared_error(noise_pred, noise)
        return loss.mean()
    
    def weighted_p_loss(self, key: jax.Array, weights: jax.Array, model: DiffusionModel, t: jax.Array,
                        x_start: jax.Array):
        if len(weights.shape) == 1:
            weights = weights.reshape(-1, 1)
        assert t.ndim == 1 and t.shape[0] == x_start.shape[0]
        noise = jax.random.normal(key, x_start.shape)
        x_noisy = jax.vmap(self.q_sample)(t, x_start, noise)
        noise_pred = model(t, x_noisy)

        # per-sample mse
        per_elem = optax.squared_error(noise_pred, noise)        # (B, act_dim)
        per_sample = jnp.mean(per_elem, axis=-1)                 # (B,)

        w = weights.squeeze(-1)                                  # (B,)
        return jnp.sum(per_sample * w) / (jnp.sum(w) + 1e-6)
