from dataclasses import dataclass
from typing import Callable, NamedTuple, Sequence, Tuple

import jax, jax.numpy as jnp
import haiku as hk
import math

from relax.network.blocks import Activation, DistributionalQNet2, DACERPolicyNet
from relax.network.common import WithSquashedGaussianPolicy
from relax.utils.diffusion import GaussianDiffusion
from relax.utils.jax_utils import random_key_from_data

class DACERParams(NamedTuple):
    q1: hk.Params
    q2: hk.Params
    target_q1: hk.Params
    target_q2: hk.Params
    policy: hk.Params
    log_alpha: jax.Array


@dataclass
class DACERNet:
    q: Callable[[hk.Params, jax.Array, jax.Array], jax.Array]
    policy: Callable[[hk.Params, jax.Array, jax.Array, jax.Array], jax.Array]
    num_timesteps: int
    act_dim: int
    target_entropy: float
    action_noise_coef: float

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.num_timesteps)

    def get_action(self, key: jax.Array, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        key, noise_key = jax.random.split(key)
        action = self.diffusion.p_sample(key, model_fn, (*obs.shape[:-1], self.act_dim))
        action = action + jax.random.normal(noise_key, action.shape) * jnp.exp(log_alpha) * self.action_noise_coef
        return action.clip(-1, 1)

    def get_deterministic_action(self, policy_params: hk.Params, obs: jax.Array) -> jax.Array:
        policy_params, log_alpha = policy_params

        def model_fn(t, x):
            return self.policy(policy_params, obs, x, t)

        # Keep evaluation deterministic, but condition the initial latent on the
        # observation instead of using one global fixed latent for every state.
        eval_key = random_key_from_data(obs)
        action = self.diffusion.p_sample_deterministic(eval_key, model_fn, (*obs.shape[:-1], self.act_dim))
        return action.clip(-1, 1)

    def get_action_guided(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        q_params: Tuple[hk.Params, hk.Params],
        obs: jax.Array,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_q_agg: int = 3,
        guidance_kappa: float = 1.0,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: jax.Array | None = None,
        max_action_shift: float = -1.0,
    ) -> jax.Array:
        """Inference-time proxy-value-guided diffusion sampling."""
        action, _ = self.get_action_guided_with_metrics(
            key,
            policy_params,
            q_params,
            obs,
            sigma_ref,
            lambda_0=lambda_0,
            beta_unc=beta_unc,
            p_decay=p_decay,
            grad_clip=grad_clip,
            guidance_mode=guidance_mode,
            guidance_target=guidance_target,
            guidance_step_interval=guidance_step_interval,
            guidance_q_agg=guidance_q_agg,
            guidance_kappa=guidance_kappa,
            guidance_injection=guidance_injection,
            guidance_schedule=guidance_schedule,
            critic_objective=critic_objective,
            anchor_action=anchor_action,
            max_action_shift=max_action_shift,
        )
        return action

    def get_action_guided_with_metrics(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        q_params: Tuple[hk.Params, hk.Params],
        obs: jax.Array,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_q_agg: int = 3,
        guidance_kappa: float = 1.0,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: jax.Array | None = None,
        max_action_shift: float = -1.0,
    ) -> Tuple[jax.Array, dict]:
        """Inference-time proxy-value-guided sampling with diagnostics."""
        policy_params_inner, log_alpha = policy_params
        target_q1_params, target_q2_params = q_params

        def model_fn(t, x):
            return self.policy(policy_params_inner, obs, x, t)

        def q_grad_value_fn(a_hat):
            q1m, _ = self.q(target_q1_params, obs, a_hat)
            q2m, _ = self.q(target_q2_params, obs, a_hat)
            min_q = jnp.minimum(q1m, q2m)
            max_q = jnp.maximum(q1m, q2m)
            mean_q = 0.5 * (q1m + q2m)
            # TeX convention: C_LCB=min_i C_i, C_UCB=max_i C_i,
            # U_psi=C_UCB-C_LCB. guidance_kappa is kept only for legacy
            # non-TeX ablations; the main lcb/ucb choices are exact min/max.
            lcb_q = min_q
            ucb_q = max_q
            # objective-aware conservative critic target:
            #   cost/risk critic: robust ablation minimizes UCB(C)
            #   value critic: robust ablation maximizes LCB(Q), implemented as minimizing -LCB(Q)
            is_cost = jnp.asarray(critic_objective, dtype=jnp.float32) > 0.5
            conservative_q = jnp.where(is_cost, ucb_q, lcb_q)
            return jnp.where(
                guidance_q_agg == 5,
                conservative_q,
                jnp.where(
                    guidance_q_agg == 1,
                    mean_q,
                    jnp.where(
                        guidance_q_agg == 2,
                        q1m,
                        jnp.where(guidance_q_agg == 3, lcb_q, jnp.where(guidance_q_agg == 4, ucb_q, min_q)),
                    ),
                ),
            )

        def q_uncertainty_fn(a_hat):
            q1m, q1s = self.q(target_q1_params, obs, a_hat)
            q2m, q2s = self.q(target_q2_params, obs, a_hat)
            # U_psi(s,a)=C_UCB-C_LCB=|C1-C2| in the TeX trust-region/gate.
            # Distributional std is intentionally not mixed into U_psi here.
            twin_disagreement = jnp.abs(q1m - q2m)
            return twin_disagreement

        key, noise_key = jax.random.split(key)
        action, info = self.diffusion.p_sample_value_guided_with_metrics(
            key,
            model_fn,
            (*obs.shape[:-1], self.act_dim),
            q_grad_value_fn=q_grad_value_fn,
            q_uncertainty_fn=q_uncertainty_fn,
            sigma_ref=sigma_ref,
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
        action = action + jax.random.normal(noise_key, action.shape) * jnp.exp(log_alpha) * self.action_noise_coef
        return action.clip(-1, 1), info

    def get_deterministic_action_guided(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        q_params: Tuple[hk.Params, hk.Params],
        obs: jax.Array,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_q_agg: int = 3,
        guidance_kappa: float = 1.0,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: jax.Array | None = None,
        max_action_shift: float = -1.0,
    ) -> jax.Array:
        """Deterministic proxy-value-guided diffusion sampling for evaluation."""
        action, _ = self.get_deterministic_action_guided_with_metrics(
            key,
            policy_params,
            q_params,
            obs,
            sigma_ref,
            lambda_0=lambda_0,
            beta_unc=beta_unc,
            p_decay=p_decay,
            grad_clip=grad_clip,
            guidance_mode=guidance_mode,
            guidance_target=guidance_target,
            guidance_step_interval=guidance_step_interval,
            guidance_q_agg=guidance_q_agg,
            guidance_kappa=guidance_kappa,
            guidance_injection=guidance_injection,
            guidance_schedule=guidance_schedule,
            critic_objective=critic_objective,
            anchor_action=anchor_action,
            max_action_shift=max_action_shift,
        )
        return action

    def get_deterministic_action_guided_with_metrics(
        self,
        key: jax.Array,
        policy_params: hk.Params,
        q_params: Tuple[hk.Params, hk.Params],
        obs: jax.Array,
        sigma_ref: jax.Array,
        lambda_0: float = 0.5,
        beta_unc: float = 1.0,
        p_decay: float = 1.0,
        grad_clip: float = 1.0,
        guidance_mode: int = 0,
        guidance_target: int = 0,
        guidance_step_interval: int = 1,
        guidance_q_agg: int = 3,
        guidance_kappa: float = 1.0,
        guidance_injection: int = 0,
        guidance_schedule: int = 0,
        critic_objective: int = 1,
        anchor_action: jax.Array | None = None,
        max_action_shift: float = -1.0,
    ) -> Tuple[jax.Array, dict]:
        """Deterministic guided sampling with diagnostics and no action noise."""
        policy_params_inner, _ = policy_params
        target_q1_params, target_q2_params = q_params

        def model_fn(t, x):
            return self.policy(policy_params_inner, obs, x, t)

        def q_grad_value_fn(a_hat):
            q1m, _ = self.q(target_q1_params, obs, a_hat)
            q2m, _ = self.q(target_q2_params, obs, a_hat)
            min_q = jnp.minimum(q1m, q2m)
            max_q = jnp.maximum(q1m, q2m)
            mean_q = 0.5 * (q1m + q2m)
            # TeX convention: C_LCB=min_i C_i, C_UCB=max_i C_i,
            # U_psi=C_UCB-C_LCB. guidance_kappa is kept only for legacy
            # non-TeX ablations; the main lcb/ucb choices are exact min/max.
            lcb_q = min_q
            ucb_q = max_q
            # objective-aware conservative critic target:
            #   cost/risk critic: robust ablation minimizes UCB(C)
            #   value critic: robust ablation maximizes LCB(Q), implemented as minimizing -LCB(Q)
            is_cost = jnp.asarray(critic_objective, dtype=jnp.float32) > 0.5
            conservative_q = jnp.where(is_cost, ucb_q, lcb_q)
            return jnp.where(
                guidance_q_agg == 5,
                conservative_q,
                jnp.where(
                    guidance_q_agg == 1,
                    mean_q,
                    jnp.where(
                        guidance_q_agg == 2,
                        q1m,
                        jnp.where(guidance_q_agg == 3, lcb_q, jnp.where(guidance_q_agg == 4, ucb_q, min_q)),
                    ),
                ),
            )

        def q_uncertainty_fn(a_hat):
            q1m, q1s = self.q(target_q1_params, obs, a_hat)
            q2m, q2s = self.q(target_q2_params, obs, a_hat)
            # U_psi(s,a)=C_UCB-C_LCB=|C1-C2| in the TeX trust-region/gate.
            # Distributional std is intentionally not mixed into U_psi here.
            twin_disagreement = jnp.abs(q1m - q2m)
            return twin_disagreement

        action, info = self.diffusion.p_sample_value_guided_deterministic_with_metrics(
            key,
            model_fn,
            (*obs.shape[:-1], self.act_dim),
            q_grad_value_fn=q_grad_value_fn,
            q_uncertainty_fn=q_uncertainty_fn,
            sigma_ref=sigma_ref,
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
        return action.clip(-1, 1), info

    def q_evaluate(
        self, key: jax.Array, q_params: hk.Params, obs: jax.Array, act: jax.Array
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        q_mean, q_std = self.q(q_params, obs, act)
        z = jax.random.normal(key, q_mean.shape)
        z = jnp.clip(z, -3.0, 3.0)  # NOTE: Why not truncated normal?
        q_value = q_mean + q_std * z
        return q_mean, q_std, q_value

    def predict_noise(self, policy_params: hk.Params, obs: jax.Array, t: jax.Array, x_noisy: jax.Array) -> jax.Array:
        """
        Predict noise for diffusion denoising BC.
        
        Args:
            policy_params: Policy parameters (policy, log_alpha)
            obs: Observation [obs_dim] or [B, obs_dim]
            t: Timestep scalar int or [B]
            x_noisy: Noisy action [act_dim] or [B, act_dim]
            
        Returns:
            noise_pred: Predicted noise same shape as x_noisy
        """
        policy_params, log_alpha = policy_params
        # Direct call to policy denoiser (no sampling, just forward pass)
        return self.policy(policy_params, obs, x_noisy, t)

def create_dacer_net(
    key: jax.Array,
    obs_dim: int,
    act_dim: int,
    hidden_sizes: Sequence[int],
    diffusion_hidden_sizes: Sequence[int],
    activation: Activation = jax.nn.relu,
    num_timesteps: int = 20,
    action_noise_coef: float = 0.0,
) -> Tuple[DACERNet, DACERParams]:
    q = hk.without_apply_rng(hk.transform(lambda obs, act: DistributionalQNet2(hidden_sizes, activation)(obs, act)))
    policy = hk.without_apply_rng(hk.transform(lambda obs, act, t: DACERPolicyNet(diffusion_hidden_sizes, activation)(obs, act, t)))

    @jax.jit
    def init(key, obs, act):
        q1_key, q2_key, policy_key = jax.random.split(key, 3)
        q1_params = q.init(q1_key, obs, act)
        q2_params = q.init(q2_key, obs, act)
        target_q1_params = q1_params
        target_q2_params = q2_params
        policy_params = policy.init(policy_key, obs, act, 0)
        log_alpha = jnp.log(0.1)  # 初始探索强度：exp(log_alpha)=0.1 (减少过度探索)
        return DACERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha)

    sample_obs = jnp.zeros((1, obs_dim))
    sample_act = jnp.zeros((1, act_dim))
    params = init(key, sample_obs, sample_act)

    net = DACERNet(
        q=q.apply,
        policy=policy.apply,
        num_timesteps=num_timesteps,
        act_dim=act_dim,
        target_entropy=-act_dim * 0.01,
        action_noise_coef=action_noise_coef,
    )
    return net, params
