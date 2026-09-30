"""
PVP-Enhanced DACER Algorithm
Implements TD backup on a_behavior + PV loss on intervention samples + stop_td masking
"""
from typing import NamedTuple, Tuple, Optional

import jax, jax.numpy as jnp
import optax
import haiku as hk
import numpy as np
import pickle

from relax.algorithm.base import Algorithm
from relax.network.dacer import DACERNet, DACERParams
from relax.buffer.pvp_replay_buffer import PVPBatch
from relax.utils.experience import Experience
from relax.utils.typing import Metric


class DACEROptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class DACERTrainState(NamedTuple):
    params: DACERParams
    opt_state: DACEROptStates
    step: int
    mean_q1_std: float
    mean_q2_std: float
    entropy: float

class DACER(Algorithm):

    def __init__(
        self,
        agent: DACERNet,
        params: DACERParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        tau: float = 0.005,
        delay_alpha_update: int = 1000,
        delay_update: int = 1,
        reward_scale: float = 1.0,
        num_samples: int = 200,
    ):
        self.agent = agent
        self.gamma = gamma
        self.lr = lr  # 修正2：保存lr
        self.alpha_lr = alpha_lr  # 修正2：保存alpha_lr
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = delay_update
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.optim = optax.adam(lr)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = DACERTrainState(
            params=params,
            opt_state=DACEROptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            mean_q1_std=jnp.float32(-1.0),
            mean_q2_std=jnp.float32(-1.0),
            entropy=jnp.float32(0.0),
        )

        @jax.jit
        def stateless_update(
            key: jax.Array, state: DACERTrainState, data: Experience
        ) -> Tuple[DACERTrainState, Metric]:
            obs, action, reward, next_obs, done = data.obs, data.action, data.reward, data.next_obs, data.done
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step, mean_q1_std, mean_q2_std = state.step, state.mean_q1_std, state.mean_q2_std
            next_eval_key, new_eval_key, new_q1_eval_key, new_q2_eval_key, log_alpha_key = jax.random.split(key, 5)

            reward *= self.reward_scale

            # compute target q
            next_action = self.agent.get_action(next_eval_key, (policy_params, log_alpha), next_obs)
            next_q1_mean, _, next_q1_sample = self.agent.q_evaluate(new_q1_eval_key, target_q1_params, next_obs, next_action)
            next_q2_mean, _, next_q2_sample = self.agent.q_evaluate(new_q2_eval_key, target_q2_params, next_obs, next_action)
            next_q_mean = jnp.minimum(next_q1_mean, next_q2_mean)
            next_q_sample = jnp.where(next_q1_mean < next_q2_mean, next_q1_sample, next_q2_sample)
            q_target = next_q_mean
            q_target_sample = next_q_sample
            q_backup = reward + (1 - done) * self.gamma * q_target
            q_backup_sample = reward + (1 - done) * self.gamma * q_target_sample

            # update q
            def q_loss_fn(q_params: hk.Params, mean_q_std: float) -> jax.Array:
                q_mean, q_std = self.agent.q(q_params, obs, action)
                new_mean_q_std = jnp.mean(q_std)
                mean_q_std = jax.lax.stop_gradient(
                    (mean_q_std == -1.0) * new_mean_q_std +
                    (mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
                )
                q_backup_bounded = jax.lax.stop_gradient(q_mean + jnp.clip(q_backup_sample - q_mean, -3 * mean_q_std, 3 * mean_q_std))
                q_std_detach = jax.lax.stop_gradient(jnp.maximum(q_std, 0))
                epsilon = 0.1
                q_loss = -(mean_q_std ** 2 + epsilon) * jnp.mean(
                    q_mean * jax.lax.stop_gradient(q_backup - q_mean) / (q_std_detach ** 2 + epsilon) +
                    q_std * ((jax.lax.stop_gradient(q_mean) - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon)
                )
                return q_loss, (q_mean, q_std, mean_q_std)

            (q1_loss, (q1_mean, q1_std, mean_q1_std)), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, mean_q1_std)
            (q2_loss, (q2_mean, q2_std, mean_q2_std)), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, mean_q2_std)

            def cal_entropy():
                keys = jax.random.split(log_alpha_key, self.num_samples)
                actions = jax.vmap(self.agent.get_action, in_axes=(0, None, None), out_axes=1)(keys, (policy_params, jax.lax.stop_gradient(log_alpha)), obs)
                
                # 纯 JAX 高斯近似 entropy 估计（无sklearn依赖）
                def entropy_for_state(state_actions):
                    # state_actions: (num_samples, action_dim)
                    mean = jnp.mean(state_actions, axis=0)
                    centered = state_actions - mean
                    cov = jnp.matmul(centered.T, centered) / (self.num_samples - 1)
                    
                    # 高斯分布的微分熵: 0.5 * log((2πe)^d * det(cov))
                    d = state_actions.shape[1]
                    sign, logdet = jnp.linalg.slogdet(cov + 1e-6 * jnp.eye(d))  # 防止奇异
                    entropy = 0.5 * (d * (1 + jnp.log(2 * jnp.pi)) + logdet)
                    return entropy
                
                # 对 batch 中每个 state 计算 entropy
                entropies = jax.vmap(entropy_for_state)(actions)  # (batch_size,)
                entropy = jnp.mean(entropies)  # 平均 entropy
                
                return entropy

            prev_entropy = state.entropy if hasattr(state, 'entropy') else jnp.float32(0.0)

            entropy = jax.lax.cond(
                jnp.logical_and(step % self.delay_alpha_update == 0, step > 0),
                cal_entropy,
                lambda: prev_entropy
            )

            # update policy
            def policy_loss_fn(policy_params) -> jax.Array:
                new_action = self.agent.get_action(new_eval_key, (policy_params, log_alpha), obs)
                q1_mean, _ = self.agent.q(q1_params, obs, new_action)
                q2_mean, _ = self.agent.q(q2_params, obs, new_action)
                q_mean = jnp.minimum(q1_mean, q2_mean)
                policy_loss = jnp.mean(-q_mean)
                return policy_loss

            total_loss, policy_grads = jax.value_and_grad(policy_loss_fn)(policy_params)

            # update alpha
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                log_alpha_loss = -jnp.mean(log_alpha * (-entropy + self.agent.target_entropy))
                return log_alpha_loss

            # update networks
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_param_update(optim, params, grads, opt_state):
                return jax.lax.cond(
                    step % self.delay_update == 0,
                    lambda params, opt_state: param_update(optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    step % self.delay_alpha_update == 0,
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                # critic_only_update is only used in warmup/ramp phases.
                # Always soft-update targets here so target tracking does not
                # depend on the parity of the shared global step.
                return optax.incremental_update(params, target_params, tau)

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_param_update(self.optim, policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)

            state = DACERTrainState(
                params=DACERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha),
                opt_state=DACEROptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                mean_q1_std=mean_q1_std,
                mean_q2_std=mean_q2_std,
                entropy=entropy,
            )
            info = {
                "q1_loss": q1_loss,
                "q1_mean": jnp.mean(q1_mean),
                "q1_std": jnp.mean(q1_std),
                "q2_loss": q2_loss,
                "q2_mean": jnp.mean(q2_mean),
                "q2_std": jnp.mean(q2_std),
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "mean_q1_std": mean_q1_std,
                "mean_q2_std": mean_q2_std,
                "entropy": entropy,
            }
            return state, info

        self._implement_common_behavior(stateless_update, self.agent.get_action, self.agent.get_deterministic_action)

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha)


class PVPOptStates(NamedTuple):
    q1: optax.OptState
    q2: optax.OptState
    policy: optax.OptState
    log_alpha: optax.OptState


class PVPTrainState(NamedTuple):
    params: DACERParams
    opt_state: PVPOptStates
    step: int
    mean_q1_std: float
    mean_q2_std: float
    entropy: float
    lambda_pv: jax.Array
    lambda_bc: jax.Array
    B: jax.Array
    reward_free: jax.Array
    lambda_qreg: jax.Array
    policy_mode: jax.Array
    lambda_rl: jax.Array
    lambda_reg: jax.Array
    rl_gain_clip: jax.Array
    pre_takeover_bc_coef: jax.Array
    pre_takeover_pv_coef: jax.Array = jnp.float32(0.0)
    # 0 = value / higher-is-better, 1 = proxy risk-cost / lower-is-better.
    critic_objective: jax.Array = jnp.float32(1.0)
    # EnergyRank / DOVE-ER diffusion-prior reshaping.
    lambda_er: jax.Array = jnp.float32(0.0)
    er_margin: jax.Array = jnp.float32(0.05)
    er_min_action_gap: jax.Array = jnp.float32(0.03)
    # 0 = behavior/executed positive action, 1 = human positive action.
    er_positive_action: jax.Array = jnp.float32(0.0)
    # EnergyRank as an intervention-derived action-space constraint.
    # If er_use_primal_dual=0, lambda_er is the fixed/static Lagrange multiplier.
    # If er_use_primal_dual=1, eta_er is adapted by eta += rho * (V_E - epsilon_E).
    er_use_primal_dual: jax.Array = jnp.float32(0.0)
    er_budget: jax.Array = jnp.float32(0.0)
    er_dual_lr: jax.Array = jnp.float32(1e-3)
    eta_er: jax.Array = jnp.float32(0.0)
    # Optional critic-calibrated energy margin m_E(s). Disabled by default.
    er_adaptive_margin: jax.Array = jnp.float32(0.0)
    er_margin_alpha: jax.Array = jnp.float32(0.25)
    er_margin_min: jax.Array = jnp.float32(0.01)
    er_margin_max: jax.Array = jnp.float32(0.20)
    # Critic-side preference constraint, complementary to the original PVP anchor.
    lambda_pv_constraint: jax.Array = jnp.float32(0.0)
    pv_constraint_margin: jax.Array = jnp.float32(0.1)
    pv_use_primal_dual: jax.Array = jnp.float32(0.0)
    pv_constraint_budget: jax.Array = jnp.float32(0.0)
    pv_dual_lr: jax.Array = jnp.float32(1e-3)
    eta_pv: jax.Array = jnp.float32(0.0)


def compute_pv_loss(
    q_fn,
    params,
    obs,
    a_human,
    a_novice,
    interventions,
    B: float = 1.0,
    lambda_qreg: float = 0.01,
    pre_takeover_weights: Optional[jax.Array] = None,
    pre_takeover_pv_coef: float = 0.0,
    critic_objective: float = 1.0,
):
    """PVP / proxy-risk anchor loss.

    ``critic_objective`` controls the score convention:
    - 0 = value mode, higher score is better: human/behavior high, novice low.
    - 1 = cost/risk mode, lower score is better: human/behavior low, novice high.
    """
    if interventions.ndim == 2 and interventions.shape[-1] == 1:
        I = interventions.squeeze(-1)
    else:
        I = interventions
    I = I.astype(jnp.float32)

    if pre_takeover_weights is not None:
        # Optional ablation only.  The main DOVEER method uses true
        # intervention-time pairs; keep this path inactive by default.
        PT = pre_takeover_weights.squeeze(-1) if (pre_takeover_weights.ndim == 2 and pre_takeover_weights.shape[-1] == 1) else pre_takeover_weights
        PT = PT.astype(jnp.float32)
        W = I + jnp.asarray(pre_takeover_pv_coef, dtype=jnp.float32) * PT
    else:
        W = I
    W = jnp.clip(W, 0.0, 1.0)

    q_h_mean, _ = q_fn(params, obs, a_human)
    q_n_mean, _ = q_fn(params, obs, a_novice)

    q_h = q_h_mean.squeeze(-1) if (q_h_mean.ndim == 2 and q_h_mean.shape[-1] == 1) else q_h_mean
    q_n = q_n_mean.squeeze(-1) if (q_n_mean.ndim == 2 and q_n_mean.shape[-1] == 1) else q_n_mean

    hard = (I > 0.5).astype(jnp.float32)
    scale = 0.5 + 0.5 * hard
    is_cost = jnp.asarray(critic_objective, dtype=jnp.float32) > 0.5
    sign = jnp.where(is_cost, -1.0, 1.0)
    target_h = sign * B * scale
    target_n = -sign * B * scale
    pv_anchor = (q_h - target_h) ** 2 + (q_n - target_n) ** 2
    q_reg = q_h ** 2 + q_n ** 2

    denom = jnp.sum(W) + 1e-6
    pv_loss = jnp.sum((pv_anchor + lambda_qreg * q_reg) * W) / denom

    return pv_loss


def compute_pv_constraint_metrics(
    q_fn,
    params,
    obs,
    a_human,
    a_novice,
    weights,
    margin: float = 0.1,
    critic_objective: float = 1.0,
):
    """Critic-side HITL preference constraint violation.

    Cost/risk mode (critic_objective=1, lower is better):
        C(s,a^+) + m_C <= C(s,a^-), i.e. C_n - C_h >= m_C.
    Value mode (critic_objective=0, higher is better):
        Q(s,a^+) >= Q(s,a^-) + m_Q, i.e. Q_h - Q_n >= m_Q.

    Returns (mean_violation, violation_rate, objective_gap_mean), where
    objective_gap is positive when the critic respects the HITL ordering.
    """
    W = weights.squeeze(-1) if (weights.ndim == 2 and weights.shape[-1] == 1) else weights
    W = jnp.clip(W.astype(jnp.float32), 0.0, 1.0)
    q_h_mean, _ = q_fn(params, obs, a_human)
    q_n_mean, _ = q_fn(params, obs, a_novice)
    q_h = q_h_mean.squeeze(-1) if (q_h_mean.ndim == 2 and q_h_mean.shape[-1] == 1) else q_h_mean
    q_n = q_n_mean.squeeze(-1) if (q_n_mean.ndim == 2 and q_n_mean.shape[-1] == 1) else q_n_mean
    is_cost = jnp.asarray(critic_objective, dtype=jnp.float32) > 0.5
    objective_gap = jnp.where(is_cost, q_n - q_h, q_h - q_n)
    violation = jnp.maximum(0.0, jnp.asarray(margin, dtype=jnp.float32) - objective_gap)
    denom = jnp.sum(W) + 1e-6
    loss = jnp.sum(violation * W) / denom
    violation_rate = jnp.sum((violation > 0.0).astype(jnp.float32) * W) / denom
    gap_mean = jnp.sum(objective_gap * W) / denom
    return loss, violation_rate, gap_mean


def compute_real_td_weight(
    base_td_1d: jax.Array,
    interventions_1d: jax.Array,
    is_pre_takeover_1d: jax.Array,
    is_demo_1d: jax.Array,
) -> jax.Array:
    """Only Bellman-update real, non-intervention, non-demo transitions."""
    normal_mask = jnp.clip(1.0 - interventions_1d, 0.0, 1.0)
    non_pt_mask = jnp.clip(1.0 - is_pre_takeover_1d, 0.0, 1.0)
    non_demo_mask = jnp.clip(1.0 - is_demo_1d, 0.0, 1.0)
    return base_td_1d * normal_mask * non_pt_mask * non_demo_mask


def compute_pv_mask(
    interventions_1d: jax.Array,
    is_pre_takeover_1d: jax.Array,
    is_demo_1d: jax.Array,
) -> jax.Array:
    """PV only learns from real online takeover samples."""
    return (
        interventions_1d
        * jnp.clip(1.0 - is_pre_takeover_1d, 0.0, 1.0)
        * jnp.clip(1.0 - is_demo_1d, 0.0, 1.0)
    )


def sanitize_sigma_ref(raw_sigma_ref: jax.Array) -> jax.Array:
    """Use a neutral sigma_ref until critic uncertainty statistics are valid."""
    raw_sigma_ref = jnp.asarray(raw_sigma_ref, dtype=jnp.float32)
    finite_positive = jnp.logical_and(jnp.isfinite(raw_sigma_ref), raw_sigma_ref > 0.0)
    return jnp.where(finite_positive, raw_sigma_ref, jnp.float32(1.0))


def _objective_twin(q1: jax.Array, q2: jax.Array, critic_objective: jax.Array) -> jax.Array:
    """Aggregate twin critics under value or proxy-risk semantics."""
    is_cost = jnp.asarray(critic_objective, dtype=jnp.float32) > 0.5
    return jnp.where(is_cost, jnp.maximum(q1, q2), jnp.minimum(q1, q2))


def _objective_twin_sample(
    q1m: jax.Array,
    q2m: jax.Array,
    q1s: jax.Array,
    q2s: jax.Array,
    critic_objective: jax.Array,
) -> jax.Array:
    """Select the sampled critic target matching the objective-aware mean."""
    is_cost = jnp.asarray(critic_objective, dtype=jnp.float32) > 0.5
    take_q1 = jnp.where(is_cost, q1m > q2m, q1m < q2m)
    return jnp.where(take_q1, q1s, q2s)


def _td_reward_1d(reward: jax.Array, reward_free: jax.Array, critic_objective: jax.Array) -> jax.Array:
    """Build the immediate TD term for value or proxy-risk critics."""
    raw_reward_1d = reward.squeeze(-1) if reward.ndim > 1 else reward
    td_reward_1d = jnp.where(
        jnp.asarray(reward_free, dtype=jnp.float32) > 0.5,
        jnp.zeros_like(raw_reward_1d),
        raw_reward_1d,
    )
    return jnp.where(
        jnp.asarray(critic_objective, dtype=jnp.float32) > 0.5,
        -td_reward_1d,
        td_reward_1d,
    )


class PVPDACER(Algorithm):
    """
    PVP-Enhanced DACER with:
    - TD backup only on a_behavior (actually executed actions)
    - PV shaping on true intervention samples
    - stop_td masking for intervention boundaries
    - Balanced dual buffer sampling
    """
    
    def __init__(
        self,
        agent: DACERNet,
        params: DACERParams,
        *,
        gamma: float = 0.99,
        lr: float = 1e-4,
        alpha_lr: float = 3e-2,
        tau: float = 0.005,
        delay_alpha_update: int = 10000,
        delay_update: int = 2,
        actor_delay: Optional[int] = None,
        target_update_delay: Optional[int] = None,
        reward_scale: float = 1.0,
        num_samples: int = 200,
        actor_lr: Optional[float] = None,
        # PVP-specific hyperparameters
        lambda_pv: float = 1.0,   # PV loss weight
        B: float = 1.0,           # PV margin
        lambda_bc: float = 5.0,   # BC loss weight
        reward_free: bool = True, # whether to use reward-free mode
        lambda_qreg: float = 0.01,
        policy_mode: str = "hybrid_dacer",  # ["pvp_paper", "hybrid_dacer"]
        lambda_rl: float = 0.1,
        lambda_reg: float = 2.0,
        rl_gain_clip: float = 0.5,
        pre_takeover_bc_coef: float = 0.0,
        pre_takeover_pv_coef: float = 0.0,
        critic_objective: str = "cost",
        lambda_er: float = 0.0,
        er_margin: float = 0.05,
        er_min_action_gap: float = 0.03,
        er_positive_action: str = "behavior",
        # Constrained EnergyRank / actor-side primal-dual options.
        er_use_primal_dual: bool = False,
        er_budget: float = 0.0,
        er_dual_lr: float = 1e-3,
        er_eta_init: Optional[float] = None,
        er_adaptive_margin: bool = False,
        er_margin_alpha: float = 0.25,
        er_margin_min: float = 0.01,
        er_margin_max: float = 0.20,
        # Critic-side preference-constraint options.
        lambda_pv_constraint: float = 0.0,
        pv_constraint_margin: float = 0.1,
        pv_use_primal_dual: bool = False,
        pv_constraint_budget: float = 0.0,
        pv_dual_lr: float = 1e-3,
        pv_eta_init: Optional[float] = None,
    ):
        self.agent = agent
        self.gamma = gamma
        self.lr = lr  # 修正2：保存lr
        self.alpha_lr = alpha_lr  # 修正2：保存alpha_lr
        self.tau = tau
        self.delay_alpha_update = delay_alpha_update
        self.delay_update = max(1, int(delay_update))  # legacy shared cadence
        self.actor_delay = max(1, int(self.delay_update if actor_delay is None else actor_delay))
        self.target_update_delay = max(1, int(self.delay_update if target_update_delay is None else target_update_delay))
        self.reward_scale = reward_scale
        self.num_samples = num_samples
        self.actor_lr = float(lr if actor_lr is None else actor_lr)
        self.lambda_pv = lambda_pv
        self.B = B
        self.lambda_bc = lambda_bc
        self.reward_free = reward_free
        self.lambda_qreg = lambda_qreg
        self.policy_mode = policy_mode
        self.lambda_rl = lambda_rl
        self.lambda_reg = lambda_reg
        self.rl_gain_clip = rl_gain_clip
        self.pre_takeover_bc_coef = pre_takeover_bc_coef
        self.pre_takeover_pv_coef = pre_takeover_pv_coef

        critic_objective_norm = str(critic_objective).lower()
        if critic_objective_norm in ("cost", "risk", "lower", "lower_is_better"):
            critic_objective_id = 1.0
        elif critic_objective_norm in ("value", "q", "higher", "higher_is_better"):
            critic_objective_id = 0.0
        else:
            raise ValueError(f"Unknown critic_objective={critic_objective}")

        er_positive_action_norm = str(er_positive_action).lower()
        if er_positive_action_norm in ("behavior", "a_b", "executed"):
            er_positive_action_id = 0.0
        elif er_positive_action_norm in ("human", "a_h"):
            er_positive_action_id = 1.0
        else:
            raise ValueError(f"Unknown er_positive_action={er_positive_action}")

        self.critic_objective = critic_objective_norm
        self.critic_objective_id = float(critic_objective_id)
        self.lambda_er = float(lambda_er)
        self.er_margin = float(er_margin)
        self.er_min_action_gap = float(er_min_action_gap)
        self.er_positive_action = er_positive_action_norm
        self.er_positive_action_id = float(er_positive_action_id)
        self.er_use_primal_dual = bool(er_use_primal_dual)
        self.er_budget = float(er_budget)
        self.er_dual_lr = float(er_dual_lr)
        self.eta_er_init = float(lambda_er if er_eta_init is None else er_eta_init)
        self.er_adaptive_margin = bool(er_adaptive_margin)
        self.er_margin_alpha = float(er_margin_alpha)
        self.er_margin_min = float(er_margin_min)
        self.er_margin_max = float(er_margin_max)
        self.lambda_pv_constraint = float(lambda_pv_constraint)
        self.pv_constraint_margin = float(pv_constraint_margin)
        self.pv_use_primal_dual = bool(pv_use_primal_dual)
        self.pv_constraint_budget = float(pv_constraint_budget)
        self.pv_dual_lr = float(pv_dual_lr)
        self.eta_pv_init = float(lambda_pv_constraint if pv_eta_init is None else pv_eta_init)
        
        self.optim = optax.adam(lr)
        self.policy_optim = optax.adam(self.actor_lr)
        self.alpha_optim = optax.adam(alpha_lr)
        self.entropy = 0.0

        self.state = PVPTrainState(
            params=params,
            opt_state=PVPOptStates(
                q1=self.optim.init(params.q1),
                q2=self.optim.init(params.q2),
                policy=self.policy_optim.init(params.policy),
                log_alpha=self.alpha_optim.init(params.log_alpha),
            ),
            step=jnp.int32(0),
            mean_q1_std=jnp.float32(-1.0),
            mean_q2_std=jnp.float32(-1.0),
            entropy=jnp.float32(0.0),
            lambda_pv=jnp.float32(lambda_pv),
            lambda_bc=jnp.float32(lambda_bc),
            B=jnp.float32(B),
            reward_free=jnp.float32(1.0 if reward_free else 0.0),
            lambda_qreg=jnp.float32(lambda_qreg),
            policy_mode=jnp.float32(0.0 if str(policy_mode).lower() == "pvp_paper" else 1.0),
            lambda_rl=jnp.float32(lambda_rl),
            lambda_reg=jnp.float32(lambda_reg),
            rl_gain_clip=jnp.float32(rl_gain_clip),
            pre_takeover_bc_coef=jnp.float32(pre_takeover_bc_coef),
            pre_takeover_pv_coef=jnp.float32(pre_takeover_pv_coef),
            critic_objective=jnp.float32(critic_objective_id),
            lambda_er=jnp.float32(lambda_er),
            er_margin=jnp.float32(er_margin),
            er_min_action_gap=jnp.float32(er_min_action_gap),
            er_positive_action=jnp.float32(er_positive_action_id),
            er_use_primal_dual=jnp.float32(1.0 if er_use_primal_dual else 0.0),
            er_budget=jnp.float32(er_budget),
            er_dual_lr=jnp.float32(er_dual_lr),
            eta_er=jnp.float32(self.eta_er_init),
            er_adaptive_margin=jnp.float32(1.0 if er_adaptive_margin else 0.0),
            er_margin_alpha=jnp.float32(er_margin_alpha),
            er_margin_min=jnp.float32(er_margin_min),
            er_margin_max=jnp.float32(er_margin_max),
            lambda_pv_constraint=jnp.float32(lambda_pv_constraint),
            pv_constraint_margin=jnp.float32(pv_constraint_margin),
            pv_use_primal_dual=jnp.float32(1.0 if pv_use_primal_dual else 0.0),
            pv_constraint_budget=jnp.float32(pv_constraint_budget),
            pv_dual_lr=jnp.float32(pv_dual_lr),
            eta_pv=jnp.float32(self.eta_pv_init),
        )
        
        # 修复：强制设置较小的初始alpha (0.1 而不是默认的 3.0)
        # 减少过度探索，让BC能更好地收敛
        import copy
        new_params = copy.deepcopy(self.state.params)
        new_log_alpha = jnp.log(0.1)  # alpha = 0.1
        new_params = new_params._replace(log_alpha=new_log_alpha)
        self.state = self.state._replace(params=new_params)

        @jax.jit
        def pvp_stateless_update(
            key: jax.Array, state: PVPTrainState, batch: PVPBatch
        ) -> Tuple[PVPTrainState, Metric]:
            # Unpack batch
            obs, next_obs, done, reward = batch.obs, batch.next_obs, batch.done, batch.reward
            a_b, a_n, a_h = batch.actions_behavior, batch.actions_novice, batch.actions_human
            interventions, stop_td, is_pre_takeover, is_demo, pair_ok_batch = (
                batch.interventions,
                batch.stop_td,
                batch.is_pre_takeover,
                batch.is_demo,
                batch.pair_ok,
            )
            
            # Unpack state
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha = state.params
            q1_opt_state, q2_opt_state, policy_opt_state, log_alpha_opt_state = state.opt_state
            step, mean_q1_std, mean_q2_std = state.step, state.mean_q1_std, state.mean_q2_std
            # 修正1：从 state 读取动态超参数
            lambda_pv = state.lambda_pv
            lambda_bc = state.lambda_bc
            B = state.B
            reward_free = state.reward_free
            lambda_qreg = state.lambda_qreg
            pre_takeover_pv_coef = state.pre_takeover_pv_coef
            policy_mode = state.policy_mode
            lambda_rl = state.lambda_rl
            lambda_reg = state.lambda_reg
            rl_gain_clip = state.rl_gain_clip
            pre_takeover_bc_coef = state.pre_takeover_bc_coef
            critic_objective = state.critic_objective
            lambda_er = state.lambda_er
            er_margin = state.er_margin
            er_min_action_gap = state.er_min_action_gap
            er_positive_action = state.er_positive_action
            er_use_primal_dual = state.er_use_primal_dual
            er_budget = state.er_budget
            er_dual_lr = state.er_dual_lr
            eta_er = state.eta_er
            er_adaptive_margin = state.er_adaptive_margin
            er_margin_alpha = state.er_margin_alpha
            er_margin_min = state.er_margin_min
            er_margin_max = state.er_margin_max
            lambda_pv_constraint = state.lambda_pv_constraint
            pv_constraint_margin = state.pv_constraint_margin
            pv_use_primal_dual = state.pv_use_primal_dual
            pv_constraint_budget = state.pv_constraint_budget
            pv_dual_lr = state.pv_dual_lr
            eta_pv = state.eta_pv
            
            # Split keys
            (new_eval_key, new_q1_eval_key, new_q2_eval_key,
             log_alpha_key, bc_key, t_key, er_key) = jax.random.split(key, 7)

            reward *= self.reward_scale

            # === Target Q computation ===
            # Keep critic bootstrap stable with deterministic target actions.
            next_action = self.agent.get_deterministic_action((policy_params, log_alpha), next_obs)
            next_q1_mean, _, next_q1_sample = self.agent.q_evaluate(new_q1_eval_key, target_q1_params, next_obs, next_action)
            next_q2_mean, _, next_q2_sample = self.agent.q_evaluate(new_q2_eval_key, target_q2_params, next_obs, next_action)
            next_q_mean = _objective_twin(next_q1_mean, next_q2_mean, critic_objective)
            next_q_sample = _objective_twin_sample(
                next_q1_mean,
                next_q2_mean,
                next_q1_sample,
                next_q2_sample,
                critic_objective,
            )
            q_target = next_q_mean
            q_target_sample = next_q_sample

            # === TD masking with stop_td ===
            td_mask = (1.0 - stop_td).clip(0.0, 1.0)

            # === Pre-compute TD targets outside loss function ===
            td_reward_1d = _td_reward_1d(reward, reward_free, critic_objective)
            done_1d = done.squeeze(-1) if done.ndim > 1 else done
            base_td_1d = td_mask.squeeze(-1) if td_mask.ndim > 1 else td_mask
            interventions_1d = (interventions.squeeze(-1) if interventions.ndim > 1 else interventions).astype(jnp.float32)
            is_pre_takeover_1d = (is_pre_takeover.squeeze(-1) if is_pre_takeover.ndim > 1 else is_pre_takeover).astype(jnp.float32)
            is_demo_1d = (is_demo.squeeze(-1) if is_demo.ndim > 1 else is_demo).astype(jnp.float32)
            pair_ok_1d = (pair_ok_batch.squeeze(-1) if pair_ok_batch.ndim > 1 else pair_ok_batch).astype(jnp.float32)

            # Pre-takeover relabels remain BC-only in round one; TD uses only real transitions.
            td_weight_1d = compute_real_td_weight(
                base_td_1d,
                interventions_1d,
                is_pre_takeover_1d,
                is_demo_1d,
            )
            td_effective_count = jnp.sum(td_weight_1d)
            pt_td_leak_count = jnp.sum(td_weight_1d * is_pre_takeover_1d)
            demo_td_leak_count = jnp.sum(td_weight_1d * is_demo_1d)
            pair_semantic_weight_1d = jnp.clip(pair_ok_1d, 0.0, 1.0)
            pv_weight_1d = compute_pv_mask(interventions_1d, is_pre_takeover_1d, is_demo_1d) * pair_semantic_weight_1d
            pre_pv_weight_1d = (
                is_pre_takeover_1d
                * jnp.clip(1.0 - interventions_1d, 0.0, 1.0)
                * jnp.clip(1.0 - is_demo_1d, 0.0, 1.0)
                * pair_semantic_weight_1d
            )
            # Use the same HITL preference pair semantics as EnergyRank:
            # a+ is either executed behavior a_b or human correction a_h,
            # while a- remains novice action a_n.  The original PVP anchor
            # below intentionally keeps (a_h, a_n) as its stabilizing target.
            a_pref_pos = jnp.where(er_positive_action > 0.5, a_h, a_b)
            a_pref_neg = a_n

            q_target_1d = q_target.squeeze(-1) if q_target.ndim > 1 else q_target
            q_target_sample_1d = q_target_sample.squeeze(-1) if q_target_sample.ndim > 1 else q_target_sample

            backup_mean_1d = td_reward_1d + (1.0 - done_1d) * self.gamma * q_target_1d
            backup_sample_1d = td_reward_1d + (1.0 - done_1d) * self.gamma * q_target_sample_1d

            # === Q-loss function with TD masking + PV loss ===
            def q_loss_fn(q_params: hk.Params, mean_q_std: float) -> jax.Array:
                # Q-values on BEHAVIOR action (a_b)
                q_mean, q_std = self.agent.q(q_params, obs, a_b)
                new_mean_q_std = jnp.mean(q_std)
                mean_q_std = jax.lax.stop_gradient(
                    (mean_q_std == -1.0) * new_mean_q_std +
                    (mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
                )
                
                # Ensure proper dimension handling for Q-values
                q_mean_1d = q_mean.squeeze(-1) if q_mean.ndim > 1 else q_mean
                q_std_1d = q_std.squeeze(-1) if q_std.ndim > 1 else q_std
                
                q_std_detach = jax.lax.stop_gradient(jnp.maximum(q_std_1d, 0.0))
                epsilon = 0.1
                
                q_backup_bounded = jax.lax.stop_gradient(
                    q_mean_1d + jnp.clip(backup_sample_1d - q_mean_1d, -3 * mean_q_std, 3 * mean_q_std)
                )
                
                per_sample_td_loss = -(mean_q_std ** 2 + epsilon) * (
                    q_mean_1d * jax.lax.stop_gradient(backup_mean_1d - q_mean_1d) / (q_std_detach ** 2 + epsilon) +
                    q_std_1d * ((jax.lax.stop_gradient(q_mean_1d) - q_backup_bounded) ** 2 - q_std_detach ** 2) / (q_std_detach ** 3 + epsilon)
                )
                
                # Apply TD mask per sample
                per_sample_td_loss = per_sample_td_loss * td_weight_1d
                td_loss_masked = jnp.sum(per_sample_td_loss) / (jnp.sum(td_weight_1d) + 1e-6)
                
                # === PV loss (only on real online intervention samples) ===
                pv_loss = compute_pv_loss(
                    self.agent.q, q_params, obs, a_h, a_n, pv_weight_1d, B, lambda_qreg,
                    pre_takeover_weights=pre_pv_weight_1d,
                    pre_takeover_pv_coef=pre_takeover_pv_coef,
                    critic_objective=critic_objective,
                )
                
                # === Critic-side intervention preference constraint ===
                pv_constraint_loss, pv_constraint_rate, pv_constraint_gap = compute_pv_constraint_metrics(
                    self.agent.q,
                    q_params,
                    obs,
                    a_pref_pos,
                    a_pref_neg,
                    pv_weight_1d,
                    margin=pv_constraint_margin,
                    critic_objective=critic_objective,
                )
                pv_constraint_term = jnp.where(
                    pv_use_primal_dual > 0.5,
                    eta_pv * (pv_constraint_loss - pv_constraint_budget),
                    lambda_pv_constraint * pv_constraint_loss,
                )

                # === Total loss ===
                # PVP anchor remains as the stabilizing value/risk target; the
                # new hinge term is the explicit critic-side HITL constraint.
                total_loss = td_loss_masked + lambda_pv * pv_loss + pv_constraint_term
                
                return total_loss, (
                    q_mean_1d, q_std_1d, mean_q_std, td_loss_masked, pv_loss,
                    pv_constraint_loss, pv_constraint_rate, pv_constraint_gap,
                    pv_constraint_term,
                )

            # Compute Q-losses
            (total_q1_loss, (q1_mean, q1_std, mean_q1_std, q1_td_loss, q1_pv_loss,
                              q1_pv_constraint_loss, q1_pv_constraint_rate, q1_pv_constraint_gap,
                              q1_pv_constraint_term)), q1_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, mean_q1_std)
            (total_q2_loss, (q2_mean, q2_std, mean_q2_std, q2_td_loss, q2_pv_loss,
                              q2_pv_constraint_loss, q2_pv_constraint_rate, q2_pv_constraint_gap,
                              q2_pv_constraint_term)), q2_grads = jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, mean_q2_std)
            pv_constraint_loss_mean = 0.5 * (q1_pv_constraint_loss + q2_pv_constraint_loss)
            pv_constraint_rate_mean = 0.5 * (q1_pv_constraint_rate + q2_pv_constraint_rate)
            pv_constraint_gap_mean = 0.5 * (q1_pv_constraint_gap + q2_pv_constraint_gap)
            pv_constraint_term_mean = 0.5 * (q1_pv_constraint_term + q2_pv_constraint_term)

            # === PV debugging metrics (no Python if - JIT compatible) ===
            PV = pv_weight_1d
            denom = jnp.sum(PV) + 1e-6  # Avoid division by zero
            
            # Q-values for human and novice actions (unpack mean/std tuple)
            q1_h_mean, _ = self.agent.q(q1_params, obs, a_h)
            q1_n_mean, _ = self.agent.q(q1_params, obs, a_n)
            q2_h_mean, _ = self.agent.q(q2_params, obs, a_h)
            q2_n_mean, _ = self.agent.q(q2_params, obs, a_n)
            
            # Shape handling
            q1_h = q1_h_mean.squeeze(-1) if q1_h_mean.ndim > 1 else q1_h_mean
            q1_n = q1_n_mean.squeeze(-1) if q1_n_mean.ndim > 1 else q1_n_mean
            q2_h = q2_h_mean.squeeze(-1) if q2_h_mean.ndim > 1 else q2_h_mean
            q2_n = q2_n_mean.squeeze(-1) if q2_n_mean.ndim > 1 else q2_n_mean
            
            # Average only on true online intervention samples.
            pv_q1_h = jnp.sum(q1_h * PV) / denom
            pv_q1_n = jnp.sum(q1_n * PV) / denom
            pv_q2_h = jnp.sum(q2_h * PV) / denom
            pv_q2_n = jnp.sum(q2_n * PV) / denom
            critic_sign = jnp.where(critic_objective > 0.5, -1.0, 1.0)
            pv_margin = critic_sign * (pv_q1_h - pv_q1_n + pv_q2_h - pv_q2_n) / 2
            pv_gap_per_sample = ((q1_h - q1_n) + (q2_h - q2_n)) / 2.0
            pv_target_gap = 2.0 * B
            pv_objective_gap_per_sample = critic_sign * pv_gap_per_sample
            pv_constraint_violation = jnp.sum(jnp.maximum(0.0, pv_target_gap - pv_objective_gap_per_sample) * PV) / denom

            # === Entropy estimation (optimized JAX version) ===
            def cal_entropy():
                keys = jax.random.split(log_alpha_key, self.num_samples)
                actions = jax.vmap(self.agent.get_action, in_axes=(0, None, None), out_axes=1)(keys, (policy_params, jax.lax.stop_gradient(log_alpha)), obs)
                
                # 纯 JAX 高斯近似 entropy 估计（比 sklearn GMM 快很多）
                # 对每个 state 的 action samples 计算协方差和 entropy
                def entropy_for_state(state_actions):
                    # state_actions: (num_samples, action_dim)
                    mean = jnp.mean(state_actions, axis=0)
                    centered = state_actions - mean
                    cov = jnp.matmul(centered.T, centered) / (self.num_samples - 1)
                    
                    # 高斯分布的微分熵: 0.5 * log((2πe)^d * det(cov))
                    d = state_actions.shape[1]
                    sign, logdet = jnp.linalg.slogdet(cov + 1e-6 * jnp.eye(d))  # 添加小正则防止奇异
                    entropy = 0.5 * (d * (1 + jnp.log(2 * jnp.pi)) + logdet)
                    return entropy
                
                # 对 batch 中每个 state 计算 entropy
                entropies = jax.vmap(entropy_for_state)(actions)  # (batch_size,)
                entropy = jnp.mean(entropies)  # 平均 entropy
                
                return entropy

            prev_entropy = state.entropy if hasattr(state, 'entropy') else jnp.float32(0.0)

            entropy = jax.lax.cond(
                jnp.logical_and(step % self.delay_alpha_update == 0, step > 0),
                cal_entropy,
                lambda: prev_entropy
            )
            actor_update_applied = (step % self.actor_delay) == 0
            target_update_applied = (step % self.target_update_delay) == 0

            # === Policy loss ===
            def policy_loss_fn(policy_params) -> jax.Array:
                I = interventions
                if I.ndim == 1:
                    I = I.reshape(-1, 1)
                I = I.astype(jnp.float32)

                PT = is_pre_takeover
                if PT.ndim == 1:
                    PT = PT.reshape(-1, 1)
                PT = PT.astype(jnp.float32)
                demo_mask = is_demo
                if demo_mask.ndim == 1:
                    demo_mask = demo_mask.reshape(-1, 1)
                demo_mask = demo_mask.astype(jnp.float32)

                # Main-paper HITL pair mask: the EnergyRank constraint is
                # indexed by the actual intervention indicator I.  Pre-takeover
                # and demo flags are metadata/legacy sampling fields; they must
                # not silently add extra conditions to the TeX constraint.
                # pair_ok_mask is only a data-validity guard for malformed pairs.
                pair_ok_mask = pair_ok_batch
                if pair_ok_mask.ndim == 1:
                    pair_ok_mask = pair_ok_mask.reshape(-1, 1)
                pair_ok_mask = jnp.clip(pair_ok_mask.astype(jnp.float32), 0.0, 1.0)

                intervention_pair_mask = I * pair_ok_mask

                intervention_rate = jnp.mean(intervention_pair_mask)
                demo_rate = jnp.mean(demo_mask)

                expert_weights = intervention_pair_mask + demo_mask + pre_takeover_bc_coef * PT
                expert_mask = (expert_weights > 0.0).astype(jnp.float32)
                non_expert_mask = 1.0 - expert_mask

                Bsz = obs.shape[0]
                t = jax.random.randint(
                    t_key,
                    shape=(Bsz,),
                    minval=0,
                    maxval=self.agent.diffusion.num_timesteps,
                )

                def model(t_batch, x_batch):
                    return jax.vmap(
                        lambda o, ti, xi: self.agent.predict_noise((policy_params, log_alpha), o, ti, xi)
                    )(obs, t_batch, x_batch)

                # Unified BC target: executed expert behavior on intervention/pre-takeover samples.
                bc_loss = self.agent.diffusion.weighted_p_loss(
                    bc_key,
                    weights=expert_weights,
                    model=model,
                    t=t,
                    x_start=a_b,
                )

                a_pos = jnp.where(er_positive_action > 0.5, a_h, a_b)
                a_neg = a_n
                pair_gap = jnp.linalg.norm(a_pos - a_neg, axis=-1, keepdims=True)
                pair_ok = (pair_gap >= er_min_action_gap).astype(jnp.float32)
                # w_ER(s) = 1[I=1] * 1[pair_ok=1].  This is only the
                # EnergyRank sample weight, not a reward/risk metric.  Keep it
                # formula-aligned with the paper: no pre-takeover/demo exclusion.
                er_weights = intervention_pair_mask * pair_ok
                er_w = er_weights.squeeze(-1) if er_weights.ndim > 1 else er_weights
                er_denom = jnp.sum(er_w) + 1e-6

                shared_noise = jax.random.normal(er_key, a_pos.shape)
                x_pos_noisy = jax.vmap(self.agent.diffusion.q_sample)(t, a_pos, shared_noise)
                x_neg_noisy = jax.vmap(self.agent.diffusion.q_sample)(t, a_neg, shared_noise)
                pred_pos = model(t, x_pos_noisy)
                pred_neg = model(t, x_neg_noisy)
                e_pos = jnp.mean((pred_pos - shared_noise) ** 2, axis=-1)
                e_neg = jnp.mean((pred_neg - shared_noise) ** 2, axis=-1)

                # Optional critic-calibrated margin m_E(s).  This mirrors the
                # TeX definition m_0 + alpha_C * clip(C_LCB(s,a^-) -
                # C_UCB(s,a^+), 0, m_bar) in cost/risk mode.  For value-mode
                # ablations we use the symmetric safe-improvement gap
                # Q_LCB(s,a^+) - Q_UCB(s,a^-).  The gap is detached so the
                # actor-side energy-rank loss never backpropagates through the
                # critic ensemble.
                q1_pos_er, _ = self.agent.q(target_q1_params, obs, a_pos)
                q2_pos_er, _ = self.agent.q(target_q2_params, obs, a_pos)
                q1_neg_er, _ = self.agent.q(target_q1_params, obs, a_neg)
                q2_neg_er, _ = self.agent.q(target_q2_params, obs, a_neg)

                def _squeeze_last(x):
                    return x.squeeze(-1) if x.ndim > 1 else x

                q1_pos_er = _squeeze_last(q1_pos_er)
                q2_pos_er = _squeeze_last(q2_pos_er)
                q1_neg_er = _squeeze_last(q1_neg_er)
                q2_neg_er = _squeeze_last(q2_neg_er)
                q_pos_mean_er = 0.5 * (q1_pos_er + q2_pos_er)
                q_neg_mean_er = 0.5 * (q1_neg_er + q2_neg_er)
                # TeX convention for proxy-risk critics: C_LCB=min_i C_i,
                # C_UCB=max_i C_i, U_psi=C_UCB-C_LCB.  Do not use a tunable
                # mean +/- kappa half-width here; tau_U handles disagreement.
                q_pos_lcb_er = jnp.minimum(q1_pos_er, q2_pos_er)
                q_pos_ucb_er = jnp.maximum(q1_pos_er, q2_pos_er)
                q_neg_lcb_er = jnp.minimum(q1_neg_er, q2_neg_er)
                q_neg_ucb_er = jnp.maximum(q1_neg_er, q2_neg_er)
                critic_gap_er = jnp.where(
                    critic_objective > 0.5,
                    q_neg_lcb_er - q_pos_ucb_er,
                    q_pos_lcb_er - q_neg_ucb_er,
                )
                margin_increment = er_margin_alpha * jnp.clip(
                    jax.lax.stop_gradient(critic_gap_er),
                    jnp.float32(0.0),
                    er_margin_max,
                )
                adaptive_margin = jnp.clip(
                    er_margin + margin_increment,
                    er_margin_min,
                    er_margin + er_margin_max,
                )
                er_margin_vec = jnp.where(er_adaptive_margin > 0.5, adaptive_margin, er_margin)

                er_raw = er_margin_vec + e_pos - e_neg
                energy_rank_loss = jnp.sum(jnp.maximum(0.0, er_raw) * er_w) / er_denom
                energy_rank_gap = jnp.sum((e_pos - e_neg) * er_w) / er_denom
                energy_rank_e_pos = jnp.sum(e_pos * er_w) / er_denom
                energy_rank_e_neg = jnp.sum(e_neg * er_w) / er_denom
                energy_rank_effective_frac = jnp.mean((er_w > 0.0).astype(jnp.float32))
                energy_rank_violation_rate = jnp.sum((er_raw > 0.0).astype(jnp.float32) * er_w) / er_denom
                energy_rank_pair_gap = jnp.sum(pair_gap.squeeze(-1) * er_w) / er_denom
                energy_rank_gap_neg_minus_pos = -energy_rank_gap
                energy_rank_margin_mean = jnp.sum(er_margin_vec * er_w) / er_denom

                new_action = self.agent.get_action(new_eval_key, (policy_params, log_alpha), obs)
                q1_pi, _ = self.agent.q(target_q1_params, obs, new_action)
                q2_pi, _ = self.agent.q(target_q2_params, obs, new_action)
                q_pi = _objective_twin(q1_pi, q2_pi, critic_objective)
                q_pi = q_pi.squeeze(-1) if q_pi.ndim > 1 else q_pi

                q1_beh, _ = self.agent.q(target_q1_params, obs, a_b)
                q2_beh, _ = self.agent.q(target_q2_params, obs, a_b)
                q_beh = _objective_twin(q1_beh, q2_beh, critic_objective)
                q_beh = q_beh.squeeze(-1) if q_beh.ndim > 1 else q_beh

                non_expert = non_expert_mask.squeeze(-1) if non_expert_mask.ndim > 1 else non_expert_mask

                is_cost = critic_objective > 0.5
                value_improvement = q_pi - jax.lax.stop_gradient(q_beh)
                risk_reduction = jax.lax.stop_gradient(q_beh) - q_pi
                improvement = jnp.where(is_cost, risk_reduction, value_improvement)
                improvement_clip = jnp.clip(improvement, -rl_gain_clip, rl_gain_clip)
                rl_loss = -jnp.sum(improvement_clip * non_expert) / (jnp.sum(non_expert) + 1e-6)

                det_action = self.agent.get_deterministic_action((policy_params, log_alpha), obs)
                reg_per_sample = jnp.mean((det_action - a_b) ** 2, axis=-1)
                reg_loss = jnp.sum(reg_per_sample * non_expert) / (jnp.sum(non_expert) + 1e-6)
                action_diff_det = jnp.sqrt(reg_per_sample + 1e-8)
                expert_indicator = expert_mask.squeeze(-1) if expert_mask.ndim > 1 else expert_mask
                pre_takeover_rate = jnp.mean(PT)

                rl_weight = jax.lax.cond(
                    policy_mode > 0.5,
                    lambda: jnp.float32(1.0),
                    lambda: jnp.float32(0.0),
                )
                er_multiplier = jnp.where(er_use_primal_dual > 0.5, eta_er, lambda_er)
                er_lagrangian_term = jnp.where(
                    er_use_primal_dual > 0.5,
                    eta_er * (energy_rank_loss - er_budget),
                    lambda_er * energy_rank_loss,
                )
                total_policy_loss = (
                    lambda_bc * bc_loss
                    + er_lagrangian_term
                    + rl_weight * lambda_rl * rl_loss
                    + rl_weight * lambda_reg * reg_loss
                )
                return total_policy_loss, (
                    bc_loss,
                    rl_loss,
                    reg_loss,
                    intervention_rate,
                    rl_weight,
                    jnp.sum(q_pi * non_expert) / (jnp.sum(non_expert) + 1e-6),
                    jnp.sum(q_beh * non_expert) / (jnp.sum(non_expert) + 1e-6),
                    jnp.sum(improvement * non_expert) / (jnp.sum(non_expert) + 1e-6),
                    jnp.sum(value_improvement * non_expert) / (jnp.sum(non_expert) + 1e-6),
                    jnp.sum(risk_reduction * non_expert) / (jnp.sum(non_expert) + 1e-6),
                    jnp.sum(action_diff_det * non_expert) / (jnp.sum(non_expert) + 1e-6),
                    jnp.sum(action_diff_det * expert_indicator) / (jnp.sum(expert_indicator) + 1e-6),
                    jnp.mean(intervention_pair_mask),
                    pre_takeover_rate,
                    demo_rate,
                    energy_rank_loss,
                    energy_rank_gap,
                    energy_rank_e_pos,
                    energy_rank_e_neg,
                    energy_rank_effective_frac,
                    energy_rank_violation_rate,
                    energy_rank_pair_gap,
                    energy_rank_gap_neg_minus_pos,
                    energy_rank_margin_mean,
                    er_lagrangian_term,
                    er_multiplier,
                    jnp.mean(er_w),
                    jnp.mean(pair_ok),
                )

            (total_loss, (bc_loss, rl_loss, reg_loss, intervention_rate, rl_weight,
                          q_pi_mean, q_beh_mean, proxy_improvement_mean,
                          value_improvement_mean, risk_reduction_mean, action_diff_nonexpert,
                          action_diff_expert, intervention_sample_rate, pre_takeover_rate,
                          demo_rate, energy_rank_loss, energy_rank_gap, energy_rank_e_pos,
                          energy_rank_e_neg, energy_rank_effective_frac,
                          energy_rank_violation_rate, energy_rank_pair_gap,
                          energy_rank_gap_neg_minus_pos, energy_rank_margin_mean,
                          er_lagrangian_term, er_multiplier,
                          energy_rank_sample_weight_mean,
                          energy_rank_pair_ok_rate)), policy_grads = jax.value_and_grad(
                policy_loss_fn, has_aux=True
            )(policy_params)

            # === Alpha loss (unchanged) ===
            def log_alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
                log_alpha_loss = -jnp.mean(log_alpha * (-entropy + self.agent.target_entropy))
                return log_alpha_loss

            # === Parameter updates ===
            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_policy_param_update(params, grads, opt_state):
                return jax.lax.cond(
                    actor_update_applied,
                    lambda params, opt_state: param_update(self.policy_optim, params, grads, opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_alpha_param_update(optim, params, opt_state):
                return jax.lax.cond(
                    jnp.logical_and(step % self.delay_alpha_update == 0, step > 0),
                    lambda params, opt_state: param_update(optim, params, jax.grad(log_alpha_loss_fn)(params), opt_state),
                    lambda params, opt_state: (params, opt_state),
                    params, opt_state
                )

            def delay_target_update(params, target_params, tau):
                return jax.lax.cond(
                    target_update_applied,
                    lambda target_params: optax.incremental_update(params, target_params, tau),
                    lambda target_params: target_params,
                    target_params
                )

            # Update networks
            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            policy_params, policy_opt_state = delay_policy_param_update(policy_params, policy_grads, policy_opt_state)
            log_alpha, log_alpha_opt_state = delay_alpha_param_update(self.alpha_optim, log_alpha, log_alpha_opt_state)

            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)

            eta_er_new = jax.lax.cond(
                jnp.logical_and(actor_update_applied, er_use_primal_dual > 0.5),
                lambda eta: jnp.maximum(
                    0.0,
                    eta + er_dual_lr * (jax.lax.stop_gradient(energy_rank_loss) - er_budget),
                ),
                lambda eta: eta,
                eta_er,
            )
            eta_pv_new = jax.lax.cond(
                pv_use_primal_dual > 0.5,
                lambda eta: jnp.maximum(
                    0.0,
                    eta + pv_dual_lr * (jax.lax.stop_gradient(pv_constraint_loss_mean) - pv_constraint_budget),
                ),
                lambda eta: eta,
                eta_pv,
            )

            # === New state ===
            # 修正1：保留 lambda_pv/lambda_bc/B/reward_free
            state = PVPTrainState(
                params=DACERParams(q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha),
                opt_state=PVPOptStates(q1=q1_opt_state, q2=q2_opt_state, policy=policy_opt_state, log_alpha=log_alpha_opt_state),
                step=step + 1,
                mean_q1_std=mean_q1_std,
                mean_q2_std=mean_q2_std,
                entropy=entropy,
                lambda_pv=lambda_pv,
                lambda_bc=lambda_bc,
                B=B,
                reward_free=reward_free,
                lambda_qreg=lambda_qreg,
                policy_mode=policy_mode,
                lambda_rl=lambda_rl,
                lambda_reg=lambda_reg,
                rl_gain_clip=rl_gain_clip,
                pre_takeover_bc_coef=pre_takeover_bc_coef,
                pre_takeover_pv_coef=pre_takeover_pv_coef,
                critic_objective=critic_objective,
                lambda_er=lambda_er,
                er_margin=er_margin,
                er_min_action_gap=er_min_action_gap,
                er_positive_action=er_positive_action,
                er_use_primal_dual=er_use_primal_dual,
                er_budget=er_budget,
                er_dual_lr=er_dual_lr,
                eta_er=eta_er_new,
                er_adaptive_margin=er_adaptive_margin,
                er_margin_alpha=er_margin_alpha,
                er_margin_min=er_margin_min,
                er_margin_max=er_margin_max,
                lambda_pv_constraint=lambda_pv_constraint,
                pv_constraint_margin=pv_constraint_margin,
                pv_use_primal_dual=pv_use_primal_dual,
                pv_constraint_budget=pv_constraint_budget,
                pv_dual_lr=pv_dual_lr,
                eta_pv=eta_pv_new,
            )

            # === Logging (keep JAX scalars, float conversion moved to Algorithm.update) ===
            # Helper function to compute gradient norm
            def grad_norm(grads):
                leaves = jax.tree_util.tree_leaves(grads)
                return jnp.sqrt(sum(jnp.sum(g**2) for g in leaves))
            
            # Prepare td_q_backup for logging - use a simple approach to avoid dimension issues
            # Instead of using the problematic td_q_backup, recalculate it safely
            done_1d = done.squeeze(-1) if done.ndim > 1 else done
            q_target_1d = q_target.squeeze(-1) if q_target.ndim > 1 else q_target
            td_q_backup_local = td_reward_1d + (1.0 - done_1d) * self.gamma * q_target_1d
            q_objective_mean = _objective_twin(q1_mean, q2_mean, critic_objective)
            
            info = {
                "q1_loss": q1_td_loss,
                "q2_loss": q2_td_loss,
                "pv1_loss": q1_pv_loss,
                "pv2_loss": q2_pv_loss,
                "total_q1_loss": total_q1_loss,
                "total_q2_loss": total_q2_loss,
                "q1_mean": jnp.mean(q1_mean),
                "q2_mean": jnp.mean(q2_mean),
                "q1_std": jnp.mean(q1_std),
                "q2_std": jnp.mean(q2_std),
                "mean_q1_std": mean_q1_std,
                "mean_q2_std": mean_q2_std,
                "policy_loss": total_loss,
                "alpha": jnp.exp(log_alpha),
                "entropy": entropy,
                "intervention_frac": jnp.mean(interventions),
                "td_weight_mean": jnp.mean(td_weight_1d),
                # 修正1：使用 state 中的值而不是 self
                "lambda_pv": lambda_pv,
                "B": B,
                "policy/total_loss": total_loss,
                "policy/bc_loss_weighted": lambda_bc * bc_loss,
                "policy/energy_rank_loss_weighted": er_multiplier * energy_rank_loss,
                "policy/energy_constraint_lagrangian_term": er_lagrangian_term,
                "policy/rl_loss_weighted": rl_weight * lambda_rl * rl_loss,
                "policy/reg_loss_weighted": rl_weight * lambda_reg * reg_loss,
                "policy/alpha": jnp.exp(log_alpha),
                "policy/entropy": entropy,
                # === PVP分离loss指标 ===
                "policy/rl_loss": rl_loss,
                "policy/bc_loss": bc_loss,
                "policy/bc_loss_expert": bc_loss,
                "policy/rl_loss_nonexpert": rl_loss,
                "policy/reg_loss_nonexpert": reg_loss,
                "policy/intervention_rate": intervention_rate,
                "policy/pre_takeover_rate": pre_takeover_rate,
                "policy/demo_rate": demo_rate,
                "policy/lambda_bc": lambda_bc,
                "policy/lambda_rl": lambda_rl,
                "policy/lambda_reg": lambda_reg,
                "policy/rl_gain_clip": rl_gain_clip,
                "policy/pre_takeover_bc_coef": pre_takeover_bc_coef,
                "policy/pre_takeover_pv_coef": pre_takeover_pv_coef,
                "policy/actor_lr": jnp.float32(self.actor_lr),
                "policy/actor_delay": jnp.float32(self.actor_delay),
                "policy/actor_update_applied": jnp.float32(actor_update_applied),
                "policy/steps_since_actor_update": jnp.float32(step % self.actor_delay),
                "policy/mode": policy_mode,
                "policy/rl_weight": rl_weight,
                "policy/q_pi_nonexpert": q_pi_mean,
                "policy/q_beh_nonexpert": q_beh_mean,
                "policy/q_gain_nonexpert": proxy_improvement_mean,
                "policy/proxy_improvement_nonexpert": proxy_improvement_mean,
                "policy/value_improvement_nonexpert": value_improvement_mean,
                "policy/risk_reduction_nonexpert": risk_reduction_mean,
                "policy/critic_objective": critic_objective,
                "policy/action_diff_det_nonexpert": action_diff_nonexpert,
                "policy/action_diff_det_expert": action_diff_expert,
                "energy_rank/loss": energy_rank_loss,
                "energy_rank/loss_weighted": er_multiplier * energy_rank_loss,
                "energy_rank/gap_pos_minus_neg": energy_rank_gap,
                "energy_rank/e_pos": energy_rank_e_pos,
                "energy_rank/e_neg": energy_rank_e_neg,
                "energy_rank/effective_frac": energy_rank_effective_frac,
                "energy_rank/violation_rate": energy_rank_violation_rate,
                "energy_rank/pair_gap": energy_rank_pair_gap,
                "energy_rank/lambda": lambda_er,
                "energy_rank/margin": er_margin,
                "energy_rank/sample_weight_mean": energy_rank_sample_weight_mean,
                "energy_rank/pair_ok_rate": energy_rank_pair_ok_rate,
                "energy_rank/min_action_gap": er_min_action_gap,
                "energy_rank/positive_action_id": er_positive_action,
                "energy_rank/gap_neg_minus_pos": energy_rank_gap_neg_minus_pos,
                "energy_rank/margin_mean": energy_rank_margin_mean,
                "energy_rank/adaptive_margin": er_adaptive_margin,
                "energy_constraint/violation": energy_rank_loss,
                "energy_constraint/budget": er_budget,
                "energy_constraint/slack": er_budget - energy_rank_loss,
                "energy_constraint/eta": eta_er_new,
                "energy_constraint/use_primal_dual": er_use_primal_dual,
                "energy_constraint/multiplier": er_multiplier,
                "energy_constraint/lagrangian_term": er_lagrangian_term,
                "critic/lambda_qreg": lambda_qreg,
                "critic/objective": critic_objective,
                "critic/critic_loss": 0.5 * (total_q1_loss + total_q2_loss),
                "critic/q1_loss": q1_td_loss,
                "critic/q2_loss": q2_td_loss,
                "critic/pv1_loss": q1_pv_loss,
                "critic/pv2_loss": q2_pv_loss,
                "critic/pv_constraint_loss": pv_constraint_loss_mean,
                "critic/pv_constraint_rate": pv_constraint_rate_mean,
                "critic/pv_constraint_gap": pv_constraint_gap_mean,
                "critic/pv_constraint_term": pv_constraint_term_mean,
                "critic/pv_constraint_margin": pv_constraint_margin,
                "critic/pv_constraint_budget": pv_constraint_budget,
                "critic/pv_constraint_eta": eta_pv_new,
                "critic/pv_constraint_lambda": lambda_pv_constraint,
                "critic/pv_constraint_use_primal_dual": pv_use_primal_dual,
                "critic/total_q1_loss": total_q1_loss,
                "critic/total_q2_loss": total_q2_loss,
                "critic/q1_mean": jnp.mean(q1_mean),
                "critic/q2_mean": jnp.mean(q2_mean),
                "critic/q1_std": jnp.mean(q1_std),
                "critic/q2_std": jnp.mean(q2_std),
                "critic/mean_q1_std": mean_q1_std,
                "critic/mean_q2_std": mean_q2_std,
                "critic/pre_takeover_frac": jnp.mean(is_pre_takeover_1d),
                "critic/demo_frac": jnp.mean(is_demo_1d),
                "critic/intervention_frac": jnp.mean(interventions_1d),
                "critic/td_weight_mean": jnp.mean(td_weight_1d),
                "critic/td_effective_count": td_effective_count,
                "critic/td_effective_frac": td_effective_count / jnp.maximum(jnp.float32(td_weight_1d.shape[0]), 1.0),
                "critic/pt_td_leak_count": pt_td_leak_count,
                "critic/demo_td_leak_count": demo_td_leak_count,
                "critic/target_update_delay": jnp.float32(self.target_update_delay),
                "critic/target_update_applied": jnp.float32(target_update_applied),
                # PV metrics
                "pv/q1_h_mean": pv_q1_h,
                "pv/q1_n_mean": pv_q1_n,
                "pv/q2_h_mean": pv_q2_h,
                "pv/q2_n_mean": pv_q2_n,
                "pv/q_h_mean": (pv_q1_h + pv_q2_h) / 2.0,
                "pv/q_n_mean": (pv_q1_n + pv_q2_n) / 2.0,
                "pv/margin": pv_margin,
                "pv/q_gap": (pv_q1_h + pv_q2_h - pv_q1_n - pv_q2_n) / 2.0,
                "pv/objective_gap": pv_margin,
                "pv/q_gap_target": pv_target_gap,
                "pv/effective_frac": jnp.mean(PV),
                "pv/pre_takeover_effective_frac": jnp.mean(pre_pv_weight_1d * (pre_takeover_pv_coef > 0.0)),
                # === 新增调试指标 ===
                # 梯度监控
                "grad/q1_norm": grad_norm(q1_grads),
                "grad/q2_norm": grad_norm(q2_grads),
                "grad/policy_norm": grad_norm(policy_grads),
                # Q 值范围
                "q/min": jnp.minimum(jnp.min(q1_mean), jnp.min(q2_mean)),
                "q/max": jnp.maximum(jnp.max(q1_mean), jnp.max(q2_mean)),
                "q/range": jnp.max(q1_mean) - jnp.min(q1_mean),
                # TD 误差
                "td/effective_reward_mean": jnp.mean(td_reward_1d),
                "td/reward_free": reward_free,
                "td/error_mean": jnp.mean(jnp.abs(td_q_backup_local - q_objective_mean)),
                "td/error_max": jnp.max(jnp.abs(td_q_backup_local - q_objective_mean)),
                "td/stop_td_rate": jnp.mean(stop_td),
                # PV 收敛监控
                "pv/loss_ratio": (q1_pv_loss + q2_pv_loss) / (q1_td_loss + q2_td_loss + 1e-6),
                "pv/constraint_violation": pv_constraint_loss_mean,
                "pv/constraint_violation_anchor_debug": pv_constraint_violation,
                "batch/intervention_mix": jnp.mean(interventions_1d),
                "batch/pre_takeover_mix": jnp.mean(is_pre_takeover_1d),
                "batch/demo_mix": jnp.mean(is_demo_1d),
                "batch/pair_ok_mix": jnp.mean(pair_semantic_weight_1d),
            }
            
            # 动作分布统计（在字典外添加）
            action_dim = a_n.shape[-1]
            for i in range(min(action_dim, 4)):  # 最多记录前4维
                info[f"action/novice_mean_{i}"] = jnp.mean(a_n[:, i])
                info[f"action/novice_std_{i}"] = jnp.std(a_n[:, i])
                info[f"action/behavior_mean_{i}"] = jnp.mean(a_b[:, i])
                info[f"action/behavior_std_{i}"] = jnp.std(a_b[:, i])
                info[f"action/human_mean_{i}"] = jnp.mean(a_h[:, i])
                info[f"action/human_std_{i}"] = jnp.std(a_h[:, i])
            
            # 如果是2维，保留原有的steer/throttle命名（向后兼容）
            if action_dim == 2:
                info["action/novice_mean_steer"] = info["action/novice_mean_0"]
                info["action/novice_mean_throttle"] = info["action/novice_mean_1"]
                info["action/novice_std_steer"] = info["action/novice_std_0"]
                info["action/novice_std_throttle"] = info["action/novice_std_1"]
                info["action/behavior_mean_steer"] = info["action/behavior_mean_0"]
                info["action/behavior_mean_throttle"] = info["action/behavior_mean_1"]
                info["action/behavior_std_steer"] = info["action/behavior_std_0"]
                info["action/behavior_std_throttle"] = info["action/behavior_std_1"]
                info["action/human_mean_steer"] = info["action/human_mean_0"]
                info["action/human_mean_throttle"] = info["action/human_mean_1"]
                info["action/human_std_steer"] = info["action/human_std_0"]
                info["action/human_std_throttle"] = info["action/human_std_1"]
            
            # 接管相关统计
            info["takeover/active_rate"] = jnp.mean(interventions)
            info["takeover/boundary_rate"] = jnp.mean(stop_td)  # stop_td=1 at takeover start OR takeover end
            info["action/diff_human_novice"] = jnp.mean(jnp.linalg.norm(a_h - a_n, axis=1))
            info["action/diff_behavior_novice"] = jnp.mean(jnp.linalg.norm(a_b - a_n, axis=1))
            info["action/diff_human_behavior"] = jnp.mean(jnp.linalg.norm(a_h - a_b, axis=1))
            return state, info

        self._implement_common_behavior(pvp_stateless_update, self.agent.get_action, self.agent.get_deterministic_action)
        self._get_action_guided = jax.jit(self.agent.get_action_guided)
        self._get_action_guided_with_metrics = jax.jit(self.agent.get_action_guided_with_metrics)
        self._get_deterministic_action_guided = jax.jit(self.agent.get_deterministic_action_guided)
        self._get_deterministic_action_guided_with_metrics = jax.jit(self.agent.get_deterministic_action_guided_with_metrics)
        
        # ============================================================================
        # BC-only update for Stage1b offline training
        # ============================================================================
        @jax.jit
        def bc_only_stateless_update(
            key: jax.Array, state: PVPTrainState, batch: PVPBatch
        ) -> Tuple[PVPTrainState, Metric]:
            """Pure BC update: only train policy on human demo data, freeze Q networks"""
            # Unpack batch
            obs = batch.obs
            a_b = batch.actions_behavior
            lambda_bc = state.lambda_bc
            
            # Unpack state - only need policy params
            q1_params, q2_params, target_q1_params, target_q2_params, policy_params, log_alpha = state.params
            policy_opt_state = state.opt_state.policy
            step = state.step
            
            # Split keys
            bc_key, t_key = jax.random.split(key, 2)
            
            # === Pure BC loss: diffusion denoising on human actions ===
            def bc_loss_fn(policy_params: hk.Params) -> Tuple[jax.Array, jax.Array]:
                Bsz = obs.shape[0]
                # Sample timesteps
                t = jax.random.randint(t_key, shape=(Bsz,), minval=0, maxval=self.agent.diffusion.num_timesteps)
                
                # Model: predict noise given (obs, noisy_action, timestep)
                def model(t_batch, x_batch):
                    return jax.vmap(
                        lambda o, ti, xi: self.agent.predict_noise((policy_params, log_alpha), o, ti, xi)
                    )(obs, t_batch, x_batch)
                
                # Uniform weights (all samples contribute equally)
                weights = jnp.ones((Bsz, 1), dtype=jnp.float32)
                
                # Diffusion denoising loss
                bc_loss = self.agent.diffusion.weighted_p_loss(
                    bc_key,
                    weights=weights,
                    model=model,
                    t=t,
                    x_start=a_b  # Unified BC target: executed behavior action
                )
                total_loss = lambda_bc * bc_loss
                return total_loss, bc_loss
            
            # Compute BC loss and gradients
            (total_bc_loss, bc_loss), policy_grads = jax.value_and_grad(bc_loss_fn, has_aux=True)(policy_params)
            
            # Update only policy parameters
            policy_update, new_policy_opt_state = self.policy_optim.update(policy_grads, policy_opt_state)
            new_policy_params = optax.apply_updates(policy_params, policy_update)
            
            # Keep Q networks and alpha frozen
            # 修正1：保留 lambda_pv/lambda_bc/B/reward_free
            new_state = PVPTrainState(
                params=DACERParams(
                    q1=q1_params,  # frozen
                    q2=q2_params,  # frozen
                    target_q1=target_q1_params,  # frozen
                    target_q2=target_q2_params,  # frozen
                    policy=new_policy_params,  # updated
                    log_alpha=log_alpha  # frozen
                ),
                opt_state=PVPOptStates(
                    q1=state.opt_state.q1,  # frozen
                    q2=state.opt_state.q2,  # frozen
                    policy=new_policy_opt_state,  # updated
                    log_alpha=state.opt_state.log_alpha  # frozen
                ),
                step=step + 1,
                mean_q1_std=state.mean_q1_std,
                mean_q2_std=state.mean_q2_std,
                entropy=state.entropy,
                lambda_pv=state.lambda_pv,
                lambda_bc=state.lambda_bc,
                B=state.B,
                reward_free=state.reward_free,
                lambda_qreg=state.lambda_qreg,
                policy_mode=state.policy_mode,
                lambda_rl=state.lambda_rl,
                lambda_reg=state.lambda_reg,
                rl_gain_clip=state.rl_gain_clip,
                pre_takeover_bc_coef=state.pre_takeover_bc_coef,
                pre_takeover_pv_coef=state.pre_takeover_pv_coef,
                critic_objective=state.critic_objective,
                lambda_er=state.lambda_er,
                er_margin=state.er_margin,
                er_min_action_gap=state.er_min_action_gap,
                er_positive_action=state.er_positive_action,
                er_use_primal_dual=state.er_use_primal_dual,
                er_budget=state.er_budget,
                er_dual_lr=state.er_dual_lr,
                eta_er=state.eta_er,
                er_adaptive_margin=state.er_adaptive_margin,
                er_margin_alpha=state.er_margin_alpha,
                er_margin_min=state.er_margin_min,
                er_margin_max=state.er_margin_max,
                lambda_pv_constraint=state.lambda_pv_constraint,
                pv_constraint_margin=state.pv_constraint_margin,
                pv_use_primal_dual=state.pv_use_primal_dual,
                pv_constraint_budget=state.pv_constraint_budget,
                pv_dual_lr=state.pv_dual_lr,
                eta_pv=state.eta_pv,
            )
            
            # Logging
            info = {
                "policy/bc_loss": bc_loss,
                "policy/bc_loss_weighted": total_bc_loss,
                "policy_loss": total_bc_loss,
                "policy/total_loss": total_bc_loss,
                "policy/pv_loss": jnp.float32(0.0),  # No PV loss in BC-only mode
                "policy/rl_loss": jnp.float32(0.0),  # No RL loss in BC-only mode
                "energy_rank/loss": jnp.float32(0.0),
                "energy_rank/loss_weighted": jnp.float32(0.0),
                "energy_constraint/violation": jnp.float32(0.0),
                "energy_constraint/budget": state.er_budget,
                "energy_constraint/eta": state.eta_er,
                "critic/pv_constraint_loss": jnp.float32(0.0),
                "critic/pv_constraint_eta": state.eta_pv,
                "policy/lambda_bc": lambda_bc,
                "policy/actor_lr": jnp.float32(self.actor_lr),
                "policy/actor_delay": jnp.float32(self.actor_delay),
                "policy/actor_update_applied": jnp.float32(1.0),
                "policy/steps_since_actor_update": jnp.float32(0.0),
                "critic/critic_loss": jnp.float32(0.0),
            }
            
            return new_state, info
        
        self._bc_only_update = bc_only_stateless_update

        # ============================================================================
        # Critic-only update for Stage2 warmup
        # ============================================================================
        @jax.jit
        def critic_only_stateless_update(
            key: jax.Array, state: PVPTrainState, batch: PVPBatch
        ) -> Tuple[PVPTrainState, Metric]:
            """Critic-only update: update Q networks while freezing policy/log_alpha"""

            obs = batch.obs
            next_obs = batch.next_obs
            done = batch.done
            reward = batch.reward
            a_b = batch.actions_behavior
            a_n = batch.actions_novice
            a_h = batch.actions_human
            interventions = batch.interventions
            stop_td = batch.stop_td
            is_pre_takeover = batch.is_pre_takeover
            is_demo = batch.is_demo
            pair_ok_batch = batch.pair_ok

            params = state.params
            opt_state = state.opt_state
            q1_params, q2_params = params.q1, params.q2
            target_q1_params, target_q2_params = params.target_q1, params.target_q2
            policy_params, log_alpha = params.policy, params.log_alpha
            q1_opt_state, q2_opt_state = opt_state.q1, opt_state.q2
            step = state.step
            mean_q1_std = state.mean_q1_std
            mean_q2_std = state.mean_q2_std
            # 修正1：从 state 读取动态超参数
            lambda_pv = state.lambda_pv
            lambda_bc = state.lambda_bc
            B = state.B
            reward_free = state.reward_free
            lambda_qreg = state.lambda_qreg
            pre_takeover_pv_coef = state.pre_takeover_pv_coef
            policy_mode = state.policy_mode
            critic_objective = state.critic_objective
            er_positive_action = state.er_positive_action
            lambda_pv_constraint = state.lambda_pv_constraint
            pv_constraint_margin = state.pv_constraint_margin
            pv_use_primal_dual = state.pv_use_primal_dual
            pv_constraint_budget = state.pv_constraint_budget
            pv_dual_lr = state.pv_dual_lr
            eta_pv = state.eta_pv

            reward = reward * self.reward_scale

            new_q1_eval_key, new_q2_eval_key = jax.random.split(key, 2)
            next_action = self.agent.get_deterministic_action((policy_params, log_alpha), next_obs)
            next_q1_mean, _, next_q1_sample = self.agent.q_evaluate(new_q1_eval_key, target_q1_params, next_obs, next_action)
            next_q2_mean, _, next_q2_sample = self.agent.q_evaluate(new_q2_eval_key, target_q2_params, next_obs, next_action)
            q_target = _objective_twin(next_q1_mean, next_q2_mean, critic_objective)
            q_target_sample = _objective_twin_sample(
                next_q1_mean,
                next_q2_mean,
                next_q1_sample,
                next_q2_sample,
                critic_objective,
            )

            td_mask = (1.0 - stop_td).clip(0.0, 1.0)

            td_reward_1d = _td_reward_1d(reward, reward_free, critic_objective)
            done_1d = done.squeeze(-1) if done.ndim > 1 else done
            base_td_1d = td_mask.squeeze(-1) if td_mask.ndim > 1 else td_mask
            interventions_1d = (interventions.squeeze(-1) if interventions.ndim > 1 else interventions).astype(jnp.float32)
            is_pre_takeover_1d = (is_pre_takeover.squeeze(-1) if is_pre_takeover.ndim > 1 else is_pre_takeover).astype(jnp.float32)
            is_demo_1d = (is_demo.squeeze(-1) if is_demo.ndim > 1 else is_demo).astype(jnp.float32)
            pair_ok_1d = (pair_ok_batch.squeeze(-1) if pair_ok_batch.ndim > 1 else pair_ok_batch).astype(jnp.float32)

            td_weight_1d = compute_real_td_weight(
                base_td_1d,
                interventions_1d,
                is_pre_takeover_1d,
                is_demo_1d,
            )
            td_effective_count = jnp.sum(td_weight_1d)
            pt_td_leak_count = jnp.sum(td_weight_1d * is_pre_takeover_1d)
            demo_td_leak_count = jnp.sum(td_weight_1d * is_demo_1d)
            pair_semantic_weight_1d = jnp.clip(pair_ok_1d, 0.0, 1.0)
            pv_weight_1d = compute_pv_mask(interventions_1d, is_pre_takeover_1d, is_demo_1d) * pair_semantic_weight_1d
            pre_pv_weight_1d = (
                is_pre_takeover_1d
                * jnp.clip(1.0 - interventions_1d, 0.0, 1.0)
                * jnp.clip(1.0 - is_demo_1d, 0.0, 1.0)
                * pair_semantic_weight_1d
            )
            a_pref_pos = jnp.where(er_positive_action > 0.5, a_h, a_b)
            a_pref_neg = a_n

            q_target_1d = q_target.squeeze(-1) if q_target.ndim > 1 else q_target
            q_target_sample_1d = q_target_sample.squeeze(-1) if q_target_sample.ndim > 1 else q_target_sample

            backup_mean_1d = td_reward_1d + (1.0 - done_1d) * self.gamma * q_target_1d
            backup_sample_1d = td_reward_1d + (1.0 - done_1d) * self.gamma * q_target_sample_1d

            def q_loss_fn(q_params: hk.Params, mean_q_std: float):
                q_mean, q_std = self.agent.q(q_params, obs, a_b)
                new_mean_q_std = jnp.mean(q_std)
                mean_q_std = jax.lax.stop_gradient(
                    (mean_q_std == -1.0) * new_mean_q_std +
                    (mean_q_std != -1.0) * (self.tau * new_mean_q_std + (1 - self.tau) * mean_q_std)
                )

                q_mean_1d = q_mean.squeeze(-1) if q_mean.ndim > 1 else q_mean
                q_std_1d = q_std.squeeze(-1) if q_std.ndim > 1 else q_std

                q_std_detach = jax.lax.stop_gradient(jnp.maximum(q_std_1d, 0.0))
                epsilon = 0.1

                q_backup_bounded = jax.lax.stop_gradient(
                    q_mean_1d + jnp.clip(backup_sample_1d - q_mean_1d, -3 * mean_q_std, 3 * mean_q_std)
                )

                per_sample_td_loss = -(mean_q_std ** 2 + epsilon) * (
                    q_mean_1d * jax.lax.stop_gradient(backup_mean_1d - q_mean_1d) / (q_std_detach ** 2 + epsilon) +
                    q_std_1d * ((jax.lax.stop_gradient(q_mean_1d) - q_backup_bounded) ** 2 - q_std_detach ** 2) /
                    (q_std_detach ** 3 + epsilon)
                )

                per_sample_td_loss = per_sample_td_loss * td_weight_1d
                td_loss_masked = jnp.sum(per_sample_td_loss) / (jnp.sum(td_weight_1d) + 1e-6)

                pv_loss = compute_pv_loss(
                    self.agent.q, q_params, obs, a_h, a_n, pv_weight_1d, B, lambda_qreg,
                    pre_takeover_weights=pre_pv_weight_1d,
                    pre_takeover_pv_coef=pre_takeover_pv_coef,
                    critic_objective=critic_objective,
                )
                pv_constraint_loss, pv_constraint_rate, pv_constraint_gap = compute_pv_constraint_metrics(
                    self.agent.q,
                    q_params,
                    obs,
                    a_pref_pos,
                    a_pref_neg,
                    pv_weight_1d,
                    margin=pv_constraint_margin,
                    critic_objective=critic_objective,
                )
                pv_constraint_term = jnp.where(
                    pv_use_primal_dual > 0.5,
                    eta_pv * (pv_constraint_loss - pv_constraint_budget),
                    lambda_pv_constraint * pv_constraint_loss,
                )
                total_loss = td_loss_masked + lambda_pv * pv_loss + pv_constraint_term

                return total_loss, (
                    q_mean_1d, q_std_1d, mean_q_std, td_loss_masked, pv_loss,
                    pv_constraint_loss, pv_constraint_rate, pv_constraint_gap, pv_constraint_term,
                )

            (total_q1_loss, (q1_mean, q1_std, mean_q1_std, q1_td_loss, q1_pv_loss,
                              q1_pv_constraint_loss, q1_pv_constraint_rate, q1_pv_constraint_gap,
                              q1_pv_constraint_term)), q1_grads = \
                jax.value_and_grad(q_loss_fn, has_aux=True)(q1_params, mean_q1_std)
            (total_q2_loss, (q2_mean, q2_std, mean_q2_std, q2_td_loss, q2_pv_loss,
                              q2_pv_constraint_loss, q2_pv_constraint_rate, q2_pv_constraint_gap,
                              q2_pv_constraint_term)), q2_grads = \
                jax.value_and_grad(q_loss_fn, has_aux=True)(q2_params, mean_q2_std)
            pv_constraint_loss_mean = 0.5 * (q1_pv_constraint_loss + q2_pv_constraint_loss)
            pv_constraint_rate_mean = 0.5 * (q1_pv_constraint_rate + q2_pv_constraint_rate)
            pv_constraint_gap_mean = 0.5 * (q1_pv_constraint_gap + q2_pv_constraint_gap)
            pv_constraint_term_mean = 0.5 * (q1_pv_constraint_term + q2_pv_constraint_term)

            def param_update(optim, params, grads, opt_state):
                update, new_opt_state = optim.update(grads, opt_state)
                new_params = optax.apply_updates(params, update)
                return new_params, new_opt_state

            def delay_target_update(params, target_params, tau):
                return optax.incremental_update(params, target_params, tau)

            q1_params, q1_opt_state = param_update(self.optim, q1_params, q1_grads, q1_opt_state)
            q2_params, q2_opt_state = param_update(self.optim, q2_params, q2_grads, q2_opt_state)
            target_q1_params = delay_target_update(q1_params, target_q1_params, self.tau)
            target_q2_params = delay_target_update(q2_params, target_q2_params, self.tau)
            eta_pv_new = jax.lax.cond(
                pv_use_primal_dual > 0.5,
                lambda eta: jnp.maximum(
                    0.0,
                    eta + pv_dual_lr * (jax.lax.stop_gradient(pv_constraint_loss_mean) - pv_constraint_budget),
                ),
                lambda eta: eta,
                eta_pv,
            )

            # 修正1：保留 lambda_pv/lambda_bc/B/reward_free
            new_state = PVPTrainState(
                params=DACERParams(
                    q1=q1_params,
                    q2=q2_params,
                    target_q1=target_q1_params,
                    target_q2=target_q2_params,
                    policy=policy_params,
                    log_alpha=log_alpha,
                ),
                opt_state=PVPOptStates(
                    q1=q1_opt_state,
                    q2=q2_opt_state,
                    policy=opt_state.policy,
                    log_alpha=opt_state.log_alpha,
                ),
                step=step + 1,
                mean_q1_std=mean_q1_std,
                mean_q2_std=mean_q2_std,
                entropy=state.entropy,
                lambda_pv=lambda_pv,
                lambda_bc=lambda_bc,
                B=B,
                reward_free=reward_free,
                lambda_qreg=lambda_qreg,
                policy_mode=policy_mode,
                lambda_rl=state.lambda_rl,
                lambda_reg=state.lambda_reg,
                rl_gain_clip=state.rl_gain_clip,
                pre_takeover_bc_coef=state.pre_takeover_bc_coef,
                pre_takeover_pv_coef=state.pre_takeover_pv_coef,
                critic_objective=state.critic_objective,
                lambda_er=state.lambda_er,
                er_margin=state.er_margin,
                er_min_action_gap=state.er_min_action_gap,
                er_positive_action=state.er_positive_action,
                er_use_primal_dual=state.er_use_primal_dual,
                er_budget=state.er_budget,
                er_dual_lr=state.er_dual_lr,
                eta_er=state.eta_er,
                er_adaptive_margin=state.er_adaptive_margin,
                er_margin_alpha=state.er_margin_alpha,
                er_margin_min=state.er_margin_min,
                er_margin_max=state.er_margin_max,
                lambda_pv_constraint=state.lambda_pv_constraint,
                pv_constraint_margin=state.pv_constraint_margin,
                pv_use_primal_dual=state.pv_use_primal_dual,
                pv_constraint_budget=state.pv_constraint_budget,
                pv_dual_lr=state.pv_dual_lr,
                eta_pv=eta_pv_new,
            )

            info = {
                "policy/total_loss": jnp.float32(0.0),
                "critic/q1_loss": q1_td_loss,
                "critic/q2_loss": q2_td_loss,
                "critic/pv1_loss": q1_pv_loss,
                "critic/pv2_loss": q2_pv_loss,
                "critic/pv_constraint_loss": pv_constraint_loss_mean,
                "critic/pv_constraint_rate": pv_constraint_rate_mean,
                "critic/pv_constraint_gap": pv_constraint_gap_mean,
                "critic/pv_constraint_term": pv_constraint_term_mean,
                "critic/pv_constraint_margin": pv_constraint_margin,
                "critic/pv_constraint_budget": pv_constraint_budget,
                "critic/pv_constraint_eta": eta_pv_new,
                "critic/pv_constraint_lambda": lambda_pv_constraint,
                "critic/pv_constraint_use_primal_dual": pv_use_primal_dual,
                "critic/total_q1_loss": total_q1_loss,
                "critic/total_q2_loss": total_q2_loss,
                "critic/critic_loss": 0.5 * (total_q1_loss + total_q2_loss),
                "critic/objective": critic_objective,
                "critic/q1_mean": jnp.mean(q1_mean),
                "critic/q2_mean": jnp.mean(q2_mean),
                "critic/q1_std": jnp.mean(q1_std),
                "critic/q2_std": jnp.mean(q2_std),
                "critic/mean_q1_std": mean_q1_std,
                "critic/mean_q2_std": mean_q2_std,
                "critic/pre_takeover_frac": jnp.mean(is_pre_takeover_1d),
                "critic/demo_frac": jnp.mean(is_demo_1d),
                "critic/intervention_frac": jnp.mean(interventions_1d),
                "critic/td_weight_mean": jnp.mean(td_weight_1d),
                "critic/td_effective_count": td_effective_count,
                "critic/td_effective_frac": td_effective_count / jnp.maximum(jnp.float32(td_weight_1d.shape[0]), 1.0),
                "critic/pt_td_leak_count": pt_td_leak_count,
                "critic/demo_td_leak_count": demo_td_leak_count,
                "critic/target_update_delay": jnp.float32(1.0),
                "critic/target_update_applied": jnp.float32(1.0),
                "td/effective_reward_mean": jnp.mean(td_reward_1d),
                "td/reward_free": reward_free,
                "td/stop_td_rate": jnp.mean(stop_td),
                "batch/intervention_mix": jnp.mean(interventions_1d),
                "batch/pre_takeover_mix": jnp.mean(is_pre_takeover_1d),
                "batch/demo_mix": jnp.mean(is_demo_1d),
                "batch/pair_ok_mix": jnp.mean(pair_semantic_weight_1d),
                "policy/actor_lr": jnp.float32(self.actor_lr),
                "policy/actor_delay": jnp.float32(self.actor_delay),
                "policy/actor_update_applied": jnp.float32(0.0),
                "policy/steps_since_actor_update": jnp.float32(step % self.actor_delay),
                "policy/bc_loss": jnp.float32(0.0),
                "policy/rl_loss": jnp.float32(0.0),
                "policy/pv_loss": jnp.float32(0.0),
            }

            return new_state, info

        self._critic_only_update = critic_only_stateless_update

    def bc_only_update(self, key: jax.Array, data: PVPBatch) -> Metric:
        """BC-only update for Stage1b offline training
        
        Only updates policy network with diffusion BC loss on human demonstrations.
        Q networks and alpha remain frozen.
        """
        self.state, info = self._bc_only_update(key, self.state, data)
        return {k: float(v) for k, v in info.items()}

    def update_critic_only(self, key: jax.Array, data: PVPBatch) -> Metric:
        """Critic-only update used during Stage2 warmup."""
        self.state, info = self._critic_only_update(key, self.state, data)
        return {k: float(v) for k, v in info.items()}

    def update(self, key: jax.Array, data: PVPBatch) -> Metric:
        """Use PVP-specific stateless_update directly on PVPBatch."""
        self.state, info = self._update(key, self.state, data)
        return {k: float(v) for k, v in info.items()}

    def get_policy_params(self):
        return (self.state.params.policy, self.state.params.log_alpha)

    def get_inference_kit(self):
        """Pack the parameters needed by inference-time value guidance."""
        sigma_ref = 0.5 * (self.state.mean_q1_std + self.state.mean_q2_std)
        return {
            "policy_params": (self.state.params.policy, self.state.params.log_alpha),
            "q_params_target": (self.state.params.target_q1, self.state.params.target_q2),
            "sigma_ref": sanitize_sigma_ref(sigma_ref),
        }

    def get_action_guided(
        self,
        key: jax.Array,
        obs,
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
        critic_objective: Optional[float] = None,
    ) -> np.ndarray:
        """Algorithm-level entry for proxy-value-guided sampling."""
        if not hasattr(self, "_get_action_guided"):
            self._get_action_guided = jax.jit(self.agent.get_action_guided)
        kit = self.get_inference_kit()
        critic_objective_id = self.critic_objective_id if critic_objective is None else float(critic_objective)
        action = self._get_action_guided(
            key,
            kit["policy_params"],
            kit["q_params_target"],
            obs,
            kit["sigma_ref"],
            jnp.float32(lambda_0),
            jnp.float32(beta_unc),
            jnp.float32(p_decay),
            jnp.float32(grad_clip),
            jnp.int32(guidance_mode),
            jnp.int32(guidance_target),
            jnp.int32(guidance_step_interval),
            jnp.int32(guidance_q_agg),
            jnp.float32(guidance_kappa),
            jnp.int32(guidance_injection),
            jnp.int32(guidance_schedule),
            jnp.float32(critic_objective_id),
        )
        return np.asarray(action)

    def get_action_guided_with_metrics(
        self,
        key: jax.Array,
        obs,
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
        critic_objective: Optional[float] = None,
    ) -> Tuple[np.ndarray, Metric]:
        """Guided action plus host-side diagnostics for logging."""
        if not hasattr(self, "_get_action_guided_with_metrics"):
            self._get_action_guided_with_metrics = jax.jit(self.agent.get_action_guided_with_metrics)
        kit = self.get_inference_kit()
        critic_objective_id = self.critic_objective_id if critic_objective is None else float(critic_objective)
        action, info = self._get_action_guided_with_metrics(
            key,
            kit["policy_params"],
            kit["q_params_target"],
            obs,
            kit["sigma_ref"],
            jnp.float32(lambda_0),
            jnp.float32(beta_unc),
            jnp.float32(p_decay),
            jnp.float32(grad_clip),
            jnp.int32(guidance_mode),
            jnp.int32(guidance_target),
            jnp.int32(guidance_step_interval),
            jnp.int32(guidance_q_agg),
            jnp.float32(guidance_kappa),
            jnp.int32(guidance_injection),
            jnp.int32(guidance_schedule),
            jnp.float32(critic_objective_id),
        )
        return np.asarray(action), {k: float(v) for k, v in info.items()}

    def get_deterministic_action_guided(
        self,
        key: jax.Array,
        obs,
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
        critic_objective: Optional[float] = None,
    ) -> np.ndarray:
        """Deterministic guided action for offline evaluation."""
        if not hasattr(self, "_get_deterministic_action_guided"):
            self._get_deterministic_action_guided = jax.jit(self.agent.get_deterministic_action_guided)
        kit = self.get_inference_kit()
        critic_objective_id = self.critic_objective_id if critic_objective is None else float(critic_objective)
        action = self._get_deterministic_action_guided(
            key,
            kit["policy_params"],
            kit["q_params_target"],
            obs,
            kit["sigma_ref"],
            jnp.float32(lambda_0),
            jnp.float32(beta_unc),
            jnp.float32(p_decay),
            jnp.float32(grad_clip),
            jnp.int32(guidance_mode),
            jnp.int32(guidance_target),
            jnp.int32(guidance_step_interval),
            jnp.int32(guidance_q_agg),
            jnp.float32(guidance_kappa),
            jnp.int32(guidance_injection),
            jnp.int32(guidance_schedule),
            jnp.float32(critic_objective_id),
        )
        return np.asarray(action)

    def get_deterministic_action_guided_with_metrics(
        self,
        key: jax.Array,
        obs,
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
        critic_objective: Optional[float] = None,
    ) -> Tuple[np.ndarray, Metric]:
        """Deterministic guided action plus diagnostics for offline evaluation."""
        if not hasattr(self, "_get_deterministic_action_guided_with_metrics"):
            self._get_deterministic_action_guided_with_metrics = jax.jit(
                self.agent.get_deterministic_action_guided_with_metrics
            )
        kit = self.get_inference_kit()
        critic_objective_id = self.critic_objective_id if critic_objective is None else float(critic_objective)
        action, info = self._get_deterministic_action_guided_with_metrics(
            key,
            kit["policy_params"],
            kit["q_params_target"],
            obs,
            kit["sigma_ref"],
            jnp.float32(lambda_0),
            jnp.float32(beta_unc),
            jnp.float32(p_decay),
            jnp.float32(grad_clip),
            jnp.int32(guidance_mode),
            jnp.int32(guidance_target),
            jnp.int32(guidance_step_interval),
            jnp.int32(guidance_q_agg),
            jnp.float32(guidance_kappa),
            jnp.int32(guidance_injection),
            jnp.int32(guidance_schedule),
            jnp.float32(critic_objective_id),
        )
        return np.asarray(action), {k: float(v) for k, v in info.items()}
    
    def get_policy_params_to_save(self):
        """Return policy and Q parameters for saving"""
        return (self.state.params.policy, self.state.params.log_alpha, self.state.params.q1, self.state.params.q2)
    
    def save_policy(self, path: str) -> None:
        """Save policy, critics, target critics, and UPV-GDS inference state."""
        sigma_ref = sanitize_sigma_ref(0.5 * (self.state.mean_q1_std + self.state.mean_q2_std))
        payload = {
            "version": "dove_er_v2_cost_objective",
            "policy": self.state.params.policy,
            "log_alpha": self.state.params.log_alpha,
            "q1": self.state.params.q1,
            "q2": self.state.params.q2,
            "target_q1": self.state.params.target_q1,
            "target_q2": self.state.params.target_q2,
            "mean_q1_std": self.state.mean_q1_std,
            "mean_q2_std": self.state.mean_q2_std,
            "sigma_ref": sigma_ref,
            "critic_objective": self.critic_objective,
            "critic_objective_id": self.state.critic_objective,
            "lambda_er": self.state.lambda_er,
            "er_margin": self.state.er_margin,
            "er_min_action_gap": self.state.er_min_action_gap,
            "er_positive_action_id": self.state.er_positive_action,
        }
        policy = jax.device_get(payload)
        with open(path, "wb") as f:
            pickle.dump(policy, f)
    
    def update_hyperparameters(self, lambda_pv: float = None, lambda_bc: float = None, B: float = None):
        """动态更新PVP超参数，并同步到state，确保JIT更新能看到新值。"""
        if lambda_pv is not None:
            self.lambda_pv = float(lambda_pv)
            self.state = self.state._replace(lambda_pv=jnp.float32(lambda_pv))
        if lambda_bc is not None:
            self.lambda_bc = float(lambda_bc)
            self.state = self.state._replace(lambda_bc=jnp.float32(lambda_bc))
        if B is not None:
            self.B = float(B)
            self.state = self.state._replace(B=jnp.float32(B))

    def set_lambda_bc(self, value: float):
        """修正1：更新state中的lambda_bc，确保JIT函数能看到新值"""
        self.lambda_bc = float(value)
        self.state = self.state._replace(lambda_bc=jnp.float32(value))

    def set_lambda_pv(self, value: float):
        """修正1：更新state中的lambda_pv，确保JIT函数能看到新值"""
        self.lambda_pv = float(value)
        self.state = self.state._replace(lambda_pv=jnp.float32(value))

    def set_B(self, value: float):
        """修正1：更新state中的B，确保JIT函数能看到新值"""
        self.B = float(value)
        self.state = self.state._replace(B=jnp.float32(value))
    
    def set_reward_free(self, value: bool):
        """修正1：更新state中的reward_free，确保JIT函数能看到新值"""
        self.reward_free = value
        self.state = self.state._replace(reward_free=jnp.float32(1.0 if value else 0.0))

    def set_lambda_qreg(self, value: float):
        self.lambda_qreg = float(value)
        self.state = self.state._replace(lambda_qreg=jnp.float32(value))

    def set_policy_mode(self, value: str):
        self.policy_mode = value
        self.state = self.state._replace(policy_mode=jnp.float32(0.0 if str(value).lower() == "pvp_paper" else 1.0))

    def set_lambda_rl(self, value: float):
        self.lambda_rl = float(value)
        self.state = self.state._replace(lambda_rl=jnp.float32(value))

    def set_lambda_reg(self, value: float):
        self.lambda_reg = float(value)
        self.state = self.state._replace(lambda_reg=jnp.float32(value))

    def set_rl_gain_clip(self, value: float):
        self.rl_gain_clip = float(value)
        self.state = self.state._replace(rl_gain_clip=jnp.float32(value))

    def set_pre_takeover_bc_coef(self, value: float):
        self.pre_takeover_bc_coef = float(value)
        self.state = self.state._replace(pre_takeover_bc_coef=jnp.float32(value))

    def set_pre_takeover_pv_coef(self, value: float):
        self.pre_takeover_pv_coef = float(value)
        self.state = self.state._replace(pre_takeover_pv_coef=jnp.float32(value))

    def set_lambda_er(self, value: float):
        self.lambda_er = float(value)
        self.state = self.state._replace(lambda_er=jnp.float32(value))

    def get_current_hyperparameters(self):
        """获取当前超参数，用于调试"""
        return {
            "lambda_pv": self.lambda_pv,
            "lambda_bc": self.lambda_bc,
            "B": self.B,
            "reward_free": self.reward_free,
            "lambda_qreg": self.lambda_qreg,
            "policy_mode": self.policy_mode,
            "lambda_rl": self.lambda_rl,
            "lambda_reg": self.lambda_reg,
            "rl_gain_clip": self.rl_gain_clip,
            "pre_takeover_bc_coef": self.pre_takeover_bc_coef,
            "pre_takeover_pv_coef": self.pre_takeover_pv_coef,
            "critic_objective": self.critic_objective,
            "critic_objective_id": self.critic_objective_id,
            "lambda_er": self.lambda_er,
            "er_margin": self.er_margin,
            "er_min_action_gap": self.er_min_action_gap,
            "er_positive_action": self.er_positive_action,
            "er_positive_action_id": self.er_positive_action_id,
            "actor_lr": self.actor_lr,
            "actor_delay": self.actor_delay,
            "target_update_delay": self.target_update_delay,
        }
