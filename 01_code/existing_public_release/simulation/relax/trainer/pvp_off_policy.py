"""
PVP-Enhanced Off-Policy Trainer
Handles three-action collection + intervention/stop_td computation + balanced buffer usage

Canonical PVP semantics (问题30)
-----------------------
a_novice / agent_action:
    raw action proposed by current policy (before takeover system)
    
 a_human / human_action / raw_action:
    human control input at current step
    
a_behavior / behavior_action / action:
    action actually executed in the environment

Buffer semantics (问题6)
----------------
human_buffer (aka expert buffer):
    stores all expert-supervised samples, including:
      1) human takeover transitions (online Stage2)
      2) Stage1a demo transitions
      3) pre-takeover relabeled expert samples from SharedControlMonitor
      
novice_buffer:
    stores pure agent-rollout transitions where intervention == 0.
    
Critical invariants:
- In pure demo mode (sample_human_demo): a_novice=0, a_behavior=a_human, intervention=0.0, is_demo=1.0
- In PVP mode (sample): a_behavior comes from env info, may differ from a_novice during takeover
- stop_td masks TD learning at takeover_start AND takeover_end boundaries
- is_pre_takeover marks expert-relabeled transitions before takeover (window=20)
"""
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import optax
import gymnasium as gym
from tqdm import tqdm
from tensorboardX import SummaryWriter
from tensorboardX.summary import hparams

from relax.algorithm import Algorithm
from relax.buffer.pvp_replay_buffer import PVPBalancedDualBuffer
from relax.trainer.accumulator import SampleLog, UpdateLog, Interval
from relax.utils.experience import Experience


def _safe_local_attr(obj, name: str, default=None):
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        return default


def _safe_env_config(env):
    candidates = [env]
    wrapped = _safe_local_attr(env, "env", None)
    if wrapped is not None:
        candidates.append(wrapped)
    unwrapped = _safe_local_attr(env, "unwrapped", None)
    if unwrapped is not None:
        candidates.append(unwrapped)

    seen = set()
    for obj in candidates:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))
        cfg = _safe_local_attr(obj, "config", None)
        if isinstance(cfg, dict):
            return cfg
    return None


class PVPDataCollector:
    """
    Handles PVP data collection with three-action semantics.
    
    问题30：统一的动作语义和Experience创建入口。
    所有Experience创建应通过 Experience.create_pvp_experience 工厂方法。
    """
    
    @staticmethod
    def compute_intervention_flags(
        current_intervention: float, 
        prev_intervention: float
    ) -> Tuple[float, float]:
        """
        Compute intervention and stop_td flags
        
        Args:
            current_intervention: 0/1 - whether human intervenes this step
            prev_intervention: 0/1 - whether human intervened previous step
            
        Returns:
            intervention: 0/1 - intervention flag for storage
            stop_td: 0/1 - TD masking flag (1=mask TD, 0=allow TD)
                        Note: 当前实现mask接管开始和结束边界
        """
        takeover_start = (current_intervention == 1 and prev_intervention == 0)
        takeover_end = (current_intervention == 0 and prev_intervention == 1)
        
        # 当前实现：mask TD at both takeover_start AND takeover_end boundaries
        # 这与实际训练代码中的实现保持一致
        stop_td = 1.0 if (takeover_start or takeover_end) else 0.0
        
        # Alternative options (uncomment if needed):
        # stop_td = 1.0 if takeover_start else 0.0              # mask only takeover_start (original)
        # stop_td = 1.0 if current_intervention == 1 else 0.0  # mask all intervention steps
        
        return current_intervention, stop_td

    @staticmethod
    def create_pvp_experience(
        obs: np.ndarray,
        a_novice: np.ndarray,  # agent's intended action
        a_human: np.ndarray,   # human action (during intervention)
        a_behavior: np.ndarray, # actually executed action
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        intervention: float,
        stop_td: float,
        is_pre_takeover: float = 0.0,  # 是否是pre-takeover数据
        is_demo: float = 0.0,  # 是否是纯人工 Stage1a demo
        pair_ok: float | None = None,
    ) -> Experience:
        """Create PVP Experience with all required fields"""
        # 推荐统一使用 Experience.create_pvp_experience 来避免重复实现
        return Experience.create_pvp_experience(
            obs=obs,
            a_novice=a_novice,
            a_human=a_human,
            a_behavior=a_behavior,
            reward=reward,
            next_obs=next_obs,
            done=done,
            intervention=intervention,
            stop_td=stop_td,
            is_pre_takeover=is_pre_takeover,
            is_demo=is_demo,
            pair_ok=pair_ok,
        )


class PVPOffPolicyTrainer:
    """
    PVP-Enhanced Off-Policy Trainer
    Integrates three-action collection, intervention detection, and balanced buffer training
    """
    
    def __init__(
        self,
        env: gym.Env,
        algorithm: Algorithm,
        buffer: PVPBalancedDualBuffer,
        log_path: Path,
        batch_size: int = 256,
        start_step: int = 1000,
        total_step: int = int(1e6),
        sample_per_iteration: int = 1,
        update_per_iteration: int = 1,
        evaluate_env: Optional[gym.Env] = None,
        evaluate_every: int = 10000,
        evaluate_n_episode: int = 10,
        sample_log_n_episode: int = 10,
        update_log_n_step: int = 1000,
        done_info_keys: Tuple[str, ...] = (),
        save_policy_every: int = 10000,
        hparams: Optional[dict] = None,
        policy_pkl_template: str = "policy-{sample_step}-{update_step}.pkl",
        warmup_with: str = "random",
        # PVP-specific settings
        intervention_callback: Optional[Callable] = None,  # Function to detect human intervention
        reward_free: bool = True,  # Whether to use reward-free mode (store reward=0)
    ):
        self.env = env
        self.algorithm = algorithm
        self.buffer = buffer
        self.batch_size = batch_size
        self.start_step = start_step
        self.total_step = total_step
        self.sample_per_iteration = sample_per_iteration
        self.update_per_iteration = update_per_iteration
        self.log_path = log_path
        self.policy_pkl_template = policy_pkl_template
        self.evaluate_env = evaluate_env
        self.evaluate_every = evaluate_every
        self.evaluate_n_episode = evaluate_n_episode
        self.sample_log_n_episode = sample_log_n_episode
        self.update_log_n_step = update_log_n_step
        self.done_info_keys = done_info_keys
        self.save_policy_every = save_policy_every
        self.hparams = hparams
        self.warmup_with = warmup_with
        self.intervention_callback = intervention_callback
        self.reward_free = reward_free

        # UPV-GDS inference guidance. It stays disabled until Stage2 explicitly
        # configures it and arms it after critic warmup.
        self.use_value_guidance = False
        self._guidance_armed = False
        self._guidance_arm_step = 0
        self.guidance_lambda0 = 0.5
        self.guidance_beta_unc = 1.0
        self.guidance_p_decay = 1.0
        self.guidance_grad_clip = 1.0
        self.guidance_mode = 0
        
        # PVP state tracking
        self.prev_intervention = 0.0
        self.data_collector = PVPDataCollector()
        
        # Initialize trainer state for single environment
        from relax.trainer.accumulator import SampleLog, UpdateLog, Interval
        
        self.sample_log = SampleLog()
        self.update_log = UpdateLog()
        self.last_metrics = {}
        self.sample_log_interval = Interval(self.sample_log_n_episode)
        self.save_policy_interval = Interval(self.save_policy_every)

        # 初始化 episode 统计（只初始化一次）
        self._episode_stats = {
            'length': 0,
            'return': 0.0,
            'stored_return': 0.0,  # NEW: track stored reward for debugging
            'cost': 0.0,
            'native_cost': 0.0,
            'takeover_cost': 0.0,
            'interventions': 0,
            'crash': False,
            'out_of_road': False,
            'success': False
        }

    def _safe_done(self, terminated, truncated):
        """Safely combine terminated and truncated for scalar/array inputs"""
        if isinstance(terminated, np.ndarray) or isinstance(truncated, np.ndarray):
            # Use logical_or for arrays
            if isinstance(terminated, np.ndarray):
                term_array = terminated
            else:
                term_array = np.array([terminated])
                
            if isinstance(truncated, np.ndarray):
                trunc_array = truncated
            else:
                trunc_array = np.array([truncated])
                
            return np.logical_or(term_array, trunc_array)
        else:
            # Use boolean OR for scalars
            return bool(terminated) or bool(truncated)

    def _extract_pvp_actions(self, info: dict, a_novice: np.ndarray):
        """
        Canonical three-action extraction.
        Priority:
          a_b = behavior_action > action > a_novice
          a_h = human_action > raw_action > a_novice
          intervention = takeover flag
        """
        a_behavior = info.get("behavior_action", info.get("action", a_novice))
        a_human = info.get("human_action", info.get("raw_action", a_novice))
        current_intervention = float(info.get("takeover", False))

        a_behavior = np.asarray(a_behavior, dtype=np.float32)
        a_human = np.asarray(a_human, dtype=np.float32)
        a_behavior = np.clip(a_behavior, -1.0, 1.0)
        a_human = np.clip(a_human, -1.0, 1.0)
        return a_behavior, a_human, current_intervention

    def _resolve_takeover_boundaries(self, info: Optional[dict], current_intervention: float) -> Tuple[bool, bool]:
        """Use env-provided takeover boundary flags first and keep trainer-side fallback only for compatibility."""
        info = info if isinstance(info, dict) else {}
        takeover_start = bool(info.get("takeover_start", False))
        takeover_end = bool(info.get("takeover_end", False))

        if not takeover_start:
            takeover_start = (self.prev_intervention == 0.0 and current_intervention == 1.0)
        if not takeover_end:
            takeover_end = (self.prev_intervention == 1.0 and current_intervention == 0.0)

        return takeover_start, takeover_end

    def _update_episode_stats(self, *, raw_reward: float, store_reward: float, intervention: float, info: Optional[dict] = None):
        """Shared helper to keep per-episode accounting consistent across sampling modes."""
        stats = self._episode_stats
        stats['length'] += 1
        stats['return'] += float(raw_reward)
        stats['stored_return'] += float(store_reward)
        if isinstance(info, dict):
            stats['cost'] += float(info.get("cost", info.get("native_cost", 0.0)) or 0.0)
            stats['native_cost'] += float(info.get("native_cost", 0.0) or 0.0)
            stats['takeover_cost'] += float(info.get("takeover_cost", 0.0) or 0.0)
        if intervention >= 0.5:
            stats['interventions'] += 1

        if isinstance(info, dict):
            if info.get('crash', False):
                stats['crash'] = True
            if info.get('out_of_road', False):
                stats['out_of_road'] = True
            if self._info_indicates_success(info):
                stats['success'] = True

    @staticmethod
    def _info_indicates_success(info: Optional[dict]) -> bool:
        if not isinstance(info, dict):
            return False
        return any(bool(info.get(key, False)) for key in ("success", "arrive_dest"))

    @staticmethod
    def _episode_outcome_from_stats(stats: dict) -> str:
        if stats.get('crash', False):
            return "crash"
        if stats.get('out_of_road', False):
            return "out_of_road"
        if stats.get('success', False):
            return "success"
        return "timeout"

    @staticmethod
    def _outcome_code_from_label(outcome: str) -> int:
        if outcome == "crash":
            return 1
        if outcome == "out_of_road":
            return 2
        if outcome == "success":
            return 3
        return 0

    def setup(self, dummy_data: Experience):
        """Setup trainer with PVP buffer"""
        self.algorithm.warmup(dummy_data)

        # Setup logger
        # Do not overwrite externally injected logger (e.g. TBLoggerAdapter).
        if not hasattr(self, "logger") or self.logger is None:
            self.logger = SummaryWriter(str(self.log_path))
        self.progress = tqdm(total=self.total_step, desc="Sample Step", disable=None, dynamic_ncols=True)
        
        # 初始化指标存储
        self.last_metrics = {}

        self.algorithm.save_policy_structure(self.log_path, dummy_data.obs[0])
        
        # This trimmed workspace uses the explicit MetaDrive evaluation script
        # instead of the generic DACER evaluator process from the source repo.
        self.evaluator = None

    def detect_intervention(self, obs: np.ndarray, a_novice: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect human intervention
        Override this method based on your intervention detection logic
        
        Returns:
            a_human: human action (equals a_novice if no intervention)
            intervention: 0/1 flag
        """
        if self.intervention_callback is not None:
            return self.intervention_callback(obs, a_novice)
        else:
            # Default: no intervention
            return a_novice.copy(), 0.0

    def warmup(self, n_steps: int):
        """Warmup phase to populate buffer with initial data"""
        print(f"\n=== Starting Warmup ===")
        print(f"Warmup steps: {n_steps}")
        
        obs, _ = self.env.reset()
        self.prev_intervention = 0.0
        
        for i in range(n_steps):
            self.key, sample_key = jax.random.split(self.key)
            
            # Sample action based on warmup strategy
            if self.warmup_with == "random":
                a_novice = self.env.action_space.sample()
                a_novice = np.asarray(a_novice, dtype=np.float32)
            else:
                # 使用随机采样进行warmup
                a_novice = self.algorithm.get_action(sample_key, obs)
                a_novice = np.asarray(a_novice, dtype=np.float32)
            
            a_novice = np.clip(a_novice, -1.0, 1.0)
            
            # Step environment with 原始动作 - 让BC学习真正的纠错
            next_obs, reward, terminated, truncated, info = self.env.step(a_novice)
            
            # Extract canonical PVP actions from env info
            a_behavior, a_human, current_intervention = self._extract_pvp_actions(info, a_novice)
            
            # Compute stop_td flag based on takeover boundaries (start AND end)
            takeover_start, takeover_end = self._resolve_takeover_boundaries(info, current_intervention)
            
            # TD mask: block learning at both takeover start AND end boundaries
            stop_td = 1.0 if (takeover_start or takeover_end) else 0.0
            if getattr(self, "disable_stop_td_mask", False):
                stop_td = 0.0
            
            # Apply reward-free setting
            if self.reward_free:
                reward = 0.0
            
            # Create PVP experience
            experience = self.data_collector.create_pvp_experience(
                obs=obs,
                a_novice=a_novice,
                a_human=a_human,
                a_behavior=a_behavior,
                reward=reward,
                next_obs=next_obs,
                done=self._safe_done(terminated, truncated),
                intervention=current_intervention,
                stop_td=stop_td,
                is_pre_takeover=0.0,  # warmup数据不是pre-takeover
            )
            
            # Store in buffer
            self.buffer.add(experience)

            # Update state
            self.prev_intervention = current_intervention
            
            if self._safe_done(terminated, truncated):
                obs, _ = self.env.reset()
                self.prev_intervention = 0.0  # Reset intervention state
            else:
                obs = next_obs
                
        print(f"Warmup completed. Buffer size: {len(self.buffer)}")
        return obs

    def sample_human_demo(self, obs: np.ndarray):
        """Pure human demonstration sampling - AI algorithm does not generate actions"""
        import time
        start_time = time.time()
        
        sl = self.sample_log
        cached_outcome = "timeout"

        # In pure human demo mode, the placeholder novice action never controls the car.
        # The takeover policy replaces it with current human input inside the env.
        action_dim = self.env.action_space.shape[0] if hasattr(self.env.action_space, 'shape') else 2
        a_novice = np.zeros(action_dim, dtype=np.float32)
        
        # Step environment - let takeover system handle control
        step_next_obs, reward, terminated, truncated, info = self.env.step(a_novice)
        
        # 修正：如果环境需要渲染窗口来接收键盘输入，必须调用render
        env_cfg = _safe_env_config(self.env)
        if env_cfg is not None and env_cfg.get('use_render', False):
            try:
                self.env.render()
            except Exception:
                pass  # Headless 模式下忽略错误
        
        # 计算环境步进时间
        step_time = time.time() - start_time
        info['perf/step_time'] = step_time

        # Extract human action from environment info.
        # In pure demo mode: a_behavior MUST equal a_human, but this data should not be treated as online PV intervention.
        a_behavior_raw, a_human_raw, _ = self._extract_pvp_actions(info, a_novice)
        if "human_action" not in info and "raw_action" not in info:
            a_human_raw = a_behavior_raw
        a_behavior = a_human_raw.copy()
        a_human = a_human_raw.copy()

        # Pure Stage1a demo is BC-only supervision: keep it out of TD/PV.
        stop_td = 1.0
        if getattr(self, "disable_stop_td_mask", False):
            stop_td = 0.0
        
        # Apply reward-free setting if enabled
        raw_reward = float(reward)
        if self.reward_free:
            store_reward = 0.0
        else:
            store_reward = raw_reward

        # Track episode statistics via shared helper
        self._update_episode_stats(raw_reward=raw_reward, store_reward=store_reward, intervention=1.0, info=info)

        # Log episode stats when episode ends
        episode_done = self._safe_done(terminated, truncated)
        if episode_done:
            # 修正5：episode级warning - 检查takeover rate
            demo_takeover_rate = self._episode_stats["interventions"] / max(self._episode_stats["length"], 1)
            if demo_takeover_rate < 0.95:
                print(f"⚠️  Stage1a demo takeover_rate={demo_takeover_rate:.3f} < 0.95, demo may not be pure human control")
            
            # Determine episode outcome
            outcome = self._episode_outcome_from_stats(self._episode_stats)
            
            # Add episode metrics to info
            info['episode/length'] = self._episode_stats['length']
            info['episode/return'] = self._episode_stats['return']
            info['episode/interventions'] = self._episode_stats['interventions']
            info['episode/outcome'] = outcome
            
            # Calculate and log real environment intervention rate
            episode_intervention_rate = self._episode_stats['interventions'] / max(1, self._episode_stats['length'])
            info['episode/intervention_rate'] = episode_intervention_rate
            
            # Update last_metrics for display
            self.last_metrics['episode/return'] = self._episode_stats['return']
            self.last_metrics['episode/stored_return'] = self._episode_stats['stored_return']
            self.last_metrics['episode/length'] = self._episode_stats['length']
            self.last_metrics['episode/interventions'] = self._episode_stats['interventions']
            self.last_metrics['episode/intervention_rate'] = episode_intervention_rate
            
            # 🎯 记录原始奖励到tensorboard (即使在reward-free模式下)
            if hasattr(self, 'logger') and hasattr(self, 'sample_log'):
                current_step = self.sample_log.sample_step
                # 记录原始奖励（用于监控真实性能）
                self.logger.add_scalar("demo_episode/raw_return", self._episode_stats['return'], current_step)
                self.logger.add_scalar("demo_episode/stored_return", self._episode_stats['stored_return'], current_step)
                self.logger.add_scalar("demo_episode/cost", self._episode_stats['cost'], current_step)
                self.logger.add_scalar("demo_episode/native_cost", self._episode_stats['native_cost'], current_step)
                self.logger.add_scalar("demo_episode/takeover_cost", self._episode_stats['takeover_cost'], current_step)
                # 计算并记录平均奖励
                avg_raw_reward = self._episode_stats['return'] / max(1, self._episode_stats['length'])
                avg_stored_reward = self._episode_stats['stored_return'] / max(1, self._episode_stats['length'])
                self.logger.add_scalar("demo_episode/avg_raw_return", avg_raw_reward, current_step)
                self.logger.add_scalar("demo_episode/avg_stored_return", avg_stored_reward, current_step)
                
                # 如果是reward-free模式，额外记录对比信息
                if self.reward_free:
                    self.logger.add_scalar("demo_episode/reward_free_mode", 1.0, current_step)
                    self.logger.add_scalar("demo_episode/raw_vs_stored_diff", 
                                         self._episode_stats['return'] - self._episode_stats['stored_return'], 
                                         current_step)
            
            # Reset episode stats (but preserve outcome for BC-Boost calculation)
            cached_outcome = outcome
            
            self._episode_stats = {
                'length': 0,
                'return': 0.0,
                'stored_return': 0.0,
                'cost': 0.0,
                'native_cost': 0.0,
                'takeover_cost': 0.0,
                'interventions': 0,
                'crash': False,
                'out_of_road': False,
                'success': False
            }
        
        # 决定返回给下一步的obs
        next_obs_for_return = step_next_obs
        if episode_done:
            try:
                reset_obs, reset_info = self.env.reset()
                next_obs_for_return = reset_obs
            except Exception as e:
                print(f"⚠️  Error resetting environment: {e}")
                pass
        
        # Create PVP experience for pure demo
        experience = self.data_collector.create_pvp_experience(
            obs=obs,
            a_novice=a_novice,  # Zero action (AI doesn't act)
            a_human=a_human,     # Human action
            a_behavior=a_behavior, # Human action as behavior
            reward=store_reward,
            next_obs=step_next_obs,
            done=episode_done,
            intervention=0.0,
            stop_td=stop_td,
            is_pre_takeover=0.0,
            is_demo=1.0,
        )
        
        # Store in buffer with demo metadata
        current_step = getattr(self.sample_log, 'sample_step', 0)
        
        # Determine outcome for BC-Boost weighting
        if episode_done:
            outcome = 5
        else:
            outcome = 5

        # Add to buffer - all data is human demo data
        self.buffer.add(experience, outcome=outcome, timestamp=current_step, is_pre_takeover=False)
        
        # Update sample log
        sl.accumulator.add("actions", a_behavior)
        sl.accumulator.add("rewards", reward)
        sl.accumulator.add("terminations", self._safe_done(terminated, truncated))
        sl.accumulator.add("truncations", truncated)
        sl.accumulator.add("infos", info)
        
        # Increment sample step
        sl.add(reward, terminated, truncated, info)
        
        return next_obs_for_return

    def sample(self, sample_key: jax.Array, obs: np.ndarray):
        """Sample one step with PVP data collection"""
        import time
        start_time = time.time()
        
        sl = self.sample_log
        cached_outcome = "timeout"

        # Sample agent action (this is a_novice) - 使用原始动作，不做任何滤波
        if getattr(self, "use_value_guidance", False) and getattr(self, "_guidance_armed", False):
            guidance_kwargs = dict(
                lambda_0=self.guidance_lambda0,
                beta_unc=self.guidance_beta_unc,
                p_decay=self.guidance_p_decay,
                grad_clip=self.guidance_grad_clip,
                guidance_mode=self.guidance_mode,
                guidance_target=self.guidance_target,
                guidance_step_interval=self.guidance_step_interval,
                guidance_q_agg=self.guidance_q_agg,
                guidance_kappa=self.guidance_kappa,
                guidance_injection=self.guidance_injection,
                guidance_schedule=self.guidance_schedule,
            )
            if hasattr(self.algorithm, "get_action_guided_with_metrics"):
                a_novice, guidance_info = self.algorithm.get_action_guided_with_metrics(sample_key, obs, **guidance_kwargs)
                self.last_metrics.update(guidance_info)
                if hasattr(self, "logger"):
                    for tag, value in guidance_info.items():
                        self.logger.add_scalar(tag, float(value), int(sl.sample_step))
            else:
                a_novice = self.algorithm.get_action_guided(sample_key, obs, **guidance_kwargs)
        else:
            a_novice = self.algorithm.get_action(sample_key, obs)
        # Convert JAX array to numpy to avoid type issues with environment
        a_novice = np.asarray(a_novice, dtype=np.float32)
        a_novice = np.clip(a_novice, -1.0, 1.0)
        
        # Step environment with 原始动作 - 让BC学习真正的纠错
        step_next_obs, reward, terminated, truncated, info = self.env.step(a_novice)
        
        # 计算环境步进时间
        step_time = time.time() - start_time
        info['perf/step_time'] = step_time
        
        # Extract canonical PVP actions from env info
        a_behavior, a_human, current_intervention = self._extract_pvp_actions(info, a_novice)
        
        if current_intervention:
            # Human接管步数
            if hasattr(self, 'stage2_human_steps'):
                self.stage2_human_steps += 1
        else:
            # Agent执行步数
            if hasattr(self, 'stage2_agent_steps'):
                self.stage2_agent_steps += 1
        
        # Compute stop_td flag based on takeover boundaries (start AND end)
        takeover_start, takeover_end = self._resolve_takeover_boundaries(info, current_intervention)
        
        # TD mask: block learning at both takeover start AND end boundaries
        stop_td = 1.0 if (takeover_start or takeover_end) else 0.0
        if getattr(self, "disable_stop_td_mask", False):
            stop_td = 0.0
        
        # Apply reward-free setting (uniform 0.0 reward)
        raw_reward = reward  # Keep original reward for statistics
        if self.reward_free:
            store_reward = 0.0  # Uniform 0.0 reward in reward_free mode
        else:
            store_reward = reward  # Use original reward for buffer
        
        # Track episode statistics using shared helper
        self._update_episode_stats(raw_reward=raw_reward, store_reward=store_reward, intervention=float(current_intervention), info=info)
        
        # Log episode stats when episode ends
        episode_done = self._safe_done(terminated, truncated)
        if episode_done:
            # Determine episode outcome
            outcome = self._episode_outcome_from_stats(self._episode_stats)
            
            # Add episode metrics to info
            info['episode/length'] = self._episode_stats['length']
            info['episode/return'] = self._episode_stats['return']
            info['episode/interventions'] = self._episode_stats['interventions']
            info['episode/outcome'] = outcome
            
            # Calculate and log real environment intervention rate
            episode_intervention_rate = self._episode_stats['interventions'] / max(1, self._episode_stats['length'])
            info['episode/intervention_rate'] = episode_intervention_rate
            
            # Update last_metrics for display
            self.last_metrics['episode/return'] = self._episode_stats['return']
            self.last_metrics['episode/stored_return'] = self._episode_stats['stored_return']
            self.last_metrics['episode/length'] = self._episode_stats['length']
            self.last_metrics['episode/interventions'] = self._episode_stats['interventions']
            self.last_metrics['episode/intervention_rate'] = episode_intervention_rate
            
            # 添加episode结果的one-hot指标到tensorboard
            outcome_crash = 1.0 if outcome == "crash" else 0.0
            outcome_oor = 1.0 if outcome == "out_of_road" else 0.0
            outcome_succ = 1.0 if outcome == "success" else 0.0
            outcome_time = 1.0 if outcome == "timeout" else 0.0
            
            # 记录到tensorboard（使用当前sample_step）
            if hasattr(self, 'logger') and hasattr(self, 'sample_log'):
                current_step = self.sample_log.sample_step
                self.logger.add_scalar("episode/outcome/crash", outcome_crash, current_step)
                self.logger.add_scalar("episode/outcome/out_of_road", outcome_oor, current_step)
                self.logger.add_scalar("episode/outcome/success", outcome_succ, current_step)
                self.logger.add_scalar("episode/outcome/timeout", outcome_time, current_step)
                self.logger.add_scalar("episode/intervention_rate", episode_intervention_rate, current_step)
                self.logger.add_scalar("episode/cost", self._episode_stats['cost'], current_step)
                self.logger.add_scalar("episode/native_cost", self._episode_stats['native_cost'], current_step)
                self.logger.add_scalar("episode/takeover_cost", self._episode_stats['takeover_cost'], current_step)
                self.logger.add_scalar("train_cost/episode_cost", self._episode_stats['cost'], current_step)
                self.logger.add_scalar("train_cost/native_cost", self._episode_stats['native_cost'], current_step)
                self.logger.add_scalar("train_cost/takeover_cost", self._episode_stats['takeover_cost'], current_step)
                self.logger.add_scalar("train_cost/intervention_rate", episode_intervention_rate, current_step)
                self.logger.add_scalar("train_cost/intervention_count", self._episode_stats['interventions'], current_step)
                
                # 🎯 记录原始奖励到tensorboard (即使在reward-free模式下)
                self.logger.add_scalar("episode/raw_return", self._episode_stats['return'], current_step)
                self.logger.add_scalar("episode/stored_return", self._episode_stats['stored_return'], current_step)
                # 计算并记录平均奖励
                avg_raw_reward = self._episode_stats['return'] / max(1, self._episode_stats['length'])
                avg_stored_reward = self._episode_stats['stored_return'] / max(1, self._episode_stats['length'])
                self.logger.add_scalar("episode/avg_raw_return", avg_raw_reward, current_step)
                self.logger.add_scalar("episode/avg_stored_return", avg_stored_reward, current_step)
                
                # 如果是reward-free模式，额外记录对比信息
                if self.reward_free:
                    self.logger.add_scalar("debug/reward_free_mode", 1.0, current_step)
                    self.logger.add_scalar("debug/raw_vs_stored_diff", 
                                         self._episode_stats['return'] - self._episode_stats['stored_return'], 
                                         current_step)
            
            # Reset episode stats (but preserve outcome for BC-Boost calculation)
            cached_outcome = outcome
            
            self._episode_stats = {
                'length': 0,
                'return': 0.0,
                'stored_return': 0.0,  # NEW: reset stored reward for debugging
                'cost': 0.0,
                'native_cost': 0.0,
                'takeover_cost': 0.0,
                'interventions': 0,
                'crash': False,
                'out_of_road': False,
                'success': False
            }
        
        # 决定返回给下一步的obs（关键修复）
        next_obs_for_return = step_next_obs
        if episode_done:
            try:
                reset_obs, reset_info = self.env.reset()
                next_obs_for_return = reset_obs
            except Exception as e:
                print(f"⚠️  Error resetting environment: {e}")
                # Fallback: try to continue with current observation
                pass
        
        # Create PVP experience using step_next_obs（关键：必须是step返回的next_obs）
        if getattr(self, "merge_action_semantics", False):
            a_novice = np.asarray(a_behavior, dtype=np.float32)
            a_human = np.asarray(a_behavior, dtype=np.float32)

        experience = self.data_collector.create_pvp_experience(
            obs=obs,
            a_novice=a_novice,
            a_human=a_human,
            a_behavior=a_behavior,
            reward=store_reward,  # Use modified reward for buffer
            next_obs=step_next_obs,  # 关键修复：使用step_next_obs而不是next_obs_for_return
            done=episode_done,
            intervention=current_intervention,
            stop_td=stop_td,
            is_pre_takeover=0.0,  # 正常数据不是pre-takeover
        )
        
        # Update previous intervention for next step
        self.prev_intervention = current_intervention
        
        # Store in buffer with BC-Boost metadata
        current_step = getattr(self.sample_log, 'sample_step', 0)
        
        # Determine outcome for BC-Boost weighting
        if episode_done:
            outcome = self._outcome_code_from_label(cached_outcome)
        else:
            outcome = 0  # normal for ongoing episodes
        
        # Add to buffer with enhanced metadata
        self.buffer.add(experience, outcome=outcome, timestamp=current_step, is_pre_takeover=False)
        
        # BC-Boost enhancement: Add pre-takeover expert data if available
        if hasattr(self.env, 'get_pre_takeover_experiences'):
            pre_takeover_exps = self.env.get_pre_takeover_experiences()
            pre_takeover_metadata = self.env.get_pre_takeover_metadata()
            
            for i, (pre_exp, metadata) in enumerate(zip(pre_takeover_exps, pre_takeover_metadata)):
                # Use the outcome from the takeover that triggered these pre-takeover experiences
                pre_outcome = metadata.get('outcome', outcome)
                pre_timestamp = metadata.get('original_step', current_step)
                
                # Keep pre-takeover semantics clean:
                #   intervention = 0.0   (not an actual takeover step)
                #   is_pre_takeover = 1.0
                # Route-to-expert-buffer should be handled by (intervention OR is_pre_takeover),
                # not by faking intervention=1.0 here.
                pre_exp_with_flag = self.data_collector.create_pvp_experience(
                    obs=pre_exp.obs,
                    a_novice=pre_exp.actions_novice,
                    a_human=pre_exp.actions_human,
                    a_behavior=pre_exp.actions_behavior,
                    reward=0.0 if self.reward_free else (float(pre_exp.reward.item()) if hasattr(pre_exp.reward, 'item') else float(pre_exp.reward)),
                    next_obs=pre_exp.next_obs,
                    done=bool(pre_exp.done.item()) if hasattr(pre_exp.done, 'item') else bool(pre_exp.done),
                    intervention=0.0,
                    stop_td=float(pre_exp.stop_td.item()) if hasattr(pre_exp.stop_td, 'item') else float(pre_exp.stop_td),
                    is_pre_takeover=1.0,
                    is_demo=0.0,
                )
                
                # Add pre-takeover experience with highest priority
                self.buffer.add(
                    pre_exp_with_flag, 
                    outcome=4,  # pre_takeover gets special outcome code
                    timestamp=pre_timestamp, 
                    is_pre_takeover=True
                )
            
            # Clear pre-takeover experiences after adding to buffer
            if len(pre_takeover_exps) > 0:
                self.env.clear_pre_takeover_experiences()
        
        # Update sample log
        sl.accumulator.add("actions", a_behavior)
        sl.accumulator.add("rewards", reward)
        sl.accumulator.add("terminations", self._safe_done(terminated, truncated))
        sl.accumulator.add("truncations", truncated)
        sl.accumulator.add("infos", info)
        
        # Increment sample step to ensure training loop progresses
        sl.add(reward, terminated, truncated, info)
        
        return next_obs_for_return

    def update(self, update_key: jax.Array):
        """Update algorithm with PVP batch (avoid GPU->CPU sync every step)"""
        import time
        import numpy as np

        ul = self.update_log

        # 预测 ul.add() 之后的 step（通常 UpdateLog.add 会 +1）
        next_update_step = ul.update_step + 1
        do_log = (next_update_step % self.update_log_n_step == 0)

        t0 = time.time()
        data = self.buffer.sample(self.batch_size, to_jax=True)
        algo_info = self.algorithm.update(update_key, data)
        update_time = time.time() - t0

        # 总是把算法指标更新到last_metrics（不受日志频率限制）
        algo_info_host = jax.device_get(algo_info)
        immediate_prefixes = (
            "policy/",
            "critic/",
            "pv/",
            "q/",
            "grad/",
            "action/",
            "takeover/",
            "td/",
            "batch/",
            "human/",
        )
        for k, v in algo_info_host.items():
            v = np.asarray(v)
            # 标量：直接 float；非标量：取 mean
            value = float(v) if v.size == 1 else float(v.mean())
            self.last_metrics[k] = value
            
            # 立即记录关键算法指标到tensorboard，不受do_log限制
            if hasattr(self, 'logger') and isinstance(value, (int, float, np.number)) and not np.isnan(float(value)):
                if k.startswith(immediate_prefixes) or any(key in k for key in ['loss', 'alpha', 'entropy']):
                    tag = k if "/" in k else f'algorithm/{k}'
                    # 使用当前update_step作为步数
                    self.logger.add_scalar(tag, float(value), ul.update_step)

        # 仅放"纯 python 不触发同步"的指标
        info = {
            "perf/update_time": float(update_time),
        }

        if hasattr(self.buffer, 'human_buffer') and hasattr(self.buffer, 'novice_buffer'):
            info["buffer/size_human"] = int(len(self.buffer.human_buffer))
            info["buffer/size_novice"] = int(len(self.buffer.novice_buffer))
            info["buffer/size_total"] = int(len(self.buffer))
            if hasattr(self.buffer, "get_human_buffer_stats"):
                for k, v in self.buffer.get_human_buffer_stats().items():
                    info[f"buffer/human_{k}"] = float(v)

        if do_log:
            # ---- 一次性把 algo_info 拉回 host（减少多次同步）----
            algo_info_host = jax.device_get(algo_info)

            def to_float(v):
                v = np.asarray(v)
                # 标量：直接 float；非标量：取 mean（防止再崩）
                return float(v) if v.size == 1 else float(v.mean())

            for k, v in algo_info_host.items():
                info[k] = to_float(v)

            # ---- 这些额外统计只在 log 步做（否则每步同步太亏）----
            extra = {}
            if hasattr(data, "interventions"):
                extra["batch/intervention_mix"] = jnp.mean(data.interventions)
            if hasattr(data, "stop_td"):
                extra["td/stop_td_rate"] = jnp.mean(data.stop_td)

            if hasattr(data, "actions_human") and hasattr(data, "actions_novice") and hasattr(data, "interventions"):
                I = data.interventions.squeeze(-1)
                diff = jnp.mean(jnp.abs(data.actions_human - data.actions_novice), axis=-1)
                extra["action/diff_human_novice_on_I"] = jnp.sum(diff * I) / (jnp.sum(I) + 1e-6)

            if hasattr(data, "actions_behavior") and hasattr(data, "actions_novice"):
                diff = jnp.mean(jnp.abs(data.actions_behavior - data.actions_novice), axis=-1)
                extra["action/diff_behavior_novice"] = jnp.mean(diff)

            if hasattr(data, "actions_behavior"):
                clip = jnp.any(jnp.abs(data.actions_behavior) >= 1.0, axis=-1)
                extra["action/clip_frac"] = jnp.mean(clip)

            if hasattr(data, "actions_human"):
                human_zero = jnp.all(jnp.abs(data.actions_human) < 1e-6, axis=-1)
                extra["human/input_zero_rate"] = jnp.mean(human_zero)

            extra_host = jax.device_get(extra)
            for k, v in extra_host.items():
                info[k] = to_float(v)

            # PV 派生指标（在 host 上用 python float 算，不再触发同步）
            if ("pv/q1_h_mean" in info and "pv/q2_h_mean" in info and
                "pv/q1_n_mean" in info and "pv/q2_n_mean" in info):
                info["pv/q_human_mean"] = 0.5 * (info["pv/q1_h_mean"] + info["pv/q2_h_mean"])
                info["pv/q_novice_mean"] = 0.5 * (info["pv/q1_n_mean"] + info["pv/q2_n_mean"])
                info["pv/margin_mean"] = 0.5 * ((info["pv/q1_h_mean"] - info["pv/q1_n_mean"]) +
                                                (info["pv/q2_h_mean"] - info["pv/q2_n_mean"]))

            if "q1_mean" in info and "q2_mean" in info:
                info["q/mean_avg"] = 0.5 * (info["q1_mean"] + info["q2_mean"])

        ul.add(info)
        if do_log:
            ul.log(self.add_scalar)

    def train(self, key: jax.Array):
        """Main training loop with PVP enhancements.

        Fixes:
          - Periodic policy saving is executed INSIDE the training loop (and won't miss checkpoints when sample_per_iteration>1)
          - TensorBoard curves are made continuous by avoiding mixed step systems on the SAME tag
          - tqdm progress bar is aligned to current sample_step (important for two-stage training or resuming)
        """
        self.key = key

        # Reset env and do warmup (fill buffer) if configured
        obs, _ = self.env.reset()
        obs = self.warmup(self.start_step)

        # Resume progress bar; also align it with current sample_step
        self.progress.unpause()
        try:
            self.progress.total = self.total_step
            self.progress.n = int(self.sample_log.sample_step)
            self.progress.refresh()
        except Exception:
            pass

        sl, ul = self.sample_log, self.update_log

        # Create per-iteration PRNG key splitter
        iter_key_fn = create_iter_key_fn(key, self.sample_per_iteration, self.update_per_iteration)

        print(f"\n=== Starting Training ===")
        print(f"Total training steps: {self.total_step}")
        print(f"Sample per iteration: {self.sample_per_iteration}")
        print(f"Update per iteration: {self.update_per_iteration}")
        print(f"Reward-free mode: {self.reward_free}")
        print(f"Policy save interval: {self.save_policy_every} steps")

        # -----------------------
        # Robust periodic saving schedule
        # (works even when sample_per_iteration>1 and when starting from non-zero step)
        # -----------------------
        save_every = self.save_policy_every if isinstance(self.save_policy_every, int) else 0
        next_save_step = None
        if save_every and save_every > 0:
            if sl.sample_step > 0 and (sl.sample_step % save_every == 0):
                # allow immediate save if starting exactly on a multiple (e.g., stage2 resume at 2000)
                next_save_step = int(sl.sample_step)
            else:
                next_save_step = (int(sl.sample_step) // save_every + 1) * save_every

        # Logging cadence
        print_interval = 100
        q_eval_interval = 500
        last_print_step = int(sl.sample_step)
        last_q_eval_step = int(sl.sample_step)

        try:
            while sl.sample_step <= self.total_step:
                if (
                    self.use_value_guidance
                    and not self._guidance_armed
                    and int(sl.sample_step) >= int(getattr(self, "_guidance_arm_step", 0))
                ):
                    self.arm_value_guidance(True)
                    print(f"UPV-GDS guidance armed @ sample_step={int(sl.sample_step)}")

                # Keys for this training "iteration" are derived from current sample_step
                sample_keys, update_keys = iter_key_fn(sl.sample_step)

                # ---------- Sample ----------
                for i in range(self.sample_per_iteration):
                    obs = self.sample(sample_keys[i], obs)

                    # Save policy checkpoints *during* training
                    if next_save_step is not None and sl.sample_step >= next_save_step:
                        policy_pkl_name = self.policy_pkl_template.format(
                            sample_step=next_save_step,
                            update_step=ul.update_step
                        )
                        path = self.log_path / policy_pkl_name
                        self.algorithm.save_policy(str(path))

                        # Notify external evaluator if enabled
                        if self.evaluator is not None:
                            command = f"{next_save_step},{path}\n"
                            self.evaluator.stdin.write(command.encode())

                        print(f"\n💾 Saved policy checkpoint: {path}")
                        next_save_step += save_every

                # ---------- Update ----------
                for i in range(self.update_per_iteration):
                    self.update(update_keys[i])

                # Progress bar advances by sampled env steps
                self.progress.update(self.sample_per_iteration)

                # ---------- Print + TensorBoard (sample_step axis) ----------
                if int(sl.sample_step) - last_print_step >= print_interval:
                    current_step = int(sl.sample_step)
                    progress_pct = (current_step / self.total_step) * 100

                    # Buffer sizes
                    if hasattr(self.buffer, 'novice_buffer') and hasattr(self.buffer, 'human_buffer'):
                        buffer_n = len(self.buffer.novice_buffer)
                        buffer_h = len(self.buffer.human_buffer)
                        buffer_info = f"Agent: {buffer_n:4d} | Human: {buffer_h:4d}"
                    else:
                        buffer_info = f"Buffer: {len(self.buffer)}"

                    # Recent episode snapshot (these metrics are updated when an episode ends)
                    last_episode_return = float(self.last_metrics.get('episode/return', 0.0))
                    last_episode_length = float(self.last_metrics.get('episode/length', 0))
                    last_episode_interventions = float(self.last_metrics.get('episode/interventions', 0))

                    print(
                        f" Step {current_step:6d}/{self.total_step} ({progress_pct:5.1f}%) | "
                        f"{buffer_info} | Reward: {last_episode_return:6.2f} | "
                        f"Interventions: {int(last_episode_interventions)}"
                    )

                    # Occasionally estimate Q-values (debug)
                    if current_step - last_q_eval_step >= q_eval_interval:
                        try:
                            obs_jax = jnp.asarray(obs).reshape(1, -1)
                            # create a one-off key for eval
                            self.key, key_eval = jax.random.split(self.key)
                            policy_params, log_alpha = self.algorithm.get_policy_params()
                            a_novice = self.algorithm.agent.get_action(key_eval, (policy_params, log_alpha), obs_jax)

                            # NOTE: human action is unknown here; we use 0-action as a stable reference
                            q1_h, _ = self.algorithm.agent.q(self.algorithm.state.params.q1, obs_jax, jnp.array([[0.0, 0.0]]))
                            q1_n, _ = self.algorithm.agent.q(self.algorithm.state.params.q1, obs_jax, a_novice)

                            ref_q = float(q1_h.squeeze())
                            novice_q = float(q1_n.squeeze())
                            print(f"   Q-values: Ref(0,0): {ref_q:6.2f} | Novice: {novice_q:6.2f}")

                            self.logger.add_scalar('q_values/ref_q', ref_q, current_step)
                            self.logger.add_scalar('q_values/novice_q', novice_q, current_step)
                        except Exception as e:
                            print(f"   Q evaluation skipped: {e}")
                        last_q_eval_step = current_step

                    # ✅ IMPORTANT: do NOT re-log algorithm losses here using a different step system.
                    # Update() already logs losses with a consistent step axis (update_step).
                    # Here we only log progress snapshots using sample_step.
                    self.logger.add_scalar('training/progress_pct', progress_pct, current_step)
                    if hasattr(self.buffer, 'novice_buffer'):
                        self.logger.add_scalar('training/buffer_novice_size', len(self.buffer.novice_buffer), current_step)
                    else:
                        self.logger.add_scalar('training/buffer_size', len(self.buffer), current_step)
                    if hasattr(self.buffer, 'human_buffer'):
                        self.logger.add_scalar('training/buffer_human_size', len(self.buffer.human_buffer), current_step)
                    
                    # 🔄 2阶段统计：记录human和agent步数
                    if hasattr(self, 'stage2_started') and self.stage2_started:
                        if hasattr(self, 'stage2_human_steps'):
                            self.logger.add_scalar('stage2/human_steps', self.stage2_human_steps, current_step)
                        if hasattr(self, 'stage2_agent_steps'):
                            self.logger.add_scalar('stage2/agent_steps', self.stage2_agent_steps, current_step)
                        
                        # Calculate and log stage2 human step rate
                        total_steps = self.stage2_human_steps + self.stage2_agent_steps
                        if total_steps > 0:
                            human_step_rate = self.stage2_human_steps / total_steps
                            self.logger.add_scalar('stage2/human_step_rate', human_step_rate, current_step)

                    # Snapshot episode metrics (use distinct tags to avoid mixing with SampleLog tags)
                    self.logger.add_scalar('episode/return_snapshot', last_episode_return, current_step)
                    self.logger.add_scalar('episode/length_snapshot', last_episode_length, current_step)
                    self.logger.add_scalar('episode/interventions_snapshot', last_episode_interventions, current_step)

                    last_print_step = current_step

                # Episode-level logging
                if self.sample_log_interval.check(self.sample_log.sample_episode):
                    self.sample_log.log(self.add_scalar)
                    self.logger.flush()

                if int(sl.sample_step) % 100 == 0:
                    self.logger.flush()

        finally:
            # Always save final policy checkpoint
            final_policy_path = self.log_path / f"final_policy_step_{int(sl.sample_step)}.pkl"
            try:
                self.algorithm.save_policy(str(final_policy_path))
                self.algorithm.save_policy(str(self.log_path / "final_policy.pkl"))
                print(f"\n✅ Training completed! Final step: {int(sl.sample_step)}")
                print(f"💾 Final policy saved: {final_policy_path}")
            except Exception as e:
                print(f"⚠️  Final policy save failed: {e}")

            self.logger.flush()

        return int(sl.sample_step)

    def configure_value_guidance(
        self,
        enable: bool,
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
    ):
        self.use_value_guidance = bool(enable)
        self.guidance_lambda0 = float(lambda_0)
        self.guidance_beta_unc = float(beta_unc)
        self.guidance_p_decay = float(p_decay)
        self.guidance_grad_clip = float(grad_clip)
        self.guidance_mode = int(guidance_mode)
        self.guidance_target = int(guidance_target)
        self.guidance_step_interval = int(guidance_step_interval)
        self.guidance_q_agg = int(guidance_q_agg)
        self.guidance_kappa = float(guidance_kappa)
        self.guidance_injection = int(guidance_injection)
        self.guidance_schedule = int(guidance_schedule)

    def arm_value_guidance(self, armed: bool = True):
        """Enable guidance after the critic has completed warmup."""
        self._guidance_armed = bool(armed)

    def add_scalar(self, tag: str, value: float, step: int):
        """Log scalar to tensorboard and store in last_metrics"""
        self.last_metrics[tag] = value
        if hasattr(self, 'logger'):
            self.logger.add_scalar(tag, value, step)

    def finish(self):
        """Finish training"""
        self.env.close()
        self.algorithm.save(self.log_path / "state.pkl")

        # ✅ Avoid mixing step systems on the same tag:
        # Log final metrics under a separate namespace (final/*) so TensorBoard curves stay intact.
        if len(self.last_metrics) > 0:
            print(f" Writing final metrics to tensorboard (final/*)...")
            step = int(self.sample_log.sample_step)
            for tag, value in self.last_metrics.items():
                try:
                    self.logger.add_scalar(f"final/{tag}", float(value), step)
                except Exception:
                    continue

        self.logger.flush()

        if self.hparams is not None and len(self.last_metrics) > 0:
            exp, ssi, sei = hparams(self.hparams, self.last_metrics)
            self.logger.file_writer.add_summary(exp)
            self.logger.file_writer.add_summary(ssi)
            self.logger.file_writer.add_summary(sei)

        self.logger.close()
        self.progress.close()

        if self.evaluator is not None:
            self.evaluator.stdin.close()
            self.evaluator.wait()

        print(f" Training data saved to: {self.log_path}")
        print(f" Tensorboard logs: {self.log_path}")
        print(f" Run: tensorboard --logdir {self.log_path}")


def create_iter_key_fn(key: jax.Array, sample_per_iteration: int, update_per_iteration: int) -> Callable[[int], Tuple[jax.Array, jax.Array]]:
    """Create a key function that properly handles dynamic step values."""
    
    # Remove JIT compilation to avoid concretization issues
    def iter_key_fn(key, step):
        step_int = int(step)  # Ensure step is treated as integer, not JAX array
        k = jax.random.fold_in(key, step_int)
        k_sample, k_update = jax.random.split(k, 2)
        return k_sample, k_update
    
    def get_iter_key(step):
        nonlocal key  # Use the outer key
        key, subkey = jax.random.split(key)
        sample_key, update_key = iter_key_fn(subkey, step)
        if sample_per_iteration > 1:
            sample_key = jax.random.split(sample_key, sample_per_iteration)
        else:
            sample_key = (sample_key,)
        if update_per_iteration > 1:
            update_key = jax.random.split(update_key, update_per_iteration)
        else:
            update_key = (update_key,)
        return sample_key, update_key
    
    return get_iter_key
