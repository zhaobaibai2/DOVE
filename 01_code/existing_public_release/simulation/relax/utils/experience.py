from typing import NamedTuple, Optional, Tuple, Callable, TYPE_CHECKING

import numpy as np
import jax

from relax.utils.done_utils import combine_done

if TYPE_CHECKING:
    pass

def probe_batch_size(reward: "jax.Array") -> Optional[int]:
    try:
        if reward.ndim > 0:
            return reward.shape[0]
        else:
            return None
    except AttributeError:
        return None

class Experience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"

    # 新增（PVP/HACO）：
    actions_behavior: "jax.Array"  # [B, act_dim]  执行动作（一般等于 action；冗余是为了清晰）
    actions_novice: "jax.Array"  # [B, act_dim]  智能体原计划动作（被人类替换时≠behavior）
    actions_human: "jax.Array"   # [B, act_dim]  人类动作（接管时的方向盘/手柄输入）
    interventions: "jax.Array"  # [B, 1]        是否接管（0/1）
    stop_td: "jax.Array"  # [B, 1]        TD masking flag（1=mask TD，0=keep TD）
    is_pre_takeover: "jax.Array"  # [B, 1]  是否是pre-takeover专家数据（1=是，0=否）
    is_demo: "jax.Array"  # [B, 1]  是否是纯人工 Stage1a demo（1=是，0=否）
    pair_ok: "jax.Array"  # [B, 1]  该样本是否具有同状态 rejected-action pair 语义



    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"Experience(size={self.batch_size()})"








    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return Experience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) to match buffer
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros((*leading_dims, 1), dtype=np.float32),    # Fixed: (B, 1) to match buffer
            actions_behavior=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            actions_novice=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            actions_human=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            interventions=np.zeros((*leading_dims, 1), dtype=np.float32),
            stop_td=np.zeros((*leading_dims, 1), dtype=np.float32),
            is_pre_takeover=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: 缺少此字段
            is_demo=np.zeros((*leading_dims, 1), dtype=np.float32),
            pair_ok=np.zeros((*leading_dims, 1), dtype=np.float32),
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info=None):
        # Default values for PVP fields (compatible with old interface)
        actions_behavior = action
        actions_novice = action
        actions_human = action
        interventions = 0.0
        stop_td = 0.0
        is_pre_takeover = 0.0  # 默认不是pre-takeover数据
        is_demo = 0.0
        pair_ok = 0.0
        
        # Fixed: Ensure reward and done have correct (B,1) dimensions
        if isinstance(reward, (int, float)):
            reward = np.array([[float(reward)]], dtype=np.float32)  # Shape: (1, 1)
        elif isinstance(reward, np.ndarray) and reward.ndim == 1:
            reward = reward.reshape(-1, 1)  # Shape: (B, 1)
            
        # Fixed: Use safe done combination
        done_val = combine_done(terminated, truncated)
        if isinstance(done_val, bool):
            done = np.array([[float(done_val)]], dtype=np.float32)  # Shape: (1, 1)
        elif isinstance(done_val, np.ndarray) and done_val.ndim == 1:
            done = done_val.reshape(-1, 1)  # Shape: (B, 1)
        else:
            done = done_val
        
        # Override with PVP data if available in info
        if info is not None:
            actions_behavior = info.get("behavior_action", actions_behavior)
            actions_human = info.get("human_action", info.get("raw_action", actions_human))
            actions_novice = info.get("agent_action", actions_novice)
            
            # Fixed: Handle both scalar and array inputs for interventions
            takeover_val = info.get("takeover", False)
            if isinstance(takeover_val, (np.ndarray, jax.Array)):
                interventions = takeover_val.astype(np.float32)
                # Ensure (B,1) shape
                if interventions.ndim == 1:
                    interventions = interventions.reshape(-1, 1)
            else:
                interventions = np.array([[float(takeover_val)]], dtype=np.float32)  # Shape: (1, 1)
            
            # 统一stop_td定义：takeover_start OR takeover_end
            # Fixed: Handle both scalar and array inputs for takeover flags
            takeover_start_val = info.get("takeover_start", False)
            takeover_end_val = info.get("takeover_end", False)
            
            # Convert to boolean arrays safely for vector env compatibility
            if isinstance(takeover_start_val, (np.ndarray, jax.Array)):
                takeover_start = takeover_start_val.astype(bool)
            else:
                takeover_start = bool(takeover_start_val)
                
            if isinstance(takeover_end_val, (np.ndarray, jax.Array)):
                takeover_end = takeover_end_val.astype(bool)
            else:
                takeover_end = bool(takeover_end_val)
            
            # Fixed: Handle both scalar and array inputs for stop_td
            if isinstance(takeover_val, (np.ndarray, jax.Array)):
                # For array inputs, compute stop_td element-wise
                takeover_start_arr = np.broadcast_to(takeover_start, takeover_val.shape)
                takeover_end_arr = np.broadcast_to(takeover_end, takeover_val.shape)
                stop_td = np.where(takeover_start_arr | takeover_end_arr, 1.0, 0.0).astype(np.float32)
                # Ensure (B,1) shape
                if stop_td.ndim == 1:
                    stop_td = stop_td.reshape(-1, 1)
            else:
                stop_td = np.array([[1.0 if (takeover_start or takeover_end) else 0.0]], dtype=np.float32)  # Shape: (1, 1)
        
        return Experience(
            obs=obs, 
            action=action, 
            reward=reward, 
            done=done, 
            next_obs=next_obs,
            actions_behavior=actions_behavior,
            actions_novice=actions_novice,
            actions_human=actions_human,
            interventions=interventions,
            stop_td=stop_td,
            is_pre_takeover=is_pre_takeover,
            is_demo=is_demo,
            pair_ok=pair_ok,
        )

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
        is_demo: float = 0.0,  # 是否是纯人工 demo
        pair_ok: Optional[float] = None,  # 是否是真正的同状态 a^+/a^- intervention pair；None 时按字段语义推断
    ) -> 'Experience':
        """Create PVP Experience with all required fields"""
        # Fixed: Support both scalar and array inputs for vector env compatibility
        def to_array_2d(val, dtype=np.float32):
            """Convert scalar or array to (B,1) or (B,D) format"""
            if isinstance(val, (int, float, bool)):
                return np.array([[float(val)]], dtype=dtype)  # Shape: (1, 1)
            elif isinstance(val, np.ndarray):
                if val.ndim == 0:
                    return np.array([[float(val)]], dtype=dtype)  # (1, 1)
                elif val.ndim == 1:
                    return val.reshape(-1, 1)  # (B, 1)
                else:
                    return val  # Already multi-dimensional
            elif isinstance(val, jax.Array):
                if val.ndim == 0:
                    return np.array([[float(val)]], dtype=dtype)  # (1, 1)
                elif val.ndim == 1:
                    return np.array(val).reshape(-1, 1)  # (B, 1)
                else:
                    return np.array(val)  # Already multi-dimensional
            else:
                return np.array([[float(val)]], dtype=dtype)  # Fallback
        
        # Convert all fields to consistent shapes
        reward_arr = to_array_2d(reward, np.float32)
        done_arr = to_array_2d(done, np.float32)
        intervention_arr = to_array_2d(intervention, np.float32)
        stop_td_arr = to_array_2d(stop_td, np.float32)
        is_pre_takeover_arr = to_array_2d(is_pre_takeover, np.float32)
        is_demo_arr = to_array_2d(is_demo, np.float32)

        if pair_ok is None:
            # Conservative default for TeX alignment: a sample is a valid
            # same-state rejected-action pair only if it is an actual
            # intervention step, is not pure demo / pre-takeover context, and
            # the rejected autonomous action differs from the positive action.
            try:
                pos = np.asarray(a_human if a_human is not None else a_behavior, dtype=np.float32)
                neg = np.asarray(a_novice, dtype=np.float32)
                gap = np.linalg.norm((pos - neg).reshape(-1))
                semantic_pair = float(gap > 1e-8)
            except Exception:
                semantic_pair = 0.0
            pair_ok = (
                float(np.max(intervention_arr) > 0.5)
                * float(np.max(is_pre_takeover_arr) <= 0.5)
                * float(np.max(is_demo_arr) <= 0.5)
                * semantic_pair
            )
        pair_ok_arr = to_array_2d(pair_ok, np.float32)
        
        return Experience(
            obs=obs,
            action=a_behavior,  # legacy field - equals a_behavior
            reward=reward_arr,
            done=done_arr,
            next_obs=next_obs,
            actions_behavior=a_behavior,
            actions_novice=a_novice,
            actions_human=a_human,
            interventions=intervention_arr,
            stop_td=stop_td_arr,
            is_pre_takeover=is_pre_takeover_arr,
            is_demo=is_demo_arr,
            pair_ok=pair_ok_arr,
        )

class GAEExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    ret: "jax.Array"
    adv: "jax.Array"

    def batch_size(self) -> Optional[int]:
        return probe_batch_size(self.reward)

    def __repr__(self):
        return f"GAEExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return GAEExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) for consistency
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            done=np.zeros((*leading_dims, 1), dtype=np.float32),    # Fixed: (B, 1) for consistency
            ret=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) for consistency
            adv=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) for consistency
        )

class SafeExperience(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"
    reward: "jax.Array"
    done: "jax.Array"
    next_obs: "jax.Array"
    cost: "jax.Array"
    feasible: "jax.Array"
    infeasible: "jax.Array"
    barrier: "jax.Array"
    next_barrier: "jax.Array"

    def batch_size(self) -> Optional[int]:
        try:
            if self.reward.ndim > 0:
                return self.reward.shape[0]
            else:
                return None
        except AttributeError:
            return None

    def __repr__(self):
        return f"SafeExperience(size={self.batch_size()})"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return SafeExperience(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
            reward=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) for consistency
            done=np.zeros((*leading_dims, 1), dtype=np.float32),    # Fixed: (B, 1) for consistency
            next_obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            cost=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) for consistency
            feasible=np.zeros((*leading_dims, 1), dtype=np.bool_),    # Fixed: (B, 1) for consistency
            infeasible=np.zeros((*leading_dims, 1), dtype=np.bool_),  # Fixed: (B, 1) for consistency
            barrier=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) for consistency
            next_barrier=np.zeros((*leading_dims, 1), dtype=np.float32),  # Fixed: (B, 1) for consistency
        )

    @staticmethod
    def create(obs, action, reward, terminated, truncated, next_obs, info: dict):
        cost = info.get("cost", 0.0)
        feasible = info.get("feasible", False)
        infeasible = info.get("infeasible", False)
        barrier = info.get("barrier", 0.0)
        next_barrier = info.get("next_barrier", 0.0)
        return SafeExperience(
            obs=obs,
            action=action,
            reward=reward,
            done=terminated,
            next_obs=next_obs,
            cost=cost,
            feasible=feasible,
            infeasible=infeasible,
            barrier=barrier,
            next_barrier=next_barrier,
        )

class ObsActionPair(NamedTuple):
    obs: "jax.Array"
    action: "jax.Array"

    @staticmethod
    def create_example(obs_dim: int, action_dim: int, batch_size: Optional[int] = None):
        leading_dims = (batch_size,) if batch_size is not None else ()
        return ObsActionPair(
            obs=np.zeros((*leading_dims, obs_dim), dtype=np.float32),
            action=np.zeros((*leading_dims, action_dim), dtype=np.float32),
        )

