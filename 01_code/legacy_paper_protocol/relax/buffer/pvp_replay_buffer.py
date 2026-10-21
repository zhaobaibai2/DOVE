"""
PVP (Preventive Value Pruning) Replay Buffer for DACER
Supports three-action semantics + intervention/stop_td flags + balanced sampling
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Dict, Any, Tuple, Optional
from typing import NamedTuple, Optional, Tuple, List, Any

import numpy as np
import jax
import jax.numpy as jnp

from relax.buffer.base import Buffer
from relax.utils.experience import Experience


def as_scalar(x):
    """安全提取标量，避免shape=(1,)的float()转换警告"""
    return float(np.asarray(x).reshape(()))


def _to_scalar(x):
    """修正4：安全提取标量，用于路由判断"""
    arr = np.asarray(x)
    return float(arr.reshape(-1)[0])


def _save_pickle(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


class PVPBatch(NamedTuple):
    """Batch structure for PVP training"""
    obs: jax.Array
    next_obs: jax.Array
    done: jax.Array
    reward: jax.Array
    action: jax.Array             # alias for actions_behavior (for compatibility)
    actions_behavior: jax.Array  # a_b: actually executed action
    actions_novice: jax.Array    # a_n: agent's intended action  
    actions_human: jax.Array     # a_h: human action (during intervention)
    interventions: jax.Array     # intervention flag (0/1)
    stop_td: jax.Array          # TD masking flag (0/1)
    is_pre_takeover: jax.Array  # pre-takeover标记 (0/1)
    is_demo: jax.Array          # 纯人工 Stage1a demo 标记 (0/1)
    pair_ok: jax.Array          # true same-state intervention pair flag (0/1)


class PVPSingleBuffer:
    """
    Single ring-buffer for PVP data storage
    Stores all fields including three actions and intervention flags
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        capacity: int,
        seed: int = 0,
    ):
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        
        self._ptr = 0
        self._size = 0
        
        # Initialize storage arrays
        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.done = np.zeros((self.capacity, 1), dtype=np.float32)
        self.reward = np.zeros((self.capacity, 1), dtype=np.float32)
        
        # PVP-specific fields
        self.actions_behavior = np.zeros((self.capacity, *action_shape), dtype=np.float32)
        self.actions_novice = np.zeros((self.capacity, *action_shape), dtype=np.float32)
        self.actions_human = np.zeros((self.capacity, *action_shape), dtype=np.float32)
        self.interventions = np.zeros((self.capacity, 1), dtype=np.float32)
        self.stop_td = np.zeros((self.capacity, 1), dtype=np.float32)
        self.is_pre_takeover = np.zeros((self.capacity, 1), dtype=np.float32)  # pre-takeover标记
        self.is_demo = np.zeros((self.capacity, 1), dtype=np.float32)  # pure demo 标记
        self.pair_ok = np.zeros((self.capacity, 1), dtype=np.float32)  # true same-state pair flag
        
        # BC-Boost optimization fields
        self.outcome = np.zeros(self.capacity, dtype=np.int8)  # 0=normal, 1=crash, 2=out_of_road, 3=success, 4=pre_takeover, 5=demo
        self.timestamp = np.zeros(self.capacity, dtype=np.int32)  # Global step counter for recency

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        experience: Experience,
        *,
        from_jax: bool = False,
        outcome: int = 0,
        timestamp: int = 0,
        is_pre_takeover: bool = False,
    ) -> None:
        """Add a single experience to buffer
        
        Args:
            experience: Experience to add
            from_jax: Whether data is from JAX
            outcome: Episode outcome (0=normal, 1=crash, 2=out_of_road, 3=success, 4=pre_takeover)
            timestamp: Global step counter for recency tracking
            is_pre_takeover: Fallback value if experience doesn't have is_pre_takeover field
        """
        i = self._ptr
        
        # Convert from JAX if needed
        if from_jax:
            def to_np(x): return np.array(x) if hasattr(x, '__array__') else x
            obs = to_np(experience.obs)
            next_obs = to_np(experience.next_obs)
            done = to_np(experience.done)
            reward = to_np(experience.reward)
            actions_behavior = to_np(experience.actions_behavior)
            actions_novice = to_np(experience.actions_novice)
            actions_human = to_np(experience.actions_human)
            interventions = to_np(experience.interventions)
            stop_td = to_np(experience.stop_td)
            exp_is_pre_takeover = to_np(experience.is_pre_takeover) if hasattr(experience, "is_pre_takeover") else is_pre_takeover
            exp_is_demo = to_np(experience.is_demo) if hasattr(experience, "is_demo") else 0.0
            exp_pair_ok = to_np(experience.pair_ok) if hasattr(experience, "pair_ok") else 1.0
        else:
            obs = experience.obs
            next_obs = experience.next_obs
            done = experience.done
            reward = experience.reward
            actions_behavior = experience.actions_behavior
            actions_novice = experience.actions_novice
            actions_human = experience.actions_human
            interventions = experience.interventions
            stop_td = experience.stop_td
            exp_is_pre_takeover = experience.is_pre_takeover if hasattr(experience, "is_pre_takeover") else is_pre_takeover
            exp_is_demo = experience.is_demo if hasattr(experience, "is_demo") else 0.0
            exp_pair_ok = experience.pair_ok if hasattr(experience, "pair_ok") else 1.0

        # Store data
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.done[i, 0] = as_scalar(done)
        self.reward[i, 0] = as_scalar(reward)
        
        # Store PVP data - 动作数组直接赋值，不用as_scalar
        self.actions_behavior[i] = actions_behavior
        self.actions_novice[i] = actions_novice
        self.actions_human[i] = actions_human
        self.interventions[i, 0] = as_scalar(interventions)
        self.stop_td[i, 0] = as_scalar(stop_td)
        self.is_pre_takeover[i, 0] = as_scalar(exp_is_pre_takeover)  # 优先存储experience自带标记
        self.is_demo[i, 0] = as_scalar(exp_is_demo)
        self.pair_ok[i, 0] = as_scalar(exp_pair_ok)
        
        # Store BC-Boost metadata
        self.outcome[i] = outcome
        self.timestamp[i] = timestamp

        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def add_batch(self, experiences: Experience, *, from_jax: bool = False, outcome: int = 0, timestamp: int = 0, is_pre_takeover: bool = False) -> None:
        """Add a batch of experiences (Vectorized) - PERFORMANCE OPTIMIZED"""
        if from_jax:
            # 如果是 JAX 数组，先转为 Numpy，避免后续多次 device_get
            def to_np(x): return np.asarray(x)
        else:
            def to_np(x): return x

        # 获取 batch_size
        if hasattr(experiences.obs, 'shape'):
            batch_size = experiences.obs.shape[0]
        else:
            batch_size = len(experiences.obs)
            
        if batch_size == 0:
            return

        # 计算索引（处理环形缓冲区回绕）
        idxs = np.arange(self._ptr, self._ptr + batch_size) % self.capacity
        
        # 准备数据（如果是 JAX，这里会触发同步，但是一次性同步比循环好）
        obs = to_np(experiences.obs)
        next_obs = to_np(experiences.next_obs)
        
        # 确保维度正确 (B, 1) 或 (B,) 处理
        def safe_2d(arr):
            arr = to_np(arr)
            return arr.reshape(-1, 1) if arr.ndim == 1 else arr

        done = safe_2d(experiences.done)
        reward = safe_2d(experiences.reward)
        interventions = safe_2d(experiences.interventions)
        stop_td = safe_2d(experiences.stop_td)
        pair_ok = safe_2d(experiences.pair_ok) if hasattr(experiences, "pair_ok") else np.ones_like(interventions)
        
        # 动作处理
        act_beh = to_np(experiences.actions_behavior)
        act_nov = to_np(experiences.actions_novice)
        act_hum = to_np(experiences.actions_human)

        # 批量写入（利用 Numpy 的高级索引处理回绕）
        self.obs[idxs] = obs
        self.next_obs[idxs] = next_obs
        self.done[idxs] = done
        self.reward[idxs] = reward
        self.actions_behavior[idxs] = act_beh
        self.actions_novice[idxs] = act_nov
        self.actions_human[idxs] = act_hum
        self.interventions[idxs] = interventions
        self.stop_td[idxs] = stop_td
        self.pair_ok[idxs] = pair_ok
        # FIXED: Use is_pre_takeover from experiences if available, otherwise use parameter
        if hasattr(experiences, 'is_pre_takeover') and experiences.is_pre_takeover is not None:
            pre_takeover_data = to_np(experiences.is_pre_takeover)
            # Ensure correct shape
            if pre_takeover_data.ndim == 1:
                pre_takeover_data = pre_takeover_data.reshape(-1, 1)
            self.is_pre_takeover[idxs] = pre_takeover_data
        else:
            # Fallback to parameter value (broadcast to batch)
            self.is_pre_takeover[idxs] = is_pre_takeover
        if hasattr(experiences, 'is_demo') and experiences.is_demo is not None:
            demo_data = to_np(experiences.is_demo)
            if demo_data.ndim == 1:
                demo_data = demo_data.reshape(-1, 1)
            self.is_demo[idxs] = demo_data
        else:
            self.is_demo[idxs] = 0.0
        
        # Store BC-Boost metadata for batch
        self.outcome[idxs] = outcome
        self.timestamp[idxs] = timestamp

        # 更新指针和大小
        self._ptr = (self._ptr + batch_size) % self.capacity
        self._size = min(self._size + batch_size, self.capacity)

    def sample(self, batch_size: int, *, to_jax: bool = False) -> PVPBatch:
        """Sample a batch from buffer"""
        if self._size == 0:
            # Return empty batch with correct dimensions for multi-dim obs/actions
            # Respect to_jax parameter to avoid type mixing
            xp = jnp if to_jax else np
            dtype = jnp.float32 if to_jax else np.float32
            return PVPBatch(
                obs=xp.zeros((0, *self.obs_shape), dtype=dtype),
                action=xp.zeros((0, *self.action_shape), dtype=dtype),
                reward=xp.zeros((0, 1), dtype=dtype),
                done=xp.zeros((0, 1), dtype=dtype),
                next_obs=xp.zeros((0, *self.obs_shape), dtype=dtype),
                actions_behavior=xp.zeros((0, *self.action_shape), dtype=dtype),
                actions_novice=xp.zeros((0, *self.action_shape), dtype=dtype),
                actions_human=xp.zeros((0, *self.action_shape), dtype=dtype),
                interventions=xp.zeros((0, 1), dtype=dtype),
                stop_td=xp.zeros((0, 1), dtype=dtype),
                is_pre_takeover=xp.zeros((0, 1), dtype=dtype),
                is_demo=xp.zeros((0, 1), dtype=dtype),
                pair_ok=xp.zeros((0, 1), dtype=dtype),
            )
        idx = self.rng.integers(0, self._size, size=batch_size)

        def to_jax_array(x: np.ndarray) -> jax.Array:
            """使用 JAX 默认设备放置"""
            return jax.device_put(jnp.array(x))  # JAX 会自动选择最优设备

        batch = PVPBatch(
            obs=to_jax_array(self.obs[idx]) if to_jax else self.obs[idx],
            next_obs=to_jax_array(self.next_obs[idx]) if to_jax else self.next_obs[idx],
            done=to_jax_array(self.done[idx]) if to_jax else self.done[idx],
            reward=to_jax_array(self.reward[idx]) if to_jax else self.reward[idx],
            action=to_jax_array(self.actions_behavior[idx]) if to_jax else self.actions_behavior[idx],  # alias
            actions_behavior=to_jax_array(self.actions_behavior[idx]) if to_jax else self.actions_behavior[idx],
            actions_novice=to_jax_array(self.actions_novice[idx]) if to_jax else self.actions_novice[idx],
            actions_human=to_jax_array(self.actions_human[idx]) if to_jax else self.actions_human[idx],
            interventions=to_jax_array(self.interventions[idx]) if to_jax else self.interventions[idx],
            stop_td=to_jax_array(self.stop_td[idx]) if to_jax else self.stop_td[idx],
            is_pre_takeover=to_jax_array(self.is_pre_takeover[idx]) if to_jax else self.is_pre_takeover[idx],
            is_demo=to_jax_array(self.is_demo[idx]) if to_jax else self.is_demo[idx],
            pair_ok=to_jax_array(self.pair_ok[idx]) if to_jax else self.pair_ok[idx],
        )
        return batch

    def sample_with_indices(self, batch_size: int, *, to_jax: bool = False) -> Tuple[PVPBatch, np.ndarray]:
        """Sample with returned indices"""
        if self._size == 0:
            # Return empty batch and empty indices with correct dimensions for multi-dim obs/actions
            # Respect to_jax parameter to avoid type mixing
            xp = jnp if to_jax else np
            dtype = jnp.float32 if to_jax else np.float32
            empty_batch = PVPBatch(
                obs=xp.zeros((0, *self.obs_shape), dtype=dtype),
                action=xp.zeros((0, *self.action_shape), dtype=dtype),
                reward=xp.zeros((0, 1), dtype=dtype),
                done=xp.zeros((0, 1), dtype=dtype),
                next_obs=xp.zeros((0, *self.obs_shape), dtype=dtype),
                actions_behavior=xp.zeros((0, *self.action_shape), dtype=dtype),
                actions_novice=xp.zeros((0, *self.action_shape), dtype=dtype),
                actions_human=xp.zeros((0, *self.action_shape), dtype=dtype),
                interventions=xp.zeros((0, 1), dtype=dtype),
                stop_td=xp.zeros((0, 1), dtype=dtype),
                is_pre_takeover=xp.zeros((0, 1), dtype=dtype),
                is_demo=xp.zeros((0, 1), dtype=dtype),
                pair_ok=xp.zeros((0, 1), dtype=dtype),
            )
            return empty_batch, np.array([], dtype=np.int32)
        idx = self.rng.integers(0, self._size, size=batch_size)
        
        # 直接使用 idx 构造 batch，而不是重新采样
        def to_jax_array(x: np.ndarray) -> jax.Array:
            """使用 JAX 默认设备放置"""
            return jax.device_put(jnp.array(x))  # JAX 会自动选择最优设备
        
        batch = PVPBatch(
            obs=to_jax_array(self.obs[idx]) if to_jax else self.obs[idx],
            next_obs=to_jax_array(self.next_obs[idx]) if to_jax else self.next_obs[idx],
            done=to_jax_array(self.done[idx]) if to_jax else self.done[idx],
            reward=to_jax_array(self.reward[idx]) if to_jax else self.reward[idx],
            action=to_jax_array(self.actions_behavior[idx]) if to_jax else self.actions_behavior[idx],
            actions_behavior=to_jax_array(self.actions_behavior[idx]) if to_jax else self.actions_behavior[idx],
            actions_novice=to_jax_array(self.actions_novice[idx]) if to_jax else self.actions_novice[idx],
            actions_human=to_jax_array(self.actions_human[idx]) if to_jax else self.actions_human[idx],
            interventions=to_jax_array(self.interventions[idx]) if to_jax else self.interventions[idx],
            stop_td=to_jax_array(self.stop_td[idx]) if to_jax else self.stop_td[idx],
            is_pre_takeover=to_jax_array(self.is_pre_takeover[idx]) if to_jax else self.is_pre_takeover[idx],
            is_demo=to_jax_array(self.is_demo[idx]) if to_jax else self.is_demo[idx],
            pair_ok=to_jax_array(self.pair_ok[idx]) if to_jax else self.pair_ok[idx],
        )
        return batch, idx

    def replace(self, indices: np.ndarray, experiences: Experience, *, from_jax: bool = False) -> None:
        """Replace experiences at given indices"""
        for i, idx in enumerate(indices):
            if from_jax:
                exp = Experience(
                    obs=experiences.obs[i],
                    action=experiences.action[i],
                    reward=experiences.reward[i], 
                    done=experiences.done[i],
                    next_obs=experiences.next_obs[i],
                    actions_behavior=experiences.actions_behavior[i],
                    actions_novice=experiences.actions_novice[i],
                    actions_human=experiences.actions_human[i],
                    interventions=experiences.interventions[i],
                    stop_td=experiences.stop_td[i],
                    is_pre_takeover=experiences.is_pre_takeover[i],
                    is_demo=experiences.is_demo[i],
                    pair_ok=experiences.pair_ok[i] if hasattr(experiences, "pair_ok") else 1.0,
                )
            else:
                exp = experiences[i]
            
            self.obs[idx] = exp.obs
            self.next_obs[idx] = exp.next_obs
            self.done[idx, 0] = as_scalar(exp.done)
            self.reward[idx, 0] = as_scalar(exp.reward)
            self.actions_behavior[idx] = exp.actions_behavior
            self.actions_novice[idx] = exp.actions_novice
            self.actions_human[idx] = exp.actions_human
            self.interventions[idx, 0] = as_scalar(exp.interventions)
            self.stop_td[idx, 0] = as_scalar(exp.stop_td)
            self.is_pre_takeover[idx, 0] = as_scalar(exp.is_pre_takeover)
            self.is_demo[idx, 0] = as_scalar(exp.is_demo)
            self.pair_ok[idx, 0] = as_scalar(exp.pair_ok) if hasattr(exp, "pair_ok") else 1.0

    def save(self, path):
        """Save this single buffer."""
        _save_pickle(self, path)


class PVPBalancedDualBuffer(Buffer[PVPBatch]):
    """
    PVP-style dual buffer with category-aware sampling:
    - novice_buffer: autonomous rollout samples
    - human_buffer: online takeover / optional pre-takeover / pure demo samples
    Main DOVEER defaults mirror the TeX pair semantics:
      novice 75%, true online intervention 15%, pre-takeover 0%, demo 10%

    Pre-takeover sampling is kept only as an explicit soft-context ablation.
    """
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_shape: Tuple[int, ...],
        capacity_novice: int,
        capacity_human: int,
        seed: int = 0,
        novice_fraction: float = 0.75,
        intervention_fraction: float = 0.15,
        pre_takeover_fraction: float = 0.0,
        demo_fraction: float = 0.10,
    ):
        self.novice_buffer = PVPSingleBuffer(obs_shape, action_shape, capacity_novice, seed)
        self.human_buffer = PVPSingleBuffer(obs_shape, action_shape, capacity_human, seed + 1)
        self.rng = np.random.default_rng(seed + 2)
        sample_fracs = np.asarray(
            [
                float(novice_fraction),
                float(intervention_fraction),
                float(pre_takeover_fraction),
                float(demo_fraction),
            ],
            dtype=np.float64,
        )
        sample_fracs = np.clip(sample_fracs, 0.0, None)
        if float(sample_fracs.sum()) <= 0.0:
            sample_fracs = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        sample_fracs = sample_fracs / sample_fracs.sum()
        self.novice_fraction = float(sample_fracs[0])
        self.intervention_fraction = float(sample_fracs[1])
        self.pre_takeover_fraction = float(sample_fracs[2])
        self.demo_fraction = float(sample_fracs[3])

    def __len__(self) -> int:
        return len(self.novice_buffer) + len(self.human_buffer)

    def add(
        self,
        experience: Experience,
        *,
        from_jax: bool = False,
        outcome: int = 0,
        timestamp: int = 0,
        is_pre_takeover: bool = False,
    ) -> None:
        """
        Add a single experience to the appropriate buffer.
        
        路由逻辑同时检查 intervention / is_pre_takeover / is_demo。
        只要任一为真，就路由到 human_buffer (expert buffer)。
        """
        # Extract intervention flag
        intervention = _to_scalar(np.array(experience.interventions) if from_jax else experience.interventions)
        
        # Extract pre_takeover flag
        pre_takeover_flag = _to_scalar(
            np.array(experience.is_pre_takeover) if (from_jax and hasattr(experience, "is_pre_takeover"))
            else (experience.is_pre_takeover if hasattr(experience, "is_pre_takeover") else is_pre_takeover)
        )
        demo_flag = _to_scalar(
            np.array(experience.is_demo) if (from_jax and hasattr(experience, "is_demo"))
            else (experience.is_demo if hasattr(experience, "is_demo") else 0.0)
        )
        
        # Route to appropriate buffer
        route_to_human = (intervention >= 0.5) or (pre_takeover_flag >= 0.5) or (demo_flag >= 0.5)
        
        if route_to_human:
            self.human_buffer.add(experience, from_jax=from_jax, outcome=outcome, timestamp=timestamp, is_pre_takeover=is_pre_takeover)
        else:
            self.novice_buffer.add(experience, from_jax=from_jax, outcome=outcome, timestamp=timestamp, is_pre_takeover=is_pre_takeover)

    def add_batch(self, experiences: Experience, *, from_jax: bool = False, outcome: int = 0, timestamp: int = 0, is_pre_takeover: bool = False) -> None:
        """Add batch, routing each experience to appropriate buffer (Vectorized)"""
        if from_jax:
            batch_size = experiences.obs.shape[0]
            interventions = np.array(experiences.interventions)
        else:
            batch_size = len(experiences.obs)
            interventions = experiences.interventions
            
        if batch_size == 0:
            return
            
        # 转换为numpy数组处理
        interventions_np = np.asarray(interventions)
        if interventions_np.ndim == 2:
            interventions_np = interventions_np.squeeze(-1)
        
        # 同时检查 is_pre_takeover / is_demo
        pre_takeover_np = np.asarray(experiences.is_pre_takeover) if hasattr(experiences, 'is_pre_takeover') else np.zeros_like(interventions_np)
        if pre_takeover_np.ndim == 2:
            pre_takeover_np = pre_takeover_np.squeeze(-1)
        demo_np = np.asarray(experiences.is_demo) if hasattr(experiences, 'is_demo') else np.zeros_like(interventions_np)
        if demo_np.ndim == 2:
            demo_np = demo_np.squeeze(-1)
        
        # 创建布尔mask来分离数据
        human_mask = (interventions_np >= 0.5) | (pre_takeover_np >= 0.5) | (demo_np >= 0.5)
        novice_mask = ~human_mask
        
        # 分离experiences（向量化操作）
        def split_experiences(mask):
            if from_jax:
                def slice_jax(x):
                    x_np = np.array(x)
                    return x_np[mask] if mask.any() else x_np[:0]
                return Experience(
                    obs=slice_jax(experiences.obs),
                    action=slice_jax(experiences.action),
                    reward=slice_jax(experiences.reward),
                    done=slice_jax(experiences.done),
                    next_obs=slice_jax(experiences.next_obs),
                    actions_behavior=slice_jax(experiences.actions_behavior),
                    actions_novice=slice_jax(experiences.actions_novice),
                    actions_human=slice_jax(experiences.actions_human),
                    interventions=slice_jax(experiences.interventions),
                    stop_td=slice_jax(experiences.stop_td),
                    is_pre_takeover=slice_jax(experiences.is_pre_takeover),
                    is_demo=slice_jax(experiences.is_demo),
                    pair_ok=slice_jax(experiences.pair_ok) if hasattr(experiences, "pair_ok") else np.ones_like(slice_jax(experiences.interventions)),
                )
            else:
                def slice_np(x):
                    x_arr = np.array(x)
                    return x_arr[mask] if mask.any() else x_arr[:0]
                return Experience(
                    obs=slice_np(experiences.obs),
                    action=slice_np(experiences.action),
                    reward=slice_np(experiences.reward),
                    done=slice_np(experiences.done),
                    next_obs=slice_np(experiences.next_obs),
                    actions_behavior=slice_np(experiences.actions_behavior),
                    actions_novice=slice_np(experiences.actions_novice),
                    actions_human=slice_np(experiences.actions_human),
                    interventions=slice_np(experiences.interventions),
                    stop_td=slice_np(experiences.stop_td),
                    is_pre_takeover=slice_np(experiences.is_pre_takeover),
                    is_demo=slice_np(experiences.is_demo),
                    pair_ok=slice_np(experiences.pair_ok) if hasattr(experiences, "pair_ok") else np.ones_like(slice_np(experiences.interventions)),
                )
        
        # 批量添加到对应的buffer
        if human_mask.any():
            human_exps = split_experiences(human_mask)
            self.human_buffer.add_batch(human_exps, from_jax=from_jax, outcome=outcome, timestamp=timestamp, is_pre_takeover=is_pre_takeover)
            
        if novice_mask.any():
            novice_exps = split_experiences(novice_mask)
            self.novice_buffer.add_batch(novice_exps, from_jax=from_jax, outcome=outcome, timestamp=timestamp, is_pre_takeover=is_pre_takeover)

    def _sample_with_replacement(self, buffer: PVPSingleBuffer, batch_size: int, *, to_jax: bool = False) -> PVPBatch:
        """Sample with replacement from a buffer"""
        if len(buffer) == 0:
            raise RuntimeError("Cannot sample from empty buffer")
            
        idx = self.rng.integers(0, len(buffer), size=batch_size)
        return self._sample_from_indices(buffer, idx, to_jax=to_jax)

    def _sample_from_indices(self, buffer: PVPSingleBuffer, indices: np.ndarray, *, to_jax: bool = False) -> PVPBatch:
        """Sample from specific indices in a buffer"""
        def to_jax_array(x: np.ndarray) -> jax.Array:
            return jax.device_put(jnp.array(x))
            
        return PVPBatch(
            obs=to_jax_array(buffer.obs[indices]) if to_jax else buffer.obs[indices],
            next_obs=to_jax_array(buffer.next_obs[indices]) if to_jax else buffer.next_obs[indices],
            done=to_jax_array(buffer.done[indices]) if to_jax else buffer.done[indices],
            reward=to_jax_array(buffer.reward[indices]) if to_jax else buffer.reward[indices],
            action=to_jax_array(buffer.actions_behavior[indices]) if to_jax else buffer.actions_behavior[indices],
            actions_behavior=to_jax_array(buffer.actions_behavior[indices]) if to_jax else buffer.actions_behavior[indices],
            actions_novice=to_jax_array(buffer.actions_novice[indices]) if to_jax else buffer.actions_novice[indices],
            actions_human=to_jax_array(buffer.actions_human[indices]) if to_jax else buffer.actions_human[indices],
            interventions=to_jax_array(buffer.interventions[indices]) if to_jax else buffer.interventions[indices],
            stop_td=to_jax_array(buffer.stop_td[indices]) if to_jax else buffer.stop_td[indices],
            is_pre_takeover=to_jax_array(buffer.is_pre_takeover[indices]) if to_jax else buffer.is_pre_takeover[indices],
            is_demo=to_jax_array(buffer.is_demo[indices]) if to_jax else buffer.is_demo[indices],
            pair_ok=to_jax_array(buffer.pair_ok[indices]) if to_jax else buffer.pair_ok[indices],
        )

    def _to_jax_batch(self, batch: PVPBatch) -> PVPBatch:
        """Convert a PVPBatch to JAX arrays"""
        def to_jax_array(x: np.ndarray) -> jax.Array:
            return jax.device_put(jnp.array(x))
            
        return PVPBatch(
            obs=to_jax_array(batch.obs),
            next_obs=to_jax_array(batch.next_obs),
            done=to_jax_array(batch.done),
            reward=to_jax_array(batch.reward),
            action=to_jax_array(batch.action),
            actions_behavior=to_jax_array(batch.actions_behavior),
            actions_novice=to_jax_array(batch.actions_novice),
            actions_human=to_jax_array(batch.actions_human),
            interventions=to_jax_array(batch.interventions),
            stop_td=to_jax_array(batch.stop_td),
            is_pre_takeover=to_jax_array(batch.is_pre_takeover),
            is_demo=to_jax_array(batch.is_demo),
            pair_ok=to_jax_array(batch.pair_ok),
        )

    def _concat_batches(self, batch1: PVPBatch, batch2: PVPBatch) -> PVPBatch:
        """Concatenate two PVP batches"""
        # Use more reliable JAX/NumPy detection
        is_jax = hasattr(batch1.obs, "device") or hasattr(batch1.obs, "aval")
        cat = jnp.concatenate if is_jax else np.concatenate
        return PVPBatch(
            obs=cat([batch1.obs, batch2.obs], axis=0),
            next_obs=cat([batch1.next_obs, batch2.next_obs], axis=0),
            action=cat([batch1.action, batch2.action], axis=0),
            reward=cat([batch1.reward, batch2.reward], axis=0),
            done=cat([batch1.done, batch2.done], axis=0),
            actions_behavior=cat([batch1.actions_behavior, batch2.actions_behavior], axis=0),
            actions_novice=cat([batch1.actions_novice, batch2.actions_novice], axis=0),
            actions_human=cat([batch1.actions_human, batch2.actions_human], axis=0),
            interventions=cat([batch1.interventions, batch2.interventions], axis=0),
            stop_td=cat([batch1.stop_td, batch2.stop_td], axis=0),
            is_pre_takeover=cat([batch1.is_pre_takeover, batch2.is_pre_takeover], axis=0),
            is_demo=cat([batch1.is_demo, batch2.is_demo], axis=0),
            pair_ok=cat([batch1.pair_ok, batch2.pair_ok], axis=0),
        )

    def _human_category_indices(self) -> Dict[str, np.ndarray]:
        size = len(self.human_buffer)
        if size == 0:
            empty = np.array([], dtype=np.int32)
            return {
                "intervention": empty,
                "pre_takeover": empty,
                "demo": empty,
            }

        inter = self.human_buffer.interventions[:size, 0] > 0.5
        pre = self.human_buffer.is_pre_takeover[:size, 0] > 0.5
        demo = self.human_buffer.is_demo[:size, 0] > 0.5

        demo_mask = demo
        pre_mask = pre & (~demo_mask)
        intervention_mask = inter & (~pre_mask) & (~demo_mask)

        return {
            "intervention": np.where(intervention_mask)[0].astype(np.int32),
            "pre_takeover": np.where(pre_mask)[0].astype(np.int32),
            "demo": np.where(demo_mask)[0].astype(np.int32),
        }

    def _compute_batch_category_counts(
        self,
        batch_size: int,
        human_indices: Dict[str, np.ndarray],
    ) -> Dict[str, int]:
        requested = {
            "novice": float(self.novice_fraction),
            "intervention": float(self.intervention_fraction),
            "pre_takeover": float(self.pre_takeover_fraction),
            "demo": float(self.demo_fraction),
        }
        available = {
            "novice": len(self.novice_buffer) > 0,
            "intervention": len(human_indices["intervention"]) > 0,
            "pre_takeover": len(human_indices["pre_takeover"]) > 0,
            "demo": len(human_indices["demo"]) > 0,
        }

        active_names = [name for name in requested if available[name]]
        if not active_names:
            raise RuntimeError("No samples available for category-aware PVP batch")

        weights = np.asarray([requested[name] for name in active_names], dtype=np.float64)
        if float(weights.sum()) <= 0.0:
            weights = np.ones_like(weights)
        weights = weights / weights.sum()

        raw = weights * int(batch_size)
        counts = np.floor(raw).astype(np.int32)
        remainder = int(batch_size) - int(counts.sum())
        if remainder > 0:
            order = np.argsort(-(raw - counts))
            for idx in order[:remainder]:
                counts[idx] += 1

        result = {name: 0 for name in requested}
        for name, count in zip(active_names, counts):
            result[name] = int(count)
        return result

    def _sample_human_category(self, indices: np.ndarray, batch_size: int, *, to_jax: bool = False) -> PVPBatch:
        if batch_size <= 0:
            return self.human_buffer.sample(0, to_jax=to_jax)
        if len(indices) == 0:
            raise RuntimeError("Cannot sample requested human category from empty pool")
        chosen = self.rng.choice(indices, size=int(batch_size), replace=True)
        return self._sample_from_indices(self.human_buffer, chosen, to_jax=to_jax)

    def sample(self, batch_size: int, *, to_jax: bool = False) -> PVPBatch:
        """Category-aware Stage2 sampling."""
        if len(self.human_buffer) == 0 and len(self.novice_buffer) == 0:
            raise RuntimeError("Both buffers empty")
        if len(self.human_buffer) == 0:
            return self.novice_buffer.sample(batch_size, to_jax=to_jax)

        human_indices = self._human_category_indices()
        counts = self._compute_batch_category_counts(batch_size, human_indices)

        batches: List[PVPBatch] = []
        if counts["novice"] > 0:
            batches.append(self._sample_with_replacement(self.novice_buffer, counts["novice"], to_jax=to_jax))
        if counts["intervention"] > 0:
            batches.append(self._sample_human_category(human_indices["intervention"], counts["intervention"], to_jax=to_jax))
        if counts["pre_takeover"] > 0:
            batches.append(self._sample_human_category(human_indices["pre_takeover"], counts["pre_takeover"], to_jax=to_jax))
        if counts["demo"] > 0:
            batches.append(self._sample_human_category(human_indices["demo"], counts["demo"], to_jax=to_jax))

        if not batches:
            raise RuntimeError("Category-aware PVP sampling produced an empty batch")

        batch = batches[0]
        for extra_batch in batches[1:]:
            batch = self._concat_batches(batch, extra_batch)
        return batch

    def sample_human_only(self, batch_size: int, *, to_jax: bool = False) -> PVPBatch:
        """Sample only from human buffer for pure BC training"""
        if len(self.human_buffer) == 0:
            raise RuntimeError("Human buffer is empty - cannot sample human-only batch")
        
        return self._sample_with_replacement(self.human_buffer, batch_size, to_jax=to_jax)

    def get_human_buffer_stats(self) -> Dict[str, float]:
        """Return composition stats for the expert-side human buffer."""
        size = len(self.human_buffer)
        if size == 0:
            return {
                "human_size": 0.0,
                "intervention_ratio": 0.0,
                "online_intervention_ratio": 0.0,
                "pre_takeover_ratio": 0.0,
                "pure_demo_ratio": 0.0,
                "demo_ratio": 0.0,
            }

        inter = self.human_buffer.interventions[:size, 0] > 0.5
        pt = self.human_buffer.is_pre_takeover[:size, 0] > 0.5
        demo = self.human_buffer.is_demo[:size, 0] > 0.5
        pair_ok = self.human_buffer.pair_ok[:size, 0] > 0.5
        online_intervention = inter & (~pt) & (~demo)

        return {
            "human_size": float(size),
            "intervention_ratio": float(np.mean(inter)),
            "online_intervention_ratio": float(np.mean(online_intervention)),
            "pre_takeover_ratio": float(np.mean(pt)),
            "pure_demo_ratio": float(np.mean(demo)),
            "demo_ratio": float(np.mean(demo)),
            "pair_ok_ratio": float(np.mean(pair_ok)),
        }

    def sample_recent_human_only(self, batch_size: int, recent_steps: int = 200, *, to_jax: bool = False) -> PVPBatch:
        """Sample only from recent human corrections for BC-Boost
        
        Args:
            batch_size: Number of samples to draw
            recent_steps: Only consider samples from last N steps
            to_jax: Whether to convert to JAX arrays
        """
        if len(self.human_buffer) == 0:
            raise RuntimeError("Human buffer is empty - cannot sample human-only batch")
        
        # Get current max timestamp for recency calculation
        current_max_step = np.max(self.human_buffer.timestamp[:len(self.human_buffer)])
        min_timestamp = max(0, current_max_step - recent_steps)
        
        # Filter indices by recency
        valid_indices = np.where(self.human_buffer.timestamp[:len(self.human_buffer)] >= min_timestamp)[0]
        
        if len(valid_indices) == 0:
            # Fallback to regular sampling if no recent data
            print(f"⚠️  No recent human data (last {recent_steps} steps), falling back to regular sampling")
            return self.sample_human_only(batch_size, to_jax=to_jax)
        
        # Sample from recent data with replacement if needed
        if len(valid_indices) >= batch_size:
            chosen_indices = self.human_buffer.rng.choice(valid_indices, batch_size, replace=False)
        else:
            # Not enough recent data, sample with replacement
            chosen_indices = self.human_buffer.rng.choice(valid_indices, batch_size, replace=True)
        
        return self._sample_from_indices(self.human_buffer, chosen_indices, to_jax=to_jax)

    def sample_weighted_human_only(self, batch_size: int, *, to_jax: bool = False) -> PVPBatch:
        """Sample human data with outcome-based weighting for BC-Boost
        
        Priority weights for BC-Boost only:
          optional pre_takeover(2.0) > online_intervention/out_of_road > crash > demo > success > normal

        This sampler is not used by EnergyRank.  It only affects optional
        behavior-cloning boost updates over expert-side data.
        """
        if len(self.human_buffer) == 0:
            raise RuntimeError("Human buffer is empty - cannot sample human-only batch")
        
        # Define outcome weights with pre-takeover getting highest priority
        outcome_weights = {
            0: 0.1,  # normal
            1: 1.0,  # crash (medium priority)
            2: 1.5,  # out_of_road (high priority)
            3: 0.3,  # success (low priority)
            4: 2.0,  # pre_takeover (highest priority - expert demonstration)
            5: 0.4,  # demo (keep for BC stability but downweight in Stage2 boost)
        }
        
        # Get weights for all samples
        outcomes = self.human_buffer.outcome[:len(self.human_buffer)]
        is_pre_takeover = self.human_buffer.is_pre_takeover[:len(self.human_buffer), 0] > 0.5
        is_demo = self.human_buffer.is_demo[:len(self.human_buffer), 0] > 0.5
        is_intervention = self.human_buffer.interventions[:len(self.human_buffer), 0] > 0.5
        
        # Apply weights: pre-takeover highest, demo lower, online corrections above normal demo replay.
        weights = np.array(
            [
                2.0
                if is_pre_takeover[i]
                else 0.4
                if is_demo[i]
                else max(1.2, outcome_weights.get(int(outcomes[i]), 0.1))
                if is_intervention[i]
                else outcome_weights.get(int(outcomes[i]), 0.1)
                for i in range(len(outcomes))
            ],
            dtype=np.float64,
        )
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Sample based on weights
        chosen_indices = self.human_buffer.rng.choice(
            len(self.human_buffer), batch_size, replace=True, p=weights
        )
        
        return self._sample_from_indices(self.human_buffer, chosen_indices, to_jax=to_jax)

    def sample_boost_human_only(self, batch_size: int, recent_steps: int = 200, *, to_jax: bool = False) -> PVPBatch:
        """Hybrid sampling for BC-Boost: prioritizes recent failures and critical outcomes
        
        Strategy: 50% recent data + 50% weighted by outcome severity
        """
        if len(self.human_buffer) == 0:
            raise RuntimeError("Human buffer is empty - cannot sample human-only batch")
        
        half_batch = batch_size // 2
        
        # Get recent samples
        try:
            recent_batch = self.sample_recent_human_only(half_batch, recent_steps, to_jax=False)
        except RuntimeError:
            recent_batch = None
        
        # Get weighted samples
        weighted_batch = self.sample_weighted_human_only(batch_size - (recent_batch.obs.shape[0] if recent_batch is not None else 0), to_jax=False)
        
        # Combine batches
        if recent_batch is not None:
            combined_batch = self._concat_batches(recent_batch, weighted_batch)
        else:
            combined_batch = weighted_batch
        
        # Convert to JAX if requested
        if to_jax:
            return self._to_jax_batch(combined_batch)
        return combined_batch

    def sample_with_indices(self, batch_size: int, *, to_jax: bool = False) -> Tuple[PVPBatch, np.ndarray]:
        """Sample with indices - returns meaningful indices for tracking"""
        # 计算human和novice样本数量
        half = batch_size // 2
        rest = batch_size - half
        
        # 分别从两个buffer采样，获取真实的batch和indices
        if half > 0:
            human_batch, human_indices = self.human_buffer.sample_with_indices(half, to_jax=to_jax)
        else:
            # 创建空的batch和indices for multi-dim obs/actions
            human_batch = PVPBatch(
                obs=jnp.zeros((0, *self.human_buffer.obs_shape), dtype=jnp.float32),
                action=jnp.zeros((0, *self.human_buffer.action_shape), dtype=jnp.float32),
                reward=jnp.zeros((0, 1), dtype=jnp.float32),
                done=jnp.zeros((0, 1), dtype=jnp.float32),
                next_obs=jnp.zeros((0, *self.human_buffer.obs_shape), dtype=jnp.float32),
                actions_behavior=jnp.zeros((0, *self.human_buffer.action_shape), dtype=jnp.float32),
                actions_novice=jnp.zeros((0, *self.human_buffer.action_shape), dtype=jnp.float32),
                actions_human=jnp.zeros((0, *self.human_buffer.action_shape), dtype=jnp.float32),
                interventions=jnp.zeros((0, 1), dtype=jnp.float32),
                stop_td=jnp.zeros((0, 1), dtype=jnp.float32),
                is_pre_takeover=jnp.zeros((0, 1), dtype=jnp.float32),
                is_demo=jnp.zeros((0, 1), dtype=jnp.float32),
                pair_ok=jnp.zeros((0, 1), dtype=jnp.float32),
            )
            human_indices = np.array([], dtype=np.int32)
        
        if rest > 0:
            novice_batch, novice_indices = self.novice_buffer.sample_with_indices(rest, to_jax=to_jax)
        else:
            obs_dim = self.novice_buffer.obs_shape[0] if self.novice_buffer.obs_shape else 1
            act_dim = self.novice_buffer.action_shape[0] if self.novice_buffer.action_shape else 1
            novice_batch = PVPBatch(
                obs=jnp.zeros((0, *self.novice_buffer.obs_shape), dtype=jnp.float32),
                action=jnp.zeros((0, *self.novice_buffer.action_shape), dtype=jnp.float32),
                reward=jnp.zeros((0, 1), dtype=jnp.float32),
                done=jnp.zeros((0, 1), dtype=jnp.float32),
                next_obs=jnp.zeros((0, *self.novice_buffer.obs_shape), dtype=jnp.float32),
                actions_behavior=jnp.zeros((0, *self.novice_buffer.action_shape), dtype=jnp.float32),
                actions_novice=jnp.zeros((0, *self.novice_buffer.action_shape), dtype=jnp.float32),
                actions_human=jnp.zeros((0, *self.novice_buffer.action_shape), dtype=jnp.float32),
                interventions=jnp.zeros((0, 1), dtype=jnp.float32),
                stop_td=jnp.zeros((0, 1), dtype=jnp.float32),
                is_pre_takeover=jnp.zeros((0, 1), dtype=jnp.float32),
                is_demo=jnp.zeros((0, 1), dtype=jnp.float32),
                pair_ok=jnp.zeros((0, 1), dtype=jnp.float32),
            )
            novice_indices = np.array([], dtype=np.int32)
        
        # 合并batch
        batch = self._concat_batches(human_batch, novice_batch)
        
        # 构建有意义的indices：(buffer_id, idx) 格式
        indices = []
        # human buffer的索引 (buffer_id=0)
        for idx in human_indices:
            indices.append((0, idx))
        # novice buffer的索引 (buffer_id=1) 
        for idx in novice_indices:
            indices.append((1, idx))
        
        return batch, np.array(indices, dtype=np.int32)

    def replace(self, indices: np.ndarray, experiences: Experience, *, from_jax: bool = False) -> None:
        """Replace experiences at given indices - routes to appropriate buffers"""
        if from_jax:
            batch_size = experiences.obs.shape[0]
            interventions = np.array(experiences.interventions)
        else:
            batch_size = len(experiences.obs)
            interventions = experiences.interventions
            
        for i, idx in enumerate(indices):
            buffer_id, real_idx = idx  # Unpack (buffer_id, idx)
            
            # Extract single experience
            if from_jax:
                exp = Experience(
                    obs=experiences.obs[i],
                    action=experiences.action[i],
                    reward=experiences.reward[i], 
                    done=experiences.done[i],
                    next_obs=experiences.next_obs[i],
                    actions_behavior=experiences.actions_behavior[i],
                    actions_novice=experiences.actions_novice[i],
                    actions_human=experiences.actions_human[i],
                    interventions=experiences.interventions[i],
                    stop_td=experiences.stop_td[i],
                    is_pre_takeover=experiences.is_pre_takeover[i],
                    is_demo=experiences.is_demo[i],
                    pair_ok=experiences.pair_ok[i] if hasattr(experiences, "pair_ok") else 1.0,
                )
            else:
                exp = experiences[i]
            
            # Route to correct buffer based on buffer_id and use proper field assignment
            if buffer_id == 0:  # human_buffer
                buffer = self.human_buffer
            elif buffer_id == 1:  # novice_buffer
                buffer = self.novice_buffer
            else:
                continue  # Skip invalid buffer_id
            
            # Proper field-by-field assignment (like PVPSingleBuffer.replace)
            buffer.obs[real_idx] = exp.obs
            buffer.next_obs[real_idx] = exp.next_obs
            buffer.done[real_idx, 0] = as_scalar(exp.done)
            buffer.reward[real_idx, 0] = as_scalar(exp.reward)
            buffer.actions_behavior[real_idx] = exp.actions_behavior
            buffer.actions_novice[real_idx] = exp.actions_novice
            buffer.actions_human[real_idx] = exp.actions_human
            buffer.interventions[real_idx, 0] = as_scalar(exp.interventions)
            buffer.stop_td[real_idx, 0] = as_scalar(exp.stop_td)
            buffer.is_pre_takeover[real_idx, 0] = as_scalar(exp.is_pre_takeover)
            buffer.is_demo[real_idx, 0] = as_scalar(exp.is_demo)

    def save(self, path):
        """Save the full dual-buffer state."""
        _save_pickle(self, path)
