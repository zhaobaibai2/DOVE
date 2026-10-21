"""PVP-DACER 三阶段训练脚本（修正版）

修正点：
1. _safe_get_deterministic_action 抓 Exception 而非仅 TypeError，保证 fallback 真实可用
2. _experience_targets_human_buffer 兼容 experience.intervention / experience.interventions 两种字段名
3. warmup 结束后同步重置 algorithm.state.step（避免 actor_delay / target_update_delay 错位）
4. Stage1b 显式重置 opt_state（lr 改变后继续用旧动量会漂）
5. Stage2 重建算法时显式同步 target_params（若存在），避免 target/main 不一致
6. loader 中 reward=0.0 加 reward_free 断言，防止未来配置漂移导致静默错误

Stage 1a: 人类演示收集（训练地图 seeds: 100–119）
Stage 1b: 离线 BC 预训练
Stage 2:  PVP + BC 在线训练（训练地图 seeds: 100–119）
"""

import os
import re
import sys
import pickle
import argparse
import time
import gc
import json
import shutil
import threading
import warnings
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass

# ============================================================================
# 启动日志与环境配置
# ============================================================================

def _boot_log(msg: str) -> None:
    try:
        os.write(2, (msg + "\n").encode("utf-8", errors="ignore"))
    except Exception:
        try:
            sys.stderr.write(msg + "\n")
            sys.stderr.flush()
        except Exception:
            pass

if __name__ == "__main__" or os.environ.get("PVP_DACER_BOOT_LOG", "0") == "1":
    _boot_log(f"[boot] pid={os.getpid()} starting @ {time.strftime('%Y-%m-%d %H:%M:%S')}")

try:
    import faulthandler
    faulthandler.enable(all_threads=True)
    try:
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    except Exception:
        pass
except Exception:
    pass

os.environ.setdefault("JAX_ENABLE_X64", "false")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "float32")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

for _deprecated_env_attr in (
    "config",
    "allowed_scenarios",
    "allowed_seeds",
    "scenario_seeds",
    "sequential_seed_list",
    "sequential_reset_seeds",
    "set_sequential_reset",
    "last_reset_info",
):
    warnings.filterwarnings(
        "ignore",
        message=rf"WARN: env\.{_deprecated_env_attr}\b.*deprecated",
        category=UserWarning,
        module=r"gymnasium\.core",
    )

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)


def require_jax_gpu_ready() -> None:
    try:
        backend = jax.default_backend()
        devices = list(jax.devices())
    except RuntimeError as e:
        raise RuntimeError(
            "JAX GPU backend is required for this project, but backend initialization failed. "
            "Check CUDA_VISIBLE_DEVICES, driver/CUDA compatibility, and the installed jaxlib build."
        ) from e
    if backend != "gpu":
        raise RuntimeError(
            f"JAX GPU backend is required for this project, but active backend is {backend!r}."
        )
    _boot_log(f"[boot] JAX backend={backend} devices={devices}")

_SCRIPT_DIR = os.path.dirname(__file__)
try:
    from _local_bootstrap import ensure_project_imports
except ModuleNotFoundError:
    from scripts._local_bootstrap import ensure_project_imports
_PROJECT_ROOT = str(ensure_project_imports(__file__))

from relax.algorithm.pvp_dacer import PVPDACER
from relax.trainer.pvp_off_policy import PVPOffPolicyTrainer
from relax.buffer.pvp_replay_buffer import PVPBalancedDualBuffer
from relax.network.dacer import create_dacer_net
from relax.utils.shared_control_monitor import SharedControlMonitor
from relax.utils.experience import Experience
from scripts.demo_data_manager_fix import DemoDataManager
from relax_env.human_in_the_loop_env import HumanInTheLoopEnv
from relax.utils.random_utils import seeding
from map_splits import TRAIN_MAP_SEEDS

JAX_UPDATE_LOCK = threading.Lock()

FIXED_STAGE_MAP_SEEDS = list(TRAIN_MAP_SEEDS)


def get_fixed_stage_map_seeds() -> List[int]:
    return list(FIXED_STAGE_MAP_SEEDS)

if __name__ == "__main__" or os.environ.get("PVP_DACER_BOOT_LOG", "0") == "1":
    _boot_log("[boot] import finished")

# ============================================================================
# PVPDACER 方法可用性检测
# ============================================================================


def _has_bc_only_update(algorithm) -> bool:
    return hasattr(algorithm, "bc_only_update")


def _has_critic_only_update(algorithm) -> bool:
    return hasattr(algorithm, "update_critic_only")


def _has_set_lambda_bc(algorithm) -> bool:
    return hasattr(algorithm, "set_lambda_bc")


def _has_set_lambda_rl(algorithm) -> bool:
    return hasattr(algorithm, "set_lambda_rl")


def _detect_pvpdacer_methods(algorithm) -> None:
    print(f"\n{'=' * 55}")
    print("🔍 PVPDACER 方法检测：")
    print(f"   bc_only_update:     {'✅' if _has_bc_only_update(algorithm) else '❌'}")
    print(f"   update_critic_only: {'✅' if _has_critic_only_update(algorithm) else '⚠️  不存在'}")
    print(f"   set_lambda_bc:      {'✅' if _has_set_lambda_bc(algorithm) else '⚠️  不存在'}")
    has_opt = hasattr(algorithm, "optim")
    has_policy_opt = hasattr(algorithm, "policy_optim")
    has_alpha_opt = hasattr(algorithm, "alpha_optim")
    print(f"   optim/policy/alpha: "
          f"{'✅' if has_opt else '⚠️ '} / "
          f"{'✅' if has_policy_opt else '⚠️ '} / "
          f"{'✅' if has_alpha_opt else '⚠️ '}")
    if not (has_opt and has_policy_opt):
        print("   ⚠️  BC-Boost 重置 opt_state 可能受限（将退化为不重置优化器状态）")
    print(f"{'=' * 55}\n")


def _bc_update(algorithm, key, batch) -> Dict[str, Any]:
    if _has_bc_only_update(algorithm):
        return algorithm.bc_only_update(key, batch)
    return algorithm.update(key, batch)


def _critic_update(algorithm, key, batch) -> Optional[Dict[str, Any]]:
    if _has_critic_only_update(algorithm):
        return algorithm.update_critic_only(key, batch)
    return None


# ============================================================================
# TensorBoard Logger
# ============================================================================

def _create_tb_logger(log_path: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(log_path / "tb"))
        print(f"   📊 TensorBoard: {log_path / 'tb'}")
        return writer
    except ImportError:
        pass
    try:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(logdir=str(log_path / "tb"))
        print(f"   📊 TensorBoard (tensorboardX): {log_path / 'tb'}")
        return writer
    except ImportError:
        pass
    print("   ⚠️  TensorBoard 未安装，日志禁用")
    return None


class TBLoggerAdapter:
    def __init__(self, writer):
        self._writer = writer
        self._closed = False

    def add_scalar(self, tag: str, value, step: int):
        if self._writer is not None and not self._closed:
            try:
                self._writer.add_scalar(tag, float(value), global_step=int(step))
            except Exception:
                pass

    def add_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        if self._writer is None or self._closed:
            return
        try:
            if hasattr(self._writer, "add_scalars"):
                self._writer.add_scalars(main_tag, tag_scalar_dict, global_step=int(step))
            else:
                for sub_tag, sub_value in tag_scalar_dict.items():
                    self.add_scalar(f"{main_tag}/{sub_tag}", sub_value, step)
        except Exception:
            pass

    def add_histogram(self, tag: str, values, step: int):
        if self._writer is None or self._closed:
            return
        try:
            self._writer.add_histogram(tag, values, global_step=int(step))
        except Exception:
            pass

    def flush(self):
        if self._writer is not None and not self._closed and hasattr(self._writer, "flush"):
            try:
                self._writer.flush()
            except Exception:
                pass

    def close(self):
        if self._closed:
            return
        if self._writer is not None and hasattr(self._writer, "close"):
            try:
                self._writer.close()
            except Exception:
                pass
        self._closed = True


def _metric_to_float(value) -> Optional[float]:
    try:
        arr = np.asarray(value)
        if arr.size == 0:
            return None
        scalar = float(arr) if arr.size == 1 else float(arr.mean())
        if not np.isfinite(scalar):
            return None
        return scalar
    except Exception:
        return None


def _host_metric_dict(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    host_metrics: Dict[str, float] = {}
    if not metrics:
        return host_metrics
    for tag, value in metrics.items():
        scalar = _metric_to_float(value)
        if scalar is not None:
            host_metrics[tag] = scalar
    return host_metrics


def _log_metric_dict(logger, metrics: Optional[Dict[str, Any]], step: int, prefix: Optional[str] = None) -> None:
    if logger is None or not metrics:
        return
    for tag, value in _host_metric_dict(metrics).items():
        final_tag = f"{prefix}/{tag}" if prefix else tag
        logger.add_scalar(final_tag, value, step)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class BCBoostConfig:
    burst_steps: int = 20
    burst_batch_size: int = 64
    recent_steps: int = 200
    lambda_bc: Optional[float] = None
    lr: Optional[float] = None
    ema_alpha: float = 0.2
    trigger_every: int = 200


@dataclass
class Stage1bConfig:
    total_bc_updates: int = 100_000
    update_batch_size: int = 64
    update_interval: int = 500
    bc_lr: float = 1e-4
    lambda_pv: float = 0.0
    lambda_bc: float = 50.0


# ============================================================================
# 工具函数
# ============================================================================

def clear_jax_cache():
    try:
        gc.collect()
        if hasattr(jax, "devices"):
            for device in jax.devices():
                if device.platform in ("cuda", "gpu"):
                    try:
                        from jax._src import xla_bridge
                        xla_bridge.clear_backend_cache()
                    except Exception:
                        pass
    except Exception:
        pass


def _tree_l2_norm(tree) -> float:
    leaves = jax.tree_util.tree_leaves(tree)
    total = sum(float(np.sum(np.asarray(jax.device_get(x), dtype=np.float32) ** 2)) for x in leaves)
    return float(np.sqrt(total))


def _tree_l2_diff(tree_a, tree_b) -> float:
    leaves_a = jax.tree_util.tree_leaves(tree_a)
    leaves_b = jax.tree_util.tree_leaves(tree_b)
    total = 0.0
    for a, b in zip(leaves_a, leaves_b):
        aa = np.asarray(jax.device_get(a), dtype=np.float32)
        bb = np.asarray(jax.device_get(b), dtype=np.float32)
        total += float(np.sum((aa - bb) ** 2))
    return float(np.sqrt(total))


def _policy_param_norm(algorithm) -> float:
    return _tree_l2_norm(_get_policy_only_params(algorithm))


def _policy_param_diff(a, b) -> float:
    return _tree_l2_diff(_get_policy_only_params(a), _get_policy_only_params(b))


def _get_policy_only_params(algorithm):
    params = algorithm.state.params
    if hasattr(params, "policy"):
        return params.policy
    if isinstance(params, dict) and "policy" in params:
        return params["policy"]
    return params


def _get_policy_only_params_from_params(params):
    if hasattr(params, "policy"):
        return params.policy
    if isinstance(params, dict) and "policy" in params:
        return params["policy"]
    return params


def _mix_tree_params(current_tree, boosted_tree, mix_alpha: float):
    alpha = float(np.clip(mix_alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return current_tree
    if alpha >= 1.0:
        return boosted_tree
    return jax.tree_util.tree_map(
        lambda cur, boosted: (1.0 - alpha) * cur + alpha * boosted,
        current_tree,
        boosted_tree,
    )


def _safe_get_deterministic_action(agent, full_params, obs):
    """
    [FIX-1] 原实现只抓 TypeError，实际调用失败可能抛 ValueError/AttributeError/KeyError，
    会让第一个 attempt 直接异常退出而跳过后续 fallback。改为捕获 Exception。
    """
    policy_only = _get_policy_only_params_from_params(full_params)
    log_alpha = None
    if hasattr(full_params, "log_alpha"):
        log_alpha = full_params.log_alpha
    elif isinstance(full_params, dict) and "log_alpha" in full_params:
        log_alpha = full_params["log_alpha"]

    attempts = []
    if log_alpha is not None:
        attempts.append(((policy_only, log_alpha), "(policy_params, log_alpha)"))
    attempts.append((policy_only, "policy_params only"))
    attempts.append((full_params, "full_params"))

    errors = []
    for params_candidate, label in attempts:
        try:
            return agent.get_deterministic_action(params_candidate, obs)
        except Exception as exc:  # [FIX-1] 原为 TypeError
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
            continue

    raise RuntimeError(
        "Unable to call agent.get_deterministic_action with any supported parameter layout. "
        + " | ".join(errors)
    )


def _get_current_info(env) -> Dict[str, Any]:
    e = env
    for _ in range(5):
        current_info = _safe_local_attr(e, "current_info", _MISSING)
        if current_info is not _MISSING:
            return current_info
        if hasattr(e, "env"):
            e = e.env
        else:
            break
    return {}


def _extract_seed_from_info(info: Optional[Dict[str, Any]]) -> Optional[int]:
    if not isinstance(info, dict):
        return None
    for k in ("current_seed", "seed", "scenario_seed", "map_seed"):
        v = info.get(k, None)
        if v is not None:
            try:
                return int(v)
            except Exception:
                pass
    return None


_MISSING = object()


def _safe_local_attr(obj, name: str, default=_MISSING):
    try:
        return object.__getattribute__(obj, name)
    except AttributeError:
        if default is _MISSING:
            raise
        return default


def _safe_config_dict(obj) -> Optional[dict]:
    cfg = _safe_local_attr(obj, "config", None)
    return cfg if isinstance(cfg, dict) else None


def _get_running_map_seed(env, fallback: Optional[int] = None) -> Optional[int]:
    candidates = []

    try:
        hitl_env = _find_hitl_env(env)
        candidates.append(hitl_env)
    except Exception:
        pass

    try:
        base_env = _unwrap_env(env)
        candidates.append(base_env)
    except Exception:
        pass

    candidates.append(env)

    seen = set()
    for obj in candidates:
        if obj is None or id(obj) in seen:
            continue
        seen.add(id(obj))

        try:
            info = _safe_local_attr(obj, "last_reset_info", None)
            seed = _extract_seed_from_info(info)
            if seed is not None:
                return seed
        except Exception:
            pass

        try:
            seed = _safe_local_attr(obj, "current_seed", _MISSING)
            if seed is not _MISSING and seed is not None:
                return int(seed)
        except Exception:
            pass

        try:
            info = _safe_local_attr(obj, "current_info", None)
            seed = _extract_seed_from_info(info)
            if seed is not None:
                return seed
        except Exception:
            pass

    return fallback


def _merge_policy_params(current_params, boosted_params, mix_alpha: float = 1.0):
    blended_policy = _mix_tree_params(
        _get_policy_only_params_from_params(current_params),
        _get_policy_only_params_from_params(boosted_params),
        mix_alpha,
    )

    if hasattr(current_params, "_replace"):
        kwargs = {}
        if hasattr(current_params, "policy"):
            kwargs["policy"] = blended_policy
        return current_params._replace(**kwargs) if kwargs else current_params

    try:
        from flax.core import FrozenDict
        is_frozen = isinstance(current_params, FrozenDict)
    except Exception:
        is_frozen = False

    if isinstance(current_params, dict) or is_frozen:
        cur = dict(current_params)
        if "policy" in cur:
            cur["policy"] = blended_policy
        if is_frozen:
            from flax.core import freeze
            return freeze(cur)
        return cur

    return blended_policy


def _maybe_sync_target_policy_params(target_params, source_params):
    """
    若 target_params 也包含 policy 分支，则把 policy 同步到与主 params 一致。
    这样 BC-Boost / 参数混合后不会出现 main/target 的 policy 长时间不一致。
    若 target 结构里没有 policy，则保持原样。
    """
    try:
        if target_params is None:
            return target_params
        if not _tree_has_key(target_params, "policy"):
            return target_params
        source_policy = _get_policy_only_params_from_params(source_params)
        return _tree_set_key(target_params, "policy", source_policy)
    except Exception as e:
        print(f"   ⚠️  _maybe_sync_target_policy_params 失败: {e}")
        return target_params


def _state_replace(state, **kwargs):
    if hasattr(state, "_replace"):
        return state._replace(**kwargs)
    if hasattr(state, "replace"):
        return state.replace(**kwargs)
    try:
        for k, v in kwargs.items():
            setattr(state, k, v)
        return state
    except Exception as e:
        raise TypeError(f"_state_replace failed for {type(state).__name__}: {e}") from e


def _tree_has_key(tree, key: str) -> bool:
    try:
        if hasattr(tree, key):
            return True
    except Exception:
        pass
    try:
        return isinstance(tree, dict) and (key in tree)
    except Exception:
        return False


def _tree_get_key(tree, key: str, default=None):
    try:
        if hasattr(tree, key):
            return getattr(tree, key)
    except Exception:
        pass
    try:
        if isinstance(tree, dict):
            return tree.get(key, default)
    except Exception:
        pass
    return default


def _tree_set_key(tree, key: str, value):
    if hasattr(tree, "_replace"):
        try:
            return tree._replace(**{key: value})
        except Exception:
            pass

    try:
        from flax.core import FrozenDict
        is_frozen = isinstance(tree, FrozenDict)
    except Exception:
        FrozenDict = None
        is_frozen = False

    if isinstance(tree, dict) or is_frozen:
        new_tree = dict(tree)
        new_tree[key] = value
        if is_frozen:
            from flax.core import freeze
            return freeze(new_tree)
        return new_tree

    try:
        setattr(tree, key, value)
        return tree
    except Exception as e:
        raise TypeError(f"_tree_set_key failed for {type(tree).__name__}.{key}: {e}") from e


def _try_reset_algorithm_internal_step(algorithm, to_step: int = 0) -> bool:
    """
    [FIX-3] warmup 后仅重置日志计数是不够的，算法内部 step 必须同步归零，
    否则 actor_delay / target_update_delay / delay_alpha_update 会从非零值开始计数，
    导致 warmup 结束后第一次正式更新的触发时机与预期不符。
    返回是否成功重置。
    """
    state = getattr(algorithm, "state", None)
    if state is None:
        return False
    if not (hasattr(state, "step") or (isinstance(state, dict) and "step" in state)):
        return False
    try:
        algorithm.state = _state_replace(state, step=jnp.asarray(int(to_step), dtype=jnp.int32))
        return True
    except Exception:
        try:
            algorithm.state = _state_replace(state, step=int(to_step))
            return True
        except Exception as e:
            print(f"   ⚠️  _try_reset_algorithm_internal_step 失败: {e}")
            return False


def create_iter_key_fn(master_key, sample_per_iteration: int, update_per_iteration: int):
    def iter_key_fn(step):
        key = jax.random.fold_in(master_key, int(step))
        keys = jax.random.split(key, sample_per_iteration + update_per_iteration)
        return keys[:sample_per_iteration], keys[sample_per_iteration:]
    return iter_key_fn


def _get_action_dim_from_trainer(trainer, fallback: Optional[int] = None) -> int:
    env_action_space = getattr(getattr(trainer, "env", None), "action_space", None)
    if env_action_space is not None and getattr(env_action_space, "shape", None):
        return int(env_action_space.shape[0])
    if fallback is not None:
        return int(fallback)
    raise RuntimeError("无法从 trainer.env.action_space 推断 action_dim，请检查 env wrapper 是否屏蔽了 action_space")


def _configure_allowed_seeds(env, allowed_seeds: List[int]) -> None:
    base_env = _find_hitl_env(env)
    try:
        cfg = _safe_config_dict(base_env)
        if cfg is not None:
            cfg["allowed_scenarios"] = list(allowed_seeds)
    except Exception:
        pass


def _reset_with_fixed_seed(env, allowed_seeds, rng=None):
    if rng is None:
        seed = int(np.random.choice(allowed_seeds))
    else:
        seed = int(rng.choice(allowed_seeds))

    reset_used_seed = False
    try:
        obs, _ = env.reset(seed=seed)
        reset_used_seed = True
    except TypeError:
        obs, _ = env.reset()

    if reset_used_seed:
        base_env = _find_hitl_env(env)
        try:
            if _safe_local_attr(base_env, "current_seed", _MISSING) is not _MISSING:
                base_env.current_seed = seed
        except Exception:
            pass

    return obs


def configure_fixed_stage_maps(env) -> List[int]:
    allowed_seeds = get_fixed_stage_map_seeds()
    _configure_allowed_seeds(env, allowed_seeds)
    return allowed_seeds


def reset_to_fixed_stage_map(env, rng=None):
    allowed_seeds = configure_fixed_stage_maps(env)
    return _reset_with_fixed_seed(env, allowed_seeds, rng=rng)


def reset_to_strict_stage_map(env, seed: int):
    allowed_seeds = configure_fixed_stage_maps(env)
    seed = int(seed)
    if seed not in allowed_seeds:
        raise ValueError(f"seed={seed} 不在固定地图集合 {allowed_seeds} 中")

    _apply_fixed_seed_config(env, allowed_seeds)
    _disable_env_sequential_reset(env)

    obs, _info = env.reset(seed=seed)
    _sync_env_current_seed(env, seed)
    return obs


def _iter_candidate_envs(env):
    candidates = []
    try:
        candidates.append(env)
    except Exception:
        pass
    try:
        candidates.append(_find_hitl_env(env))
    except Exception:
        pass
    try:
        candidates.append(_unwrap_env(env))
    except Exception:
        pass

    seen = set()
    uniq = []
    for obj in candidates:
        if obj is None:
            continue
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        uniq.append(obj)
    return uniq


def _disable_env_sequential_reset(env) -> None:
    for obj in _iter_candidate_envs(env):
        try:
            setter = _safe_local_attr(obj, "set_sequential_reset", None)
            if callable(setter):
                setter(False)
        except Exception:
            pass
        try:
            if _safe_local_attr(obj, "sequential_reset", _MISSING) is not _MISSING:
                obj.sequential_reset = False
        except Exception:
            pass
        try:
            cfg = _safe_config_dict(obj)
            if cfg is not None:
                cfg["sequential_reset"] = False
        except Exception:
            pass


def _apply_fixed_seed_config(env, allowed_seeds: List[int]) -> None:
    allowed_seeds = [int(x) for x in allowed_seeds]
    for obj in _iter_candidate_envs(env):
        try:
            cfg = _safe_config_dict(obj)
            if cfg is not None:
                cfg["allowed_scenarios"] = list(allowed_seeds)
        except Exception:
            pass

        for attr in [
            "allowed_scenarios",
            "allowed_seeds",
            "scenario_seeds",
            "sequential_seed_list",
            "sequential_reset_seeds",
            "_sequential_reset_seeds",
        ]:
            try:
                if _safe_local_attr(obj, attr, _MISSING) is not _MISSING:
                    setattr(obj, attr, list(allowed_seeds))
            except Exception:
                pass


def _sync_env_current_seed(env, seed: int) -> None:
    for obj in _iter_candidate_envs(env):
        try:
            if _safe_local_attr(obj, "current_seed", _MISSING) is not _MISSING:
                obj.current_seed = int(seed)
        except Exception:
            pass
        try:
            info = _safe_local_attr(obj, "last_reset_info", _MISSING)
            if info is not _MISSING:
                if not isinstance(info, dict):
                    info = {}
                    obj.last_reset_info = info
                if isinstance(info, dict):
                    info["current_seed"] = int(seed)
                    info["seed"] = int(seed)
                    info["scenario_seed"] = int(seed)
                    info["map_seed"] = int(seed)
        except Exception:
            pass


def _patch_single_env_reset_strict(obj, state: dict) -> None:
    if obj is None:
        return

    obj._strict_fixed_seed_state = state

    if getattr(obj, "_strict_fixed_seed_patched", False):
        return

    original_reset = obj.reset

    def wrapped_reset(*args, **kwargs):
        st = obj._strict_fixed_seed_state
        allowed = list(st["allowed_seeds"])
        mode = st["mode"]

        req_seed = kwargs.get("seed", None)
        use_seed = None

        if req_seed is not None:
            try:
                req_seed = int(req_seed)
            except Exception:
                req_seed = None

        if req_seed in allowed:
            use_seed = int(req_seed)
            if mode == "cycle":
                st["cursor"] = (allowed.index(use_seed) + 1) % len(allowed)
        else:
            if mode == "cycle":
                idx = int(st["cursor"]) % len(allowed)
                use_seed = int(allowed[idx])
                st["cursor"] = (idx + 1) % len(allowed)
            elif mode == "random":
                rng = st.get("rng", None)
                if rng is None:
                    rng = np.random.default_rng(0)
                    st["rng"] = rng
                use_seed = int(rng.choice(allowed))
            else:
                use_seed = int(allowed[0])

        kwargs["seed"] = int(use_seed)

        _apply_fixed_seed_config(obj, allowed)
        _disable_env_sequential_reset(obj)

        out = original_reset(*args, **kwargs)

        try:
            _sync_env_current_seed(obj, use_seed)
        except Exception:
            pass

        return out

    obj.reset = wrapped_reset
    obj._strict_fixed_seed_patched = True
    obj._strict_fixed_seed_original_reset = original_reset


def install_strict_fixed_seed_controller(
    env,
    allowed_seeds: List[int],
    mode: str = "cycle",
    start_index: int = 0,
    rng_seed: int = 0,
):
    allowed = [int(x) for x in allowed_seeds]
    if len(allowed) == 0:
        raise ValueError("allowed_seeds 不能为空")

    state = {
        "allowed_seeds": list(allowed),
        "mode": str(mode),
        "cursor": int(start_index) % len(allowed),
        "rng": np.random.default_rng(int(rng_seed)),
    }

    _apply_fixed_seed_config(env, allowed)
    _disable_env_sequential_reset(env)

    for obj in _iter_candidate_envs(env):
        _patch_single_env_reset_strict(obj, state)

    return state


# ============================================================================
# 数据加载
# ============================================================================

EXCLUDED_PKL_FILENAMES = {"stage1a_merged_demo.pkl", "integrated_demo_data.pkl"}
_DEMO_PKL_PATTERNS = (
    re.compile(r"^seed\d+.*\.pkl$"),
    re.compile(r"^stage2_seed\d+.*\.pkl$"),
)


def _is_demo_pkl_file(path: Path) -> bool:
    if path.suffix.lower() != ".pkl":
        return False
    if path.name in EXCLUDED_PKL_FILENAMES:
        return False
    return any(pat.match(path.name) for pat in _DEMO_PKL_PATTERNS)


def _collect_demo_pkl_files(root: Path) -> List[Path]:
    return sorted(p for p in root.rglob("*.pkl") if _is_demo_pkl_file(p))


def _is_stage1a_demo_file(path: Path) -> bool:
    return bool(re.match(r"^seed\d+.*\.pkl$", path.name))


def _assert_same_length(source: str, **arrays: Any) -> int:
    lengths = {}
    for name, value in arrays.items():
        if value is None:
            continue
        try:
            lengths[name] = len(value)
        except TypeError:
            continue

    if not lengths:
        return 0

    first_len = next(iter(lengths.values()))
    mismatched = {k: v for k, v in lengths.items() if v != first_len}
    if mismatched:
        detail = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(f"[{source}] 字段长度不一致: {detail}")
    return first_len


def _reconstruct_pre_takeover(
    inter_arr: np.ndarray,
    takeover_start_arr: np.ndarray,
    window: int = 20,
) -> np.ndarray:
    n = len(inter_arr)
    pt_arr = np.zeros(n, dtype=np.float32)
    ts_indices = np.where(takeover_start_arr > 0.5)[0]
    for idx in ts_indices:
        start = max(0, idx - window)
        pt_arr[start:idx][inter_arr[start:idx] < 0.5] = 1.0
    return pt_arr


def _pick_sequence_key(data: Dict[str, Any], *candidates: str) -> Optional[str]:
    for key in candidates:
        if key not in data:
            continue
        value = data[key]
        try:
            if len(value) == 0:
                continue
        except TypeError:
            pass
        return key
    return None


def _build_next_obs_sequence(obs_seq, done_arr) -> List[Any]:
    done_1d = np.asarray(done_arr, dtype=np.float32).reshape(-1)
    num_steps = len(obs_seq)
    return [
        obs_seq[i + 1] if (i + 1 < num_steps and done_1d[i] < 0.5) else obs_seq[i]
        for i in range(num_steps)
    ]


def _resolve_pre_takeover_window(trainer, default: int = 0) -> int:
    try:
        return max(0, int(getattr(trainer, "pre_takeover_window", default)))
    except Exception:
        return int(default)


def _infer_demo_flags_from_payload(
    data: Dict[str, Any],
    num_steps: int,
    *,
    source_path: Optional[Path] = None,
) -> np.ndarray:
    if "is_demo" in data:
        return np.asarray(data["is_demo"], dtype=np.float32)

    if source_path is not None and _is_stage1a_demo_file(source_path):
        return np.ones(num_steps, dtype=np.float32)

    return np.zeros(num_steps, dtype=np.float32)


def _has_novice_action(data: Dict[str, Any]) -> bool:
    return _pick_sequence_key(data, "action_novice", "action_agent", "agent_action") is not None


def _resolve_action_novice(
    data: Dict[str, Any],
    action_behavior: Any,
    *,
    source: str,
    allow_missing: bool = False,
) -> Any:
    for key in ("action_novice", "action_agent", "agent_action"):
        value = data.get(key, None)
        if value is not None:
            return value

    msg = (
        f"[{source}] 缺少 action_novice/action_agent/agent_action。"
        "不能静默用 zero action 作为 novice counterfactual；"
        "这会污染 PV 和 EnergyRank 的 pair-level 监督。"
    )
    if not allow_missing:
        raise KeyError(msg + " 如需兼容旧数据，请显式传 --allow_missing_novice_action。")

    warnings.warn(
        msg + " 已按 --allow_missing_novice_action 回退为 action_behavior；该样本不会提供有效 novice counterfactual。",
        RuntimeWarning,
        stacklevel=2,
    )
    return np.asarray(action_behavior, dtype=np.float32).copy()


def _load_single_pkl_to_trainer(demo_file: str, trainer) -> int:
    # [FIX-6] reward 被强制置 0，仅在 reward_free 模式下合理。其他模式必须显式失败。
    if not getattr(getattr(trainer, "algorithm", None), "reward_free", True):
        raise RuntimeError(
            "当前 loader 强制 reward=0.0，但 algorithm.reward_free=False。"
            "请切换 reward_free=True 或扩展 loader 支持真实 reward。"
        )

    try:
        with open(demo_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"   ❌ 无法读取 {os.path.basename(demo_file)}: {e}")
        return 0

    if "raw_data" in data and "observation" not in data:
        data = data["raw_data"]

    obs_key = _pick_sequence_key(data, "observation", "obs")
    if obs_key is None:
        return 0

    num_steps = len(data[obs_key])
    if num_steps == 0:
        return 0

    action_dim = _get_action_dim_from_trainer(trainer)

    default_actions = np.zeros((num_steps, action_dim), dtype=np.float32)

    raw_inter = data.get("intervention", None)
    if raw_inter is None:
        inter_arr = np.ones(num_steps, dtype=np.float32)
    else:
        inter_arr = np.asarray(raw_inter, dtype=np.float32)

    stop_arr = np.asarray(
        data.get("stop_td", np.zeros(num_steps, dtype=np.float32)), dtype=np.float32,
    )

    raw_ts = data.get("takeover_start", None)
    if raw_ts is not None:
        ts_arr = np.asarray(raw_ts, dtype=np.float32)
    else:
        ts_arr = np.zeros(num_steps, dtype=np.float32)
        for _k in range(1, num_steps):
            if inter_arr[_k] > 0.5 and inter_arr[_k - 1] < 0.5:
                ts_arr[_k] = 1.0

    demo_arr = _infer_demo_flags_from_payload(data, num_steps, source_path=Path(demo_file))

    if "is_pre_takeover" in data:
        pt_arr = np.asarray(data["is_pre_takeover"], dtype=np.float32)
    else:
        pt_arr = _reconstruct_pre_takeover(
            inter_arr,
            ts_arr,
            window=_resolve_pre_takeover_window(trainer),
        )

    action_behavior = data.get("action_behavior", data.get("action", default_actions))
    novice_pair_ok_default = 1.0 if _has_novice_action(data) else 0.0
    action_novice = _resolve_action_novice(
        data,
        action_behavior,
        source=f"demo file {os.path.basename(demo_file)}",
        allow_missing=bool(getattr(trainer, "allow_missing_novice_action", False)),
    )
    action_human = data.get("action_human", data.get("human_action", action_behavior))

    done_raw = data.get("done", np.zeros(num_steps, dtype=np.float32))
    terminated = data.get("terminated", done_raw)
    truncated = data.get("truncated", np.zeros_like(done_raw))
    done_arr = np.asarray(np.logical_or(done_raw, np.logical_or(terminated, truncated)), dtype=np.float32)

    next_obs_key = _pick_sequence_key(data, "next_observation", "next_obs")
    if next_obs_key is not None:
        next_obs_seq = data[next_obs_key]
    else:
        obs_seq = data[obs_key]
        next_obs_seq = _build_next_obs_sequence(obs_seq, done_arr)

    takeover_end = data.get("takeover_end", np.zeros(num_steps, dtype=np.float32))
    takeover_end = np.asarray(takeover_end, dtype=np.float32)

    _assert_same_length(
        f"demo file {os.path.basename(demo_file)}",
        observation=data[obs_key],
        action_behavior=action_behavior,
        action_novice=action_novice,
        action_human=action_human,
        done=done_arr,
        next_observation=next_obs_seq,
        intervention=inter_arr,
        stop_td=stop_arr,
        takeover_start=ts_arr,
        pre_takeover=pt_arr,
        is_demo=demo_arr,
        takeover_end=takeover_end,
    )

    loaded = 0
    for i in range(num_steps):
        obs = data[obs_key][i]
        action_beh_step = np.asarray(action_behavior[i], dtype=np.float32)
        action_novice_step = np.asarray(action_novice[i], dtype=np.float32)
        action_human_step = np.asarray(action_human[i], dtype=np.float32)
        pair_ok_flag = novice_pair_ok_default
        reward_val = 0.0  # reward_free 假设；入口已断言
        done_flag = bool(done_arr[i])
        next_obs = next_obs_seq[i]
        is_demo_flag = float(demo_arr[i] > 0.5)
        intervention = 0.0 if is_demo_flag > 0.5 else float(inter_arr[i])

        stop_td_val = 1.0 if is_demo_flag > 0.5 else float(stop_arr[i])
        if is_demo_flag <= 0.5 and stop_td_val < 0.5:
            takeover_start_flag = float(ts_arr[i] > 0.5)
            takeover_end_flag = float(takeover_end[i] > 0.5)
            if not takeover_end_flag and i + 1 < num_steps:
                if inter_arr[i] > 0.5 and inter_arr[i + 1] < 0.5:
                    takeover_end_flag = 1.0
            if takeover_start_flag > 0.5 or takeover_end_flag > 0.5:
                stop_td_val = 1.0
        pre_takeover_flag = 0.0 if is_demo_flag > 0.5 else float(pt_arr[i])
        if getattr(trainer, "disable_stop_td_mask", False):
            stop_td_val = 0.0
        if getattr(trainer, "merge_action_semantics", False):
            action_novice_step = action_beh_step.copy()
            action_human_step = action_beh_step.copy()
            pair_ok_flag = 0.0
        if getattr(trainer, "pv_on_all_expert_data", False) and (
            intervention > 0.5 or pre_takeover_flag > 0.5 or is_demo_flag > 0.5
        ):
            intervention = 1.0
            pre_takeover_flag = 0.0
            is_demo_flag = 0.0

        exp = Experience.create_pvp_experience(
            obs=obs,
            a_novice=action_novice_step,
            a_human=action_human_step,
            a_behavior=action_beh_step,
            reward=reward_val,
            next_obs=next_obs,
            done=done_flag,
            intervention=intervention,
            stop_td=stop_td_val,
            is_pre_takeover=pre_takeover_flag,
            is_demo=is_demo_flag,
            pair_ok=pair_ok_flag,
        )
        trainer.buffer.add(exp)
        loaded += 1

    return loaded


def load_demo_data_directory(demo_dir: str, trainer) -> bool:
    demo_dir_path = Path(demo_dir)
    demo_files = [str(p) for p in _collect_demo_pkl_files(demo_dir_path)]

    if not demo_files:
        print(f"❌ 在 {demo_dir} 中未找到合法演示文件（仅接受 seed*.pkl / stage2_seed*.pkl）")
        return False

    print(f"🔄 正在加载 {len(demo_files)} 个演示文件...")
    total_experiences = 0
    for demo_file in demo_files:
        try:
            n = _load_single_pkl_to_trainer(demo_file, trainer)
            if n > 0:
                print(f"   ✅ {os.path.basename(demo_file)}: {n} 条")
                total_experiences += n
            else:
                print(f"   ⚠️  {os.path.basename(demo_file)}: 0 条（跳过）")
        except Exception as e:
            print(f"   ❌ {os.path.basename(demo_file)}: {e}")

    print(f"✅ 加载完成：{total_experiences} 条 | expert={len(trainer.buffer.human_buffer)} | novice={len(trainer.buffer.novice_buffer)}")

    if len(trainer.buffer.human_buffer) == 0:
        print("❌ expert_buffer 为空！演示数据可能 intervention 全为 0。")
        return False
    return total_experiences > 0


def load_all_demos_recursive(root_path: str, trainer) -> bool:
    root = Path(root_path)
    if not root.exists():
        print(f"❌ demo_root 路径不存在: {root}")
        return False
    if not root.is_dir():
        print(f"❌ demo_root 不是目录: {root}")
        return False

    filtered_files = _collect_demo_pkl_files(root)

    if not filtered_files:
        print(f"❌ 在 {root} 及其子目录中未找到合法演示文件（仅接受 seed*.pkl / stage2_seed*.pkl）")
        return False

    subdirs = set()
    for p in filtered_files:
        rel = p.relative_to(root)
        if len(rel.parts) > 1:
            subdirs.add(str(rel.parent))

    print(f"\n{'=' * 60}")
    print(f"📂 递归加载演示数据: {root}")
    print(f"   找到 {len(filtered_files)} 个合法演示文件")
    if subdirs:
        print(f"   分布在 {len(subdirs)} 个子目录中:")
        for sd in sorted(subdirs):
            count = sum(1 for p in filtered_files if str(p.relative_to(root).parent) == sd)
            print(f"      📁 {sd}: {count} 个文件")
    else:
        print(f"   全部位于根目录下")
    print(f"{'=' * 60}")

    total_experiences = 0
    success_count = 0
    fail_count = 0

    for pkl_path in filtered_files:
        try:
            n = _load_single_pkl_to_trainer(str(pkl_path), trainer)
            if n > 0:
                rel_path = pkl_path.relative_to(root)
                print(f"   ✅ {rel_path}: {n} 条")
                total_experiences += n
                success_count += 1
            else:
                print(f"   ⚠️  {pkl_path.relative_to(root)}: 0 条（跳过）")
                fail_count += 1
        except Exception as e:
            print(f"   ❌ {pkl_path.relative_to(root)}: {e}")
            fail_count += 1

    print(f"✅ 递归加载完成：{total_experiences} 条 | success={success_count} | fail={fail_count}")
    print(f"   expert={len(trainer.buffer.human_buffer)} | novice={len(trainer.buffer.novice_buffer)}")

    if len(trainer.buffer.human_buffer) == 0:
        print("❌ expert_buffer 为空！演示数据可能 intervention 全为 0。")
        return False
    return total_experiences > 0


def load_integrated_demo_data(demo_file: str, trainer) -> bool:
    try:
        with open(demo_file, "rb") as f:
            raw_data = pickle.load(f)

        if "raw_data" in raw_data or all(
            k in raw_data for k in ["observation", "action_human", "action_behavior", "reward", "done"]
        ):
            return _load_shared_control_data(raw_data, trainer)

        if all(k in raw_data for k in ["obs", "action_human", "action_behavior", "reward", "done"]):
            return _load_shared_control_data(raw_data, trainer)

        manager = DemoDataManager()
        demo_data = manager.load_demo_data(demo_file)
        if demo_data is not None and manager.load_demo_data_to_trainer(trainer, demo_data):
            if len(trainer.buffer.human_buffer) == 0:
                print("❌ 加载后 human_buffer 为空！")
                return False
            return True
        return False
    except Exception as e:
        print(f"❌ 加载异常: {e}")
        import traceback
        traceback.print_exc()
        return False


def _load_shared_control_data(raw_data: dict, trainer) -> bool:
    # [FIX-6] reward_free 断言
    if not getattr(getattr(trainer, "algorithm", None), "reward_free", True):
        print("❌ _load_shared_control_data 当前仅支持 reward_free 模式（内部强制 reward=0）")
        return False

    try:
        if "raw_data" in raw_data:
            raw_data = raw_data["raw_data"]

        obs_key = _pick_sequence_key(raw_data, "observation", "obs")
        if obs_key is None:
            raise KeyError("shared control data 缺少 observation/obs")
        obs = raw_data[obs_key]

        done = raw_data["done"]
        terminated = raw_data.get("terminated", done)
        truncated = raw_data.get("truncated", np.zeros_like(done))
        episode_done = np.logical_or(done, np.logical_or(terminated, truncated))

        next_obs_key = _pick_sequence_key(raw_data, "next_observation", "next_obs")
        if next_obs_key is not None:
            next_obs = raw_data[next_obs_key]
        else:
            next_obs = _build_next_obs_sequence(obs, episode_done)

        action_behavior = raw_data["action_behavior"]
        novice_pair_ok_default = 1.0 if _has_novice_action(raw_data) else 0.0
        action_novice = _resolve_action_novice(
            raw_data,
            action_behavior,
            source="shared control data",
            allow_missing=bool(getattr(trainer, "allow_missing_novice_action", False)),
        )

        action_human = raw_data["action_human"]
        reward = raw_data["reward"]

        raw_inter = raw_data.get("intervention", None)
        intervention = np.ones(len(obs), dtype=np.float32) if raw_inter is None else np.asarray(raw_inter, dtype=np.float32)
        is_demo = _infer_demo_flags_from_payload(raw_data, len(obs))

        takeover_start_arr = np.asarray(raw_data.get("takeover_start", np.zeros(len(obs), dtype=np.float32)), dtype=np.float32)
        takeover_end = np.asarray(raw_data.get("takeover_end", np.zeros_like(done)), dtype=np.float32)

        if "is_pre_takeover" in raw_data:
            is_pre_takeover = np.asarray(raw_data["is_pre_takeover"], dtype=np.float32)
        else:
            is_pre_takeover = _reconstruct_pre_takeover(
                intervention,
                takeover_start_arr,
                window=_resolve_pre_takeover_window(trainer),
            )

        _assert_same_length(
            "shared control data",
            obs=obs,
            next_obs=next_obs,
            action_behavior=action_behavior,
            action_novice=action_novice,
            action_human=action_human,
            reward=reward,
            done=episode_done,
            intervention=intervention,
            takeover_start=takeover_start_arr,
            takeover_end=takeover_end,
            pre_takeover=is_pre_takeover,
            is_demo=is_demo,
        )

        human_count = novice_count = 0
        for i in range(len(obs)):
            takeover_start_flag = float(takeover_start_arr[i] > 0.5)
            takeover_end_flag = float(takeover_end[i] > 0.5)
            demo_flag = float(is_demo[i] > 0.5)
            if demo_flag <= 0.5 and not takeover_end_flag and i + 1 < len(intervention):
                if intervention[i] > 0.5 and intervention[i + 1] < 0.5:
                    takeover_end_flag = 1.0
            stop_td = 1.0 if (demo_flag > 0.5 or takeover_start_flag > 0.5 or takeover_end_flag > 0.5) else 0.0
            intervention_flag = 0.0 if demo_flag > 0.5 else float(intervention[i])
            pre_takeover_flag = 0.0 if demo_flag > 0.5 else float(is_pre_takeover[i])
            action_novice_step = np.asarray(action_novice[i], dtype=np.float32)
            action_human_step = np.asarray(action_human[i], dtype=np.float32)
            action_behavior_step = np.asarray(action_behavior[i], dtype=np.float32)
            pair_ok_flag = novice_pair_ok_default
            if getattr(trainer, "disable_stop_td_mask", False):
                stop_td = 0.0
            if getattr(trainer, "merge_action_semantics", False):
                action_novice_step = action_behavior_step.copy()
                action_human_step = action_behavior_step.copy()
                pair_ok_flag = 0.0
            if getattr(trainer, "pv_on_all_expert_data", False) and (
                intervention_flag > 0.5 or pre_takeover_flag > 0.5 or demo_flag > 0.5
            ):
                intervention_flag = 1.0
                pre_takeover_flag = 0.0
                demo_flag = 0.0

            exp = Experience.create_pvp_experience(
                obs=obs[i],
                a_novice=action_novice_step,
                a_human=action_human_step,
                a_behavior=action_behavior_step,
                reward=0.0,  # reward_free
                next_obs=next_obs[i],
                done=bool(episode_done[i]),
                intervention=intervention_flag,
                stop_td=stop_td,
                is_pre_takeover=pre_takeover_flag,
                is_demo=demo_flag,
                pair_ok=pair_ok_flag,
            )
            trainer.buffer.add(exp)
            if intervention_flag > 0.5 or pre_takeover_flag > 0.5 or demo_flag > 0.5:
                human_count += 1
            else:
                novice_count += 1

        print(f"   加载完成: human={human_count} | novice={novice_count}")
        if len(trainer.buffer.human_buffer) == 0:
            print("❌ human_buffer 为空！")
            return False
        return True
    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Algorithm 构建辅助
# ============================================================================

def _extract_target_params(state):
    """[FIX-5] 若算法 state 中有 target_params（典型 SAC/DACER 结构），返回它。"""
    if state is None:
        return None
    if hasattr(state, "target_params"):
        return state.target_params
    if isinstance(state, dict) and "target_params" in state:
        return state["target_params"]
    return None


def _create_algorithm_with_params(
    original_algorithm: PVPDACER,
    lambda_pv: float, lambda_bc: float,
    lr: Optional[float] = None, B: Optional[float] = None,
    actor_lr: Optional[float] = None,
    actor_delay: Optional[int] = None,
    target_update_delay: Optional[int] = None,
    transfer_state: bool = True,
    transfer_target_params: bool = True,
    reset_opt_state: bool = False,
    reset_internal_step: bool = False,
) -> PVPDACER:
    """
    [FIX-5 相关]
    - transfer_state=True: 完整迁移 params + opt_state + step + 统计量
    - transfer_state=False: 只迁移 params，但仍可选地迁移 target_params（transfer_target_params）
    - reset_opt_state: 强制重置优化器状态（用于 lr 改变后避免旧动量误导）
    - reset_internal_step: 强制 state.step=0（用于 warmup 后的正式训练起点）
    """
    original_params = original_algorithm.state.params
    original_opt_state = original_algorithm.state.opt_state
    original_step = original_algorithm.state.step
    original_mean_q1 = getattr(original_algorithm.state, "mean_q1_std", 0.0)
    original_mean_q2 = getattr(original_algorithm.state, "mean_q2_std", 0.0)
    original_entropy = getattr(original_algorithm.state, "entropy", 0.0)
    original_target_params = _extract_target_params(original_algorithm.state)

    if lr is None:
        lr = getattr(original_algorithm, "lr", 3e-4)
    if B is None:
        B = getattr(original_algorithm, "B", 2.0)
    reward_free = getattr(original_algorithm, "reward_free", True)
    lambda_qreg = getattr(original_algorithm, "lambda_qreg", 0.01)
    policy_mode = getattr(original_algorithm, "policy_mode", "hybrid_dacer")
    lambda_rl = getattr(original_algorithm, "lambda_rl", 0.0)
    lambda_reg = getattr(original_algorithm, "lambda_reg", 1.0)
    rl_gain_clip = getattr(original_algorithm, "rl_gain_clip", 2.0)
    pre_takeover_bc_coef = getattr(original_algorithm, "pre_takeover_bc_coef", 0.0)
    pre_takeover_pv_coef = getattr(original_algorithm, "pre_takeover_pv_coef", 0.0)
    critic_objective = getattr(original_algorithm, "critic_objective", "cost")
    lambda_er = getattr(original_algorithm, "lambda_er", 0.0)
    er_margin = getattr(original_algorithm, "er_margin", 0.05)
    er_min_action_gap = getattr(original_algorithm, "er_min_action_gap", 0.03)
    er_positive_action = getattr(original_algorithm, "er_positive_action", "behavior")
    er_use_primal_dual = getattr(original_algorithm, "er_use_primal_dual", False)
    er_budget = getattr(original_algorithm, "er_budget", 0.0)
    er_dual_lr = getattr(original_algorithm, "er_dual_lr", 1e-3)
    er_eta_init = float(getattr(getattr(original_algorithm, "state", None), "eta_er", getattr(original_algorithm, "eta_er_init", lambda_er)))
    er_adaptive_margin = getattr(original_algorithm, "er_adaptive_margin", False)
    er_margin_alpha = getattr(original_algorithm, "er_margin_alpha", 0.25)
    er_margin_min = getattr(original_algorithm, "er_margin_min", 0.01)
    er_margin_max = getattr(original_algorithm, "er_margin_max", 0.20)
    lambda_pv_constraint = getattr(original_algorithm, "lambda_pv_constraint", 0.0)
    pv_constraint_margin = getattr(original_algorithm, "pv_constraint_margin", 0.1)
    pv_use_primal_dual = getattr(original_algorithm, "pv_use_primal_dual", False)
    pv_constraint_budget = getattr(original_algorithm, "pv_constraint_budget", 0.0)
    pv_dual_lr = getattr(original_algorithm, "pv_dual_lr", 1e-3)
    pv_eta_init = float(getattr(getattr(original_algorithm, "state", None), "eta_pv", getattr(original_algorithm, "eta_pv_init", lambda_pv_constraint)))
    if actor_lr is None:
        actor_lr = getattr(original_algorithm, "actor_lr", lr)
    if actor_delay is None:
        actor_delay = int(getattr(original_algorithm, "actor_delay", getattr(original_algorithm, "delay_update", 2)))
    if target_update_delay is None:
        target_update_delay = int(getattr(original_algorithm, "target_update_delay", getattr(original_algorithm, "delay_update", 2)))

    new_algorithm = PVPDACER(
        original_algorithm.agent, original_params,
        lr=lr, gamma=original_algorithm.gamma,
        tau=getattr(original_algorithm, "tau", 0.005),
        alpha_lr=getattr(original_algorithm, "alpha_lr", 3e-4),
        delay_alpha_update=getattr(original_algorithm, "delay_alpha_update", 10000),
        delay_update=getattr(original_algorithm, "delay_update", 2),
        actor_delay=actor_delay,
        target_update_delay=target_update_delay,
        reward_scale=getattr(original_algorithm, "reward_scale", 1.0),
        num_samples=original_algorithm.num_samples,
        actor_lr=actor_lr,
        lambda_pv=lambda_pv, B=B, lambda_bc=lambda_bc,
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
        er_eta_init=er_eta_init,
        er_adaptive_margin=er_adaptive_margin,
        er_margin_alpha=er_margin_alpha,
        er_margin_min=er_margin_min,
        er_margin_max=er_margin_max,
        lambda_pv_constraint=lambda_pv_constraint,
        pv_constraint_margin=pv_constraint_margin,
        pv_use_primal_dual=pv_use_primal_dual,
        pv_constraint_budget=pv_constraint_budget,
        pv_dual_lr=pv_dual_lr,
        pv_eta_init=pv_eta_init,
    )

    # ---- 组装 state ----
    state_kwargs: Dict[str, Any] = {"params": original_params}

    if transfer_state:
        state_kwargs["opt_state"] = original_opt_state
        state_kwargs["step"] = original_step
        state_kwargs["mean_q1_std"] = original_mean_q1
        state_kwargs["mean_q2_std"] = original_mean_q2
        state_kwargs["entropy"] = original_entropy

    # [FIX-5] 显式同步 target_params（若原 state 有该字段）
    if transfer_target_params and original_target_params is not None:
        new_state = new_algorithm.state
        new_has_target = hasattr(new_state, "target_params") or (
            isinstance(new_state, dict) and "target_params" in new_state
        )
        if new_has_target:
            state_kwargs["target_params"] = original_target_params

    try:
        new_algorithm.state = _state_replace(new_algorithm.state, **state_kwargs)
    except Exception as e:
        # 兜底：若某字段不被接受，去掉 target_params 再试
        print(f"   ⚠️  _create_algorithm_with_params: state_replace 失败({e})，退回最小迁移")
        minimal_kwargs = {"params": original_params}
        if transfer_state:
            minimal_kwargs["opt_state"] = original_opt_state
            minimal_kwargs["step"] = original_step
        new_algorithm.state = _state_replace(new_algorithm.state, **minimal_kwargs)

    # [FIX-4] Stage1b 等 lr 改变的场景显式重置 opt_state
    if reset_opt_state:
        fresh_opt_state = _reset_algorithm_opt_state(new_algorithm, new_algorithm.state.params)
        if fresh_opt_state is not None:
            try:
                new_algorithm.state = _state_replace(new_algorithm.state, opt_state=fresh_opt_state)
            except Exception as e:
                print(f"   ⚠️  reset_opt_state 写回失败: {e}")

    # [FIX-3] 内部 step 归零
    if reset_internal_step:
        _try_reset_algorithm_internal_step(new_algorithm, 0)

    return new_algorithm


def _create_stage1b_algorithm(original_algorithm: PVPDACER, config: Stage1bConfig) -> PVPDACER:
    """[FIX-4] Stage1b 改了 lr/actor_lr，必须重置 opt_state，否则旧动量方向错。"""
    return _create_algorithm_with_params(
        original_algorithm,
        lambda_pv=config.lambda_pv, lambda_bc=config.lambda_bc,
        lr=config.bc_lr, actor_lr=config.bc_lr,
        transfer_state=True,
        transfer_target_params=True,
        reset_opt_state=True,       # [FIX-4]
        reset_internal_step=False,  # Stage1b 内部 step 不强制归零
    )


def _create_stage2_algorithm(
    original_algorithm: PVPDACER,
    lambda_pv: float, lambda_bc: float, lr: float, B: float,
    actor_lr: float, actor_delay: int, target_update_delay: int,
) -> PVPDACER:
    """
    [FIX-5] Stage2 虽然不迁移 opt_state（给 RL 一个全新优化器起点），
    但必须显式迁移 target_params（如果 state 有该字段），否则 target 会从 BC 后的 main params
    被构造函数重新初始化，丢失 Stage1b 的收敛结果一致性。
    """
    return _create_algorithm_with_params(
        original_algorithm,
        lambda_pv=lambda_pv, lambda_bc=lambda_bc,
        lr=lr, B=B,
        actor_lr=actor_lr,
        actor_delay=actor_delay,
        target_update_delay=target_update_delay,
        transfer_state=False,
        transfer_target_params=True,  # [FIX-5]
        reset_opt_state=False,
        reset_internal_step=True,     # Stage2 是新阶段，step 从 0 开始
    )


def _create_boosted_algorithm(
    original_algorithm: PVPDACER,
    lambda_bc: float = 20.0, lr: Optional[float] = None,
) -> PVPDACER:
    hp = getattr(original_algorithm, "get_current_hyperparameters", lambda: {})()
    lambda_pv = float(hp.get("lambda_pv", 2.0))
    return _create_algorithm_with_params(
        original_algorithm,
        lambda_pv=lambda_pv, lambda_bc=lambda_bc,
        lr=lr or 3e-4,
        transfer_state=False,
        transfer_target_params=True,
        reset_opt_state=False,
        reset_internal_step=False,
    )


def _reset_algorithm_opt_state(algorithm: PVPDACER, params):
    """
    更稳的 opt_state 重置：
    1) 不再依赖 hasattr(opt_state, sub_name) 预判，先尝试 opt.init(param)
    2) 写回时同时兼容 namedtuple / dict / FrozenDict / 普通对象
    3) 某个子状态写不回时只跳过该子状态，不影响其他部分
    """
    state = getattr(algorithm, "state", None)
    opt_state = getattr(state, "opt_state", None)
    if opt_state is None:
        return None

    def _try_init(opt_attr_name: str, param_attr_name: str):
        opt = getattr(algorithm, opt_attr_name, None)
        if opt is None:
            return None
        p = _tree_get_key(params, param_attr_name, None)
        if p is None:
            return None
        try:
            return opt.init(p)
        except Exception as e:
            print(f"   ⚠️  _reset_algorithm_opt_state[{param_attr_name}] init 失败: {e}")
            return None

    replacements = {}
    for (opt_name, param_name) in [
        ("optim", "q1"),
        ("optim", "q2"),
        ("policy_optim", "policy"),
        ("alpha_optim", "log_alpha"),
    ]:
        new_sub = _try_init(opt_name, param_name)
        if new_sub is not None:
            replacements[param_name] = new_sub

    if not replacements:
        return opt_state

    new_opt_state = opt_state
    applied = 0
    for key, value in replacements.items():
        try:
            new_opt_state = _tree_set_key(new_opt_state, key, value)
            applied += 1
        except Exception as e:
            print(f"   ⚠️  _reset_algorithm_opt_state[{key}] 写回失败: {e}")

    if applied == 0:
        return opt_state
    return new_opt_state


    if hasattr(opt_state, "_replace"):
        try:
            return opt_state._replace(**replacements)
        except Exception as e:
            print(f"   ⚠️  opt_state._replace 失败: {e}")
            return opt_state

    try:
        for key, value in replacements.items():
            setattr(opt_state, key, value)
    except Exception:
        pass
    return opt_state


# ============================================================================
# BC-Boost Scheduler
# ============================================================================

class BCBoostScheduler:
    def __init__(self, config: Optional[BCBoostConfig] = None, steps_per_tick: int = 5):
        self.config = config or BCBoostConfig()
        self.steps_per_tick = max(1, int(steps_per_tick))
        self.boost_counter = 0
        self.pending_steps = 0
        self.local_key = jax.random.key(42)
        self.boosted_algorithm: Optional[PVPDACER] = None

    def _resolve_boost_lambda_bc(self, trainer) -> float:
        if self.config.lambda_bc is not None:
            return float(self.config.lambda_bc)
        if hasattr(trainer.algorithm, "lambda_bc"):
            return float(trainer.algorithm.lambda_bc)
        state = getattr(trainer.algorithm, "state", None)
        if state is not None and hasattr(state, "lambda_bc"):
            return float(np.asarray(state.lambda_bc))
        return 20.0

    def _resolve_boost_lr(self, trainer) -> float:
        if self.config.lr is not None:
            return float(self.config.lr)
        if hasattr(trainer.algorithm, "actor_lr"):
            return float(trainer.algorithm.actor_lr)
        if hasattr(trainer.algorithm, "lr"):
            return float(trainer.algorithm.lr)
        return 1.5e-4

    def maybe_trigger(self, trainer, new_corrections, trigger_every: Optional[int] = None):
        trigger_every = int(self.config.trigger_every if trigger_every is None else trigger_every)
        if trigger_every <= 0:
            return
        if new_corrections < trigger_every * (self.boost_counter + 1):
            return

        self.boost_counter += 1
        self.pending_steps += self.config.burst_steps
        self.local_key = jax.random.fold_in(self.local_key, self.boost_counter)
        boost_lambda_bc = self._resolve_boost_lambda_bc(trainer)
        boost_lr = self._resolve_boost_lr(trainer)

        if self.boosted_algorithm is None:
            self.boosted_algorithm = _create_boosted_algorithm(
                trainer.algorithm,
                lambda_bc=boost_lambda_bc,
                lr=boost_lr,
            )
        else:
            cur_params = trainer.algorithm.state.params
            new_opt_state = _reset_algorithm_opt_state(self.boosted_algorithm, cur_params)
            self.boosted_algorithm.state = _state_replace(
                self.boosted_algorithm.state,
                params=cur_params,
                opt_state=new_opt_state,
            )

    def _sample_human_batch(self, trainer):
        return _sample_human_bc_batch(
            trainer,
            self.config.burst_batch_size,
            to_jax=True,
            recent_steps=self.config.recent_steps,
        )

    def tick(self, trainer) -> Dict[str, float]:
        if self.pending_steps <= 0 or self.boosted_algorithm is None:
            return {}

        if len(trainer.buffer.human_buffer) < self.config.burst_batch_size:
            return {}

        last_metrics: Dict[str, float] = {}
        n_this_tick = min(self.steps_per_tick, self.pending_steps)
        last_m = None

        for _ in range(n_this_tick):
            self.local_key, bc_key = jax.random.split(self.local_key)
            batch = self._sample_human_batch(trainer)
            with JAX_UPDATE_LOCK:
                last_m = _bc_update(self.boosted_algorithm, bc_key, batch)
            self.pending_steps -= 1

        if last_m is not None:
            mh = jax.device_get(last_m)
            last_metrics = {
                "policy/bc_loss": float(mh.get("policy/bc_loss", 0.0)),
                "policy/rl_loss": float(mh.get("policy/rl_loss", 0.0)),
                "policy/pv_loss": float(mh.get("policy/pv_loss", 0.0)),
            }

        try:
            cur_state = trainer.algorithm.state
            boosted_state = self.boosted_algorithm.state
            new_params = _merge_policy_params(
                cur_state.params,
                boosted_state.params,
                mix_alpha=self.config.ema_alpha,
            )

            state_updates = {"params": new_params}
            cur_target_params = _extract_target_params(cur_state)
            if cur_target_params is not None:
                synced_target_params = _maybe_sync_target_policy_params(cur_target_params, new_params)
                if synced_target_params is not None:
                    state_updates["target_params"] = synced_target_params

            trainer.algorithm.state = _state_replace(cur_state, **state_updates)
            last_metrics["policy/mix_alpha"] = float(np.clip(self.config.ema_alpha, 0.0, 1.0))
        except Exception as e:
            print(f"   ⚠️  BC-Boost apply failed: {e}")

        if self.pending_steps <= 0:
            self.boosted_algorithm = None

        return last_metrics


# ============================================================================
# Human BC sampling
# ============================================================================

def _sample_human_bc_batch(trainer, batch_size: int, *, to_jax: bool = True, recent_steps: int = 200):
    if hasattr(trainer.buffer, "sample_boost_human_only"):
        return trainer.buffer.sample_boost_human_only(
            batch_size, recent_steps=recent_steps, to_jax=to_jax,
        )
    if hasattr(trainer.buffer, "sample_recent_human_only"):
        return trainer.buffer.sample_recent_human_only(
            batch_size, recent_steps=recent_steps, to_jax=to_jax,
        )
    return trainer.buffer.sample_human_only(batch_size, to_jax=to_jax)


# ============================================================================
# 环境创建与 takeover 工具
# ============================================================================


def _unwrap_env(env, max_depth: int = 10):
    e = env
    visited = set()
    for _ in range(max_depth):
        if id(e) in visited:
            break
        visited.add(id(e))
        if hasattr(e, "env"):
            e = e.env
        else:
            break
    try:
        if hasattr(e, "unwrapped"):
            e = e.unwrapped
    except Exception:
        pass
    return e


def _find_hitl_env(env, max_depth: int = 10):
    current = env
    for _ in range(max_depth):
        if isinstance(current, HumanInTheLoopEnv):
            return current
        if hasattr(current, "env"):
            current = current.env
        else:
            break
    return getattr(env, "unwrapped", env)


def create_human_in_the_loop_pvp_env(
    env_seed: int,
    controller: str = "keyboard", manual_control: bool = False,
    use_render: bool = False, start_seed: int = TRAIN_MAP_SEEDS[0],
    num_scenarios: int = len(TRAIN_MAP_SEEDS), traffic_density: float = 0.06,
) -> Tuple[Any, int, int]:
    env_config = {
        "manual_control": manual_control,
        "enable_takeover": True,
        "controller": controller,
        "use_render": use_render,
        "start_seed": start_seed,
        "num_scenarios": num_scenarios,
        "traffic_density": traffic_density,
        "horizon": 1000,
        "vehicle_config": {
            "show_lidar": False,
            "show_side_detector": False,
            "show_lane_line_detector": False,
        },
        "show_logo": False,
        "show_fps": True,
    }
    env = HumanInTheLoopEnv(env_config)
    reset_seed = int(start_seed + (env_seed % num_scenarios))
    obs, _info = env.reset(seed=reset_seed)
    obs_dim = len(obs)
    act_dim = 2

    print(
        f"环境创建完成 | controller={controller} | init_seed_range={start_seed}-{start_seed + num_scenarios - 1}"
        f" | fixed_stage_maps={get_fixed_stage_map_seeds()} | obs={obs_dim} | act={act_dim}"
    )
    return env, obs_dim, act_dim


def _build_no_takeover_eval_env(args) -> Any:
    env_config = {
        "manual_control": False,
        "enable_takeover": False,
        "controller": str(getattr(args, "controller", "keyboard")),
        "use_render": False,
        "start_seed": int(getattr(args, "start_seed", TRAIN_MAP_SEEDS[0])),
        "num_scenarios": int(getattr(args, "num_scenarios", len(TRAIN_MAP_SEEDS))),
        "traffic_density": float(getattr(args, "traffic_density", 0.06)),
        "horizon": 1000,
        "vehicle_config": {
            "show_lidar": False,
            "show_side_detector": False,
            "show_lane_line_detector": False,
        },
        "show_logo": False,
        "show_fps": True,
        "out_of_route_done": True,
        "crash_done": True,
    }
    return HumanInTheLoopEnv(env_config)


def _ensure_no_takeover_eval_env(trainer, args, allowed_seeds: Optional[List[int]] = None):
    env = getattr(trainer, "no_takeover_eval_env", None)
    if env is None:
        env = _build_no_takeover_eval_env(args)
        trainer.no_takeover_eval_env = env
        if allowed_seeds:
            try:
                install_strict_fixed_seed_controller(
                    env,
                    allowed_seeds,
                    mode="cycle",
                    start_index=0,
                    rng_seed=int(getattr(args, "seed", 0)) + 4040,
                )
                trainer.no_takeover_eval_allowed_seeds = tuple(allowed_seeds)
            except Exception:
                pass
    elif allowed_seeds and tuple(allowed_seeds) != getattr(trainer, "no_takeover_eval_allowed_seeds", ()):
        try:
            install_strict_fixed_seed_controller(
                env,
                allowed_seeds,
                mode="cycle",
                start_index=0,
                rng_seed=int(getattr(args, "seed", 0)) + 4040,
            )
            trainer.no_takeover_eval_allowed_seeds = tuple(allowed_seeds)
        except Exception:
            pass
    return env


def _run_no_takeover_eval(
    trainer,
    algorithm: PVPDACER,
    args,
    *,
    current_step: int,
    prefix: str,
    num_episodes: int,
    allowed_seeds: Optional[List[int]] = None,
) -> Dict[str, float]:
    if int(num_episodes) <= 0:
        return {}

    eval_env = _ensure_no_takeover_eval_env(trainer, args, allowed_seeds=allowed_seeds)
    seeds = list(allowed_seeds or get_fixed_stage_map_seeds() or [int(getattr(args, "start_seed", TRAIN_MAP_SEEDS[0]))])

    returns: List[float] = []
    lengths: List[int] = []
    successes: List[float] = []
    crashes: List[float] = []
    out_of_roads: List[float] = []
    takeover_rates: List[float] = []
    autonomous_steps: List[float] = []
    first_takeover_steps: List[float] = []

    for ep in range(int(num_episodes)):
        eval_seed = int(seeds[ep % len(seeds)])
        obs = reset_to_strict_stage_map(eval_env, eval_seed)
        ep_return = 0.0
        ep_len = 0
        ep_takeover = 0
        first_takeover_step = -1
        ep_success = 0.0
        ep_crash = 0.0
        ep_out_of_road = 0.0

        while True:
            action = np.asarray(algorithm.get_deterministic_action(obs), dtype=np.float32)
            obs, reward, terminated, truncated, info = eval_env.step(np.clip(action, -1.0, 1.0))
            ep_len += 1
            ep_return += float(reward)
            takeover_flag = float(info.get("takeover", False))
            if takeover_flag > 0.5:
                ep_takeover += 1
                if first_takeover_step < 0:
                    first_takeover_step = ep_len
            if info.get("crash", False):
                ep_crash = 1.0
            if info.get("out_of_road", False):
                ep_out_of_road = 1.0
            if info.get("success", False) or info.get("arrive_dest", False):
                ep_success = 1.0
            if bool(terminated) or bool(truncated):
                break

        returns.append(ep_return)
        lengths.append(ep_len)
        successes.append(ep_success)
        crashes.append(ep_crash)
        out_of_roads.append(ep_out_of_road)
        takeover_rates.append(ep_takeover / max(ep_len, 1))
        autonomous_steps.append(float(max(ep_len - ep_takeover, 0)))
        first_takeover_steps.append(float(first_takeover_step if first_takeover_step >= 0 else ep_len))

    metrics = {
        f"{prefix}/raw_return": float(np.mean(returns)),
        f"{prefix}/episode_length": float(np.mean(lengths)),
        f"{prefix}/success_rate": float(np.mean(successes)),
        f"{prefix}/crash_rate": float(np.mean(crashes)),
        f"{prefix}/out_of_road_rate": float(np.mean(out_of_roads)),
        f"{prefix}/takeover_rate": float(np.mean(takeover_rates)),
        f"{prefix}/autonomous_steps": float(np.mean(autonomous_steps)),
        f"{prefix}/first_takeover_step": float(np.mean(first_takeover_steps)),
    }

    logger = getattr(trainer, "logger", None)
    if logger is not None:
        for tag, value in metrics.items():
            logger.add_scalar(tag, value, int(current_step))

    print(
        f"   📉 {prefix}: return={metrics[f'{prefix}/raw_return']:.2f}"
        f" | success={metrics[f'{prefix}/success_rate']:.3f}"
        f" | crash={metrics[f'{prefix}/crash_rate']:.3f}"
        f" | oor={metrics[f'{prefix}/out_of_road_rate']:.3f}"
        f" | len={metrics[f'{prefix}/episode_length']:.1f}"
        f" | takeover={metrics[f'{prefix}/takeover_rate']:.3f}"
    )
    return metrics


# ============================================================================
# Stage 1a：演示收集
# ============================================================================

def _get_shared_control_monitor(env) -> Optional[SharedControlMonitor]:
    if isinstance(env, SharedControlMonitor):
        return env
    return getattr(env, "shared_control_monitor", None)


def _save_demo_snapshot(
    trainer, log_path: Path, map_seed: int, pass_index: int,
    prefix: str = "seed", label: str = "",
) -> bool:
    monitor = _get_shared_control_monitor(trainer.env)
    if monitor is None or not monitor.data or len(monitor.data.get("observation", [])) == 0:
        return False

    start_step = int(getattr(monitor, "last_save_step", 0))
    end_step = int(getattr(monitor, "step_count", 0))
    if end_step <= start_step:
        return False

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder = Path(monitor.folder)
    folder.mkdir(parents=True, exist_ok=True)
    target_name = f"{prefix}{map_seed}_{pass_index}_{timestamp}.pkl"
    target_path = folder / target_name
    suffix = 1
    while target_path.exists():
        target_path = folder / f"{prefix}{map_seed}_{pass_index}_{timestamp}_{suffix}.pkl"
        suffix += 1

    before_files = {p.resolve() for p in folder.glob("*.pkl")}

    try:
        monitor._save_data(num_save_steps=end_step - start_step)
    except Exception:
        return False

    after_files = {p.resolve() for p in folder.glob("*.pkl")}
    new_files = sorted(after_files - before_files)

    def _finalize(src_path: Path) -> bool:
        Path(src_path).rename(target_path)
        monitor.last_save_step = end_step
        log_label = f" [{label}]" if label else ""
        print(f"   💾{log_label} 已保存: {target_path.name}")
        return True

    if len(new_files) == 1:
        return _finalize(new_files[0])
    if len(new_files) > 1:
        newest = max((Path(p) for p in new_files), key=lambda p: p.stat().st_mtime)
        return _finalize(newest)

    expected = folder / f"{monitor.prefix}_step_{start_step}_{end_step}.pkl"
    if expected.exists():
        return _finalize(expected)
    return False


def _save_stage1a_demo_snapshot(trainer, log_path: Path, map_seed: int, pass_index: int) -> bool:
    return _save_demo_snapshot(trainer, log_path, map_seed, pass_index,
                               prefix="seed", label="Stage1a")


def _save_stage2_demo_snapshot(trainer, log_path: Path, map_seed: int, pass_index: int) -> bool:
    return _save_demo_snapshot(trainer, log_path, map_seed, pass_index,
                               prefix="stage2_seed", label="Stage2")


def _stage2_flush_monitor(trainer) -> None:
    monitor = _get_shared_control_monitor(trainer.env)
    if monitor is None:
        return
    monitor.last_save_step = int(getattr(monitor, "step_count", 0))
    monitor.data = {key: [] for key in monitor.data}


def _experience_targets_human_buffer(experience) -> bool:
    """
    [FIX-2] 原实现仅访问 experience.interventions（复数），但 create_pvp_experience 传入参数名是
    intervention（单数）。若 Experience 内部字段名是单数，原写法会被 except 静默吞掉，导致
    stage2_new_human_total 永远为 0，BC-Boost 永不触发。这里做双字段兼容。
    """
    # 读取 intervention（兼容复数/单数）
    try:
        val = getattr(experience, "interventions", None)
        if val is None:
            val = getattr(experience, "intervention", None)
        if val is None:
            intervention = 0.0
        else:
            intervention = float(np.asarray(val, dtype=np.float32).reshape(-1)[0])
    except Exception:
        intervention = 0.0

    # 读取 is_pre_takeover
    try:
        val = getattr(experience, "is_pre_takeover", None)
        if val is None:
            val = getattr(experience, "pre_takeover", None)
        if val is None:
            is_pre_takeover = 0.0
        else:
            is_pre_takeover = float(np.asarray(val, dtype=np.float32).reshape(-1)[0])
    except Exception:
        is_pre_takeover = 0.0

    try:
        val = getattr(experience, "is_demo", None)
        if val is None:
            is_demo = 0.0
        else:
            is_demo = float(np.asarray(val, dtype=np.float32).reshape(-1)[0])
    except Exception:
        is_demo = 0.0

    return intervention > 0.5 or is_pre_takeover > 0.5 or is_demo > 0.5


def _install_stage2_human_counter(trainer) -> None:
    buffer = getattr(trainer, "buffer", None)
    if buffer is None or getattr(buffer, "_stage2_human_counter_installed", False):
        return

    original_add = buffer.add

    def counted_add(experience, *extra_args, **kwargs):
        result = original_add(experience, *extra_args, **kwargs)
        try:
            if getattr(trainer, "stage2_started", False) and _experience_targets_human_buffer(experience):
                trainer.stage2_new_human_total = int(getattr(trainer, "stage2_new_human_total", 0)) + 1
        except Exception:
            pass
        return result

    buffer.add = counted_add
    buffer._stage2_human_counter_installed = True
    buffer._stage2_human_counter_original_add = original_add


def _merge_stage1a_demo_files(
    recorded_data_dir: Path,
    output_path: Path,
    action_dim: Optional[int] = None,
    allow_missing_novice_action: bool = False,
) -> bool:
    pkl_files = sorted(recorded_data_dir.glob("seed*.pkl"))
    if not pkl_files:
        return False
    if action_dim is None:
        raise RuntimeError("合并 Stage1a demo 时必须提供 action_dim，禁止静默退回 2 维动作")

    merged: Dict[str, List] = {
        "observation": [], "next_observation": [],
        "action_behavior": [], "action_novice": [], "action_human": [],
        "reward": [], "done": [], "intervention": [], "stop_td": [],
        "takeover_start": [], "is_demo": [],
    }
    total_steps = 0

    for f in pkl_files:
        try:
            with open(f, "rb") as fh:
                data = pickle.load(fh)
            if "observation" not in data or len(data["observation"]) == 0:
                continue

            n = len(data["observation"])
            default_actions = np.zeros((n, action_dim), dtype=np.float32)

            merged["observation"].append(np.array(data["observation"]))
            action_behavior_all = data.get("action_behavior", data.get("action", default_actions))
            action_novice_all = _resolve_action_novice(
                data,
                action_behavior_all,
                source=f"merged demo file {f.name}",
                allow_missing=bool(allow_missing_novice_action),
            )
            action_human_all = data.get("action_human", data.get("human_action", action_behavior_all))
            merged["action_behavior"].append(np.array(action_behavior_all))
            merged["action_novice"].append(np.array(action_novice_all))
            merged["action_human"].append(np.array(action_human_all))
            merged["reward"].append(np.array(data.get("reward", np.zeros(n))))
            merged["done"].append(np.array(data.get("done", np.zeros(n, dtype=bool))))

            merged["intervention"].append(np.zeros(n, dtype=np.float32))
            merged["stop_td"].append(np.ones(n, dtype=np.float32))
            merged["takeover_start"].append(np.zeros(n, dtype=np.float32))
            merged["is_demo"].append(np.ones(n, dtype=np.float32))

            next_obs_key = _pick_sequence_key(data, "next_observation", "next_obs")
            if next_obs_key is not None:
                merged["next_observation"].append(np.array(data[next_obs_key]))
            else:
                obs_arr = np.array(data["observation"])
                done_arr = np.array(data.get("done", np.zeros(n, dtype=np.float32)), dtype=np.float32)
                merged["next_observation"].append(np.array(_build_next_obs_sequence(obs_arr, done_arr)))

            total_steps += n
        except Exception as e:
            print(f"   ⚠️  合并 {f.name} 失败: {e}")

    if total_steps == 0:
        return False

    result = {k: np.concatenate(v, axis=0) for k, v in merged.items() if v}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as fh:
        pickle.dump(result, fh)
    print(f"   💾 Stage1a 合并完成: {len(pkl_files)} 文件 → {total_steps} 步")
    return True


def run_stage1a_demo_collection(trainer, args, log_path: Path) -> Tuple[int, Dict[int, int]]:
    required_completions = int(args.demo_passes_per_map)
    target_maps = get_fixed_stage_map_seeds()

    print(f"\n🎮 Stage 1a: 演示收集 | 固定地图 {target_maps} | 每图 {required_completions} 遍")
    print(f"   🔒 Stage1a 严格顺序地图模式：{' -> '.join(map(str, target_maps))} 循环")

    if not args.manual_control:
        raise RuntimeError("Stage1a 需要 --manual_control")
    if not hasattr(trainer, "sample_human_demo"):
        raise RuntimeError("trainer.sample_human_demo not found")

    install_strict_fixed_seed_controller(
        trainer.env,
        target_maps,
        mode="cycle",
        start_index=0,
        rng_seed=int(args.seed) + 101,
    )

    map_stats: Dict[int, int] = {m: 0 for m in target_maps}
    original_upi = trainer.update_per_iteration

    try:
        trainer.update_per_iteration = 0

        map_cursor = 1
        obs = reset_to_strict_stage_map(trainer.env, target_maps[0])
        trainer.prev_intervention = 0.0

        current_episode_seed = _get_running_map_seed(trainer.env, fallback=target_maps[0])

        _disable_env_sequential_reset(trainer.env)

        steps = 0

        while True:
            seed_before_step = _get_running_map_seed(trainer.env, fallback=current_episode_seed)
            ep_before = int(getattr(trainer.sample_log, "sample_episode", 0))

            if getattr(args, "print_seed_every_step", False):
                seed_str = int(seed_before_step) if seed_before_step is not None else "unknown"
                print(f"[Stage1a-Step] step={steps:5d} | running_seed={seed_str}")

            obs = trainer.sample_human_demo(obs)
            steps += 1

            ep_after = int(getattr(trainer.sample_log, "sample_episode", 0))
            seed_after_step = _get_running_map_seed(trainer.env, fallback=seed_before_step)
            current_episode_seed = seed_after_step

            if steps % 500 == 0:
                done_maps = sum(1 for m in target_maps if map_stats.get(m, 0) >= required_completions)
                running_seed_str = int(seed_before_step) if seed_before_step is not None else "unknown"
                print(
                    f"[Stage1a] step={steps:5d} | running_seed={running_seed_str} "
                    f"| done={done_maps}/{len(target_maps)}"
                )

            if ep_after > ep_before:
                completed_seed = int(seed_before_step) if seed_before_step is not None else -1

                if completed_seed not in target_maps:
                    print(f"   ⚠️  检测到非目标地图 seed={completed_seed}，忽略本次统计")
                else:
                    map_stats[completed_seed] = map_stats.get(completed_seed, 0) + 1
                    cur_pass = map_stats[completed_seed]

                    print(f"🏁 地图 seed={completed_seed} 完成第 {cur_pass} 遍")

                    _save_stage1a_demo_snapshot(trainer, log_path, completed_seed, pass_index=cur_pass)

                    if all(map_stats.get(m, 0) >= required_completions for m in target_maps):
                        print(f"\n🎯 所有地图完成 {required_completions} 遍！")
                        print(f"   最终统计: {map_stats}")
                        break

                next_seed = target_maps[map_cursor % len(target_maps)]
                map_cursor += 1

                trainer.prev_intervention = 0.0
                auto_reset_seed = _get_running_map_seed(trainer.env, fallback=next_seed)
                if auto_reset_seed != next_seed:
                    print(
                        f"   ⚠️ 自动 reset seed={auto_reset_seed} 与预期 {next_seed} 不一致，执行显式纠正"
                    )
                    obs = reset_to_strict_stage_map(trainer.env, next_seed)
                    current_episode_seed = next_seed
                else:
                    current_episode_seed = auto_reset_seed

                print(f"   ↪ 下一张固定地图: seed={current_episode_seed}")

    finally:
        trainer.update_per_iteration = original_upi

    print(f"   ✅ Stage1a 完成 | steps={steps} | human_buf={len(trainer.buffer.human_buffer)}")

    data_dir = Path(str(args.data_dir))
    action_dim = _get_action_dim_from_trainer(trainer)
    _merge_stage1a_demo_files(
        data_dir / "recorded_data",
        data_dir / "stage1a_merged_demo.pkl",
        action_dim=action_dim,
        allow_missing_novice_action=bool(getattr(args, "allow_missing_novice_action", False)),
    )
    return steps, map_stats


# ============================================================================
# Stage 1b：离线 BC
# ============================================================================

def run_stage1b_bc_updates(
    trainer, algorithm: PVPDACER, args, log_path: Path,
):
    stage1b_lambda_bc = float(getattr(args, "stage1b_lambda_bc", 50.0))
    config = Stage1bConfig(total_bc_updates=int(args.stage1b_updates), lambda_bc=stage1b_lambda_bc)

    print(f"\n🔄 Stage 1b: 离线BC | {config.total_bc_updates} 次更新 | lambda_bc={config.lambda_bc}")
    print(f"   Stage1b source policy_norm(before BC)={_policy_param_norm(algorithm):.6e}")

    allow_fallback = bool(getattr(args, "allow_stage1b_update_fallback", False))

    if _has_bc_only_update(algorithm):
        pass
    elif allow_fallback:
        print("   ⚠️  bc_only_update 不存在，使用 fallback（非纯BC）")
    else:
        raise RuntimeError(
            "❌ bc_only_update 不存在！Stage1b 需要纯BC更新。\n"
            "   方案A：在 PVPDACER 中实现 bc_only_update()\n"
            "   方案B：添加 --allow_stage1b_update_fallback"
        )

    if len(trainer.buffer.human_buffer) == 0:
        loaded = False

        demo_root = getattr(args, "demo_root", None)
        if demo_root and not loaded:
            root_path = Path(demo_root)
            if not root_path.is_absolute():
                root_path = Path.cwd() / root_path
            if root_path.exists():
                print(f"📂 Stage1b: 从 --demo_root 递归加载: {root_path}")
                loaded = load_all_demos_recursive(str(root_path), trainer)

        if not loaded and getattr(args, "demo_dir", None):
            dp = Path(args.demo_dir)
            if not dp.is_absolute():
                dp = Path.cwd() / dp
            if dp.exists():
                loaded = load_demo_data_directory(str(dp), trainer)

        if not loaded and getattr(args, "demo_file", None):
            fp = Path(args.demo_file)
            if not fp.is_absolute():
                fp = Path.cwd() / fp
            if fp.exists():
                loaded = load_integrated_demo_data(str(fp), trainer)

        if not loaded:
            if getattr(args, "data_dir", None):
                data_root = Path(str(args.data_dir))
            else:
                data_root = log_path / "data"

            stage1a_merged = data_root / "stage1a_merged_demo.pkl"
            stage1a_dir = data_root / "recorded_data"
            if stage1a_merged.exists():
                loaded = load_integrated_demo_data(str(stage1a_merged), trainer)
            elif stage1a_dir.exists():
                loaded = load_demo_data_directory(str(stage1a_dir), trainer)

    print(f"   human_buffer: {len(trainer.buffer.human_buffer)} 条")

    if len(trainer.buffer.human_buffer) < config.update_batch_size:
        print(f"❌ 数据不足: {len(trainer.buffer.human_buffer)} < {config.update_batch_size}")
        return 0, algorithm

    stage1b_algorithm = _create_stage1b_algorithm(algorithm, config)
    print(
        f"   Stage1b init copied source params | "
        f"policy_diff_from_source={_policy_param_diff(stage1b_algorithm, algorithm):.6e} "
        f"| policy_norm={_policy_param_norm(stage1b_algorithm):.6e}"
    )
    trainer.algorithm = stage1b_algorithm

    debug_bs = min(int(getattr(args, "stage1b_debug_fixed_batch_size", 64)), len(trainer.buffer.human_buffer))
    debug_batch = _sample_human_bc_batch(trainer, debug_bs, to_jax=True)
    debug_human_actions = jax.device_get(debug_batch.actions_human)
    init_policy_params = _get_policy_only_params(stage1b_algorithm)

    bc_update_count = 0
    successful_count = 0
    consecutive_fail_count = 0
    MAX_CONSECUTIVE_FAILS = 5
    train_key = jax.random.fold_in(jax.random.key(int(args.seed)), 12345)
    logger = getattr(trainer, "logger", None)

    if getattr(args, "stage1b_debug_overfit_fixed_batch", False):
        print("   ⚠️  stage1b_debug_overfit_fixed_batch=True: 在固定 batch 上循环，评估 MAE 会偏乐观")

    while bc_update_count < config.total_bc_updates:
        try:
            if getattr(args, "stage1b_debug_overfit_fixed_batch", False):
                human_batch = debug_batch
            else:
                human_batch = _sample_human_bc_batch(trainer, config.update_batch_size, to_jax=True)
            update_key, train_key = jax.random.split(train_key)

            do_debug = (bc_update_count < 5) or ((bc_update_count + 1) % config.update_interval == 0)
            if do_debug:
                policy_before = _get_policy_only_params(stage1b_algorithm)

            with JAX_UPDATE_LOCK:
                metrics = _bc_update(stage1b_algorithm, update_key, human_batch)

            if do_debug:
                policy_after = _get_policy_only_params(stage1b_algorithm)
                policy_step_diff = _tree_l2_diff(policy_before, policy_after)
            else:
                policy_step_diff = None

            bc_update_count += 1
            successful_count += 1
            consecutive_fail_count = 0

            if bc_update_count % config.update_interval == 0:
                bc_loss = float(np.asarray(metrics.get("policy/bc_loss", 0.0)))
                pct = (bc_update_count / config.total_bc_updates) * 100.0

                pred_det = jax.device_get(
                    _safe_get_deterministic_action(stage1b_algorithm.agent, stage1b_algorithm.state.params, debug_batch.obs)
                )
                mae_det = float(np.mean(np.abs(pred_det - debug_human_actions)))
                policy_total_diff = _tree_l2_diff(init_policy_params, _get_policy_only_params(stage1b_algorithm))

                msg = (
                    f"🔄 BC: {bc_update_count}/{config.total_bc_updates} ({pct:.1f}%) | "
                    f"BC={bc_loss:.6f} | MAE={mae_det:.4f} | Δpolicy={policy_total_diff:.6e}"
                )
                if policy_step_diff is not None:
                    msg += f" | Δstep={policy_step_diff:.6e}"
                print(msg)

                if logger is not None:
                    logger.add_scalar("stage1b/bc_loss", bc_loss, bc_update_count)
                    logger.add_scalar("stage1b_debug/mae_det", mae_det, bc_update_count)
                    logger.add_scalar("stage1b_debug/policy_total_diff", policy_total_diff, bc_update_count)
                    _log_metric_dict(logger, metrics, bc_update_count, prefix="stage1b_metrics")

        except RuntimeError as e:
            if "buffer" in str(e).lower() and "small" in str(e).lower():
                print(f"   ⚠️  Buffer 过小，Stage1b 提前结束: {e}")
                break
            consecutive_fail_count += 1
            print(f"   ⚠️  BC RuntimeError (step={bc_update_count}, consec_fail={consecutive_fail_count}): {e}")
            if consecutive_fail_count >= MAX_CONSECUTIVE_FAILS:
                print(f"   ⚠️  连续 {MAX_CONSECUTIVE_FAILS} 次失败，终止 Stage1b")
                break
            continue
        except Exception as e:
            consecutive_fail_count += 1
            print(f"   ⚠️  BC 异常 (step={bc_update_count}, consec_fail={consecutive_fail_count}): {e}")
            if consecutive_fail_count >= MAX_CONSECUTIVE_FAILS:
                print(f"   ⚠️  连续 {MAX_CONSECUTIVE_FAILS} 次失败，终止 Stage1b")
                break
            continue

    print(f"   ✅ Stage1b 完成 | 成功: {successful_count} / {bc_update_count}")
    print(
        f"   Stage1b final policy_norm={_policy_param_norm(stage1b_algorithm):.6e} "
        f"| policy_diff_from_initial={_tree_l2_diff(init_policy_params, _get_policy_only_params(stage1b_algorithm)):.6e}"
    )

    try:
        eval_n = min(256, len(trainer.buffer.human_buffer))
        eval_batch = trainer.buffer.sample_human_only(eval_n, to_jax=True)
        pred = jax.device_get(_safe_get_deterministic_action(stage1b_algorithm.agent, stage1b_algorithm.state.params, eval_batch.obs))
        human = jax.device_get(eval_batch.actions_human)
        steer_mae = float(np.mean(np.abs(pred[:, 0] - human[:, 0])))
        thr_mae = float(np.mean(np.abs(pred[:, 1] - human[:, 1])))
        print(f"   📊 最终评估: Steering MAE={steer_mae:.4f} | Throttle MAE={thr_mae:.4f}")
        if steer_mae > 0.3:
            print(f"   ⚠️  Steering MAE > 0.3，BC 收敛不足")

        if logger is not None:
            logger.add_scalar("stage1b_eval/steering_mae", steer_mae, bc_update_count)
            logger.add_scalar("stage1b_eval/throttle_mae", thr_mae, bc_update_count)
    except Exception as e:
        print(f"   ⚠️  离线评估失败: {e}")

    print("   ⏭️  跳过 Stage1b rollout 评估，直接进入 Stage2")

    ckpt_path = log_path / "stage1b_offline_bc_policy.pkl"
    try:
        stage1b_algorithm.save_policy(str(ckpt_path))
        print(f"   💾 Checkpoint: {ckpt_path.name}")
    except Exception as e:
        print(f"   ⚠️  Checkpoint 保存失败: {e}")

    try:
        _run_no_takeover_eval(
            trainer,
            stage1b_algorithm,
            args,
            current_step=0,
            prefix="eval_no_takeover_stage1b",
            num_episodes=int(getattr(args, "eval_no_takeover_episodes", 0)),
            allowed_seeds=get_fixed_stage_map_seeds(),
        )
    except Exception as e:
        print(f"   ⚠️  Stage1b no-takeover eval 失败: {e}")

    return bc_update_count, stage1b_algorithm


def warmup_novice_buffer(
    trainer, iter_key_fn, obs: np.ndarray,
    min_novice_size: int = 2000, max_env_steps: int = 20000,
    warmup_key_offset: int = 10 ** 7,
    allowed_seeds: Optional[List[int]] = None,
) -> Tuple[np.ndarray, int]:
    current_novice = len(trainer.buffer.novice_buffer)
    if current_novice >= min_novice_size:
        print(f"   ✅ novice_buffer 已有 {current_novice} 条，跳过填充")
        return obs, 0

    print(f"\n🔧 Novice 填充：目标 {min_novice_size}，当前 {current_novice}")

    if allowed_seeds is not None:
        try:
            _configure_allowed_seeds(trainer.env, allowed_seeds)
        except Exception:
            pass
        try:
            # 保持与 Stage2 主循环一致的严格顺序地图语义，避免多 seed 时 warmup 先走随机种子。
            obs = reset_to_strict_stage_map(trainer.env, int(allowed_seeds[0]))
        except Exception as e:
            raise RuntimeError(
                f"novice warmup 初始 reset 失败，无法在正确地图上启动填充: {e}"
            ) from e

    env_steps = 0
    current_episode_seed = _get_running_map_seed(
        trainer.env,
        fallback=allowed_seeds[0] if allowed_seeds else None,
    )

    while len(trainer.buffer.novice_buffer) < min_novice_size and env_steps < max_env_steps:
        s_keys, _ = iter_key_fn(env_steps + warmup_key_offset)

        seed_before_step = _get_running_map_seed(trainer.env, fallback=current_episode_seed)
        obs = trainer.sample(s_keys[0], obs)
        seed_after_step = _get_running_map_seed(trainer.env, fallback=seed_before_step)
        current_episode_seed = seed_after_step

        env_steps += 1

        if env_steps % 500 == 0:
            running_seed_str = int(seed_before_step) if seed_before_step is not None else "unknown"
            print(
                f"   novice 填充: {env_steps}/{max_env_steps}"
                f" | seed={running_seed_str}"
                f" | novice={len(trainer.buffer.novice_buffer)}/{min_novice_size}"
            )

    print(f"   ✅ novice 填充完成: {len(trainer.buffer.novice_buffer)} | env_steps={env_steps}")
    return obs, env_steps


# ============================================================================
# Stage 2：PVP + BC 训练
# ============================================================================

def run_stage2_training(trainer, algorithm: PVPDACER, log_path: Path, args, train_key) -> int:
    if int(args.total_step) <= 0:
        print(f"\n⏭️  Stage 2: total_step={args.total_step}，跳过在线训练")
        return 0

    print(f"\n🚀 Stage 2: PVP+BC 训练 → {args.total_step} 步")

    trainer.sample_log.sample_step = 0
    trainer.update_log.update_step = 0
    trainer.start_step = 0
    if hasattr(trainer, "global_step"):
        trainer.global_step = 0

    trainer.last_metrics = dict(getattr(trainer, "last_metrics", {}) or {})
    trainer.stage2_human_steps = 0
    trainer.stage2_agent_steps = 0
    trainer.stage2_new_human_total = 0

    if isinstance(getattr(trainer, "env", None), SharedControlMonitor):
        trainer.env.save_freq = 10 ** 9
        trainer.env.prefix = "stage2"
        print("   ✅ SharedControlMonitor 保留（pre-takeover 链路完整），磁盘写入已关闭，按 episode 手动保存")

    base_env = _find_hitl_env(trainer.env)

    try:
        cfg = _safe_config_dict(base_env)
        if cfg is not None:
            cfg.update({
                "enable_takeover": True,
                "manual_control": False,
                "out_of_route_done": True,
                "crash_done": True,
            })
    except Exception:
        pass

    try:
        from relax.utils.takeover_policy import create_takeover_policy
        base_env.takeover_policy = create_takeover_policy(base_env, args.controller)
        if base_env.takeover_policy is not None:
            base_env.takeover = False
            base_env.takeover_policy.takeover = False
            base_env.takeover_policy.last_button_state = False
            print("   ✅ Stage2 takeover policy installed; agent drives by default, human takes over only while takeover input is active")
    except Exception as e:
        print(f"   ⚠️  Takeover policy reset failed: {e}")

    print(
        f"   Stage2 source policy_norm(from Stage1b/DBP)={_policy_param_norm(algorithm):.6e}"
    )
    stage2_algorithm = _create_stage2_algorithm(
        algorithm,
        lambda_pv=float(args.lambda_pv), lambda_bc=float(args.lambda_bc),
        lr=float(args.lr), B=float(args.B),
        actor_lr=float(args.actor_lr),
        actor_delay=int(args.actor_delay),
        target_update_delay=int(args.target_update_delay),
    )
    print(
        f"   Stage2 init copied Stage1b params | "
        f"policy_diff_from_stage1b={_policy_param_diff(stage2_algorithm, algorithm):.6e} "
        f"| policy_norm={_policy_param_norm(stage2_algorithm):.6e}"
    )
    trainer.algorithm = stage2_algorithm
    trainer.total_step = int(args.total_step)
    trainer.update_per_iteration = 2
    trainer.start_step = 0
    trainer.configure_value_guidance(
        enable=bool(args.use_value_guidance),
        lambda_0=float(args.guidance_lambda0),
        beta_unc=float(args.guidance_beta_unc),
        p_decay=float(args.guidance_p_decay),
        grad_clip=float(args.guidance_grad_clip),
        guidance_mode=int(args.guidance_mode_id),
        guidance_target=int(args.guidance_target_id),
        guidance_step_interval=int(args.guidance_step_interval),
        guidance_q_agg=int(args.guidance_q_agg_id),
        guidance_kappa=float(args.guidance_kappa),
        guidance_injection=int(args.guidance_injection_id),
        guidance_schedule=int(args.guidance_schedule_id),
    )
    trainer._guidance_arm_step = int(args.guidance_arm_step)
    trainer.arm_value_guidance(False)
    if args.use_value_guidance:
        print(
            "   UPV-GDS configured:"
            f" lambda0={args.guidance_lambda0}"
            f" | beta_unc={args.guidance_beta_unc}"
            f" | p_decay={args.guidance_p_decay}"
            f" | grad_clip={args.guidance_grad_clip}"
            f" | mode={args.guidance_mode}"
            f" | q_agg={args.guidance_q_agg}"
            f" | kappa={args.guidance_kappa}"
            f" | injection={args.guidance_injection}"
            f" | schedule={args.guidance_schedule}"
            f" | arm_step={args.guidance_arm_step}"
        )

    allowed_seeds = get_fixed_stage_map_seeds()
    print(f"🔒 Stage2 严格固定训练地图 seeds = {allowed_seeds}")
    print(f"   🔒 Stage2 严格顺序地图模式：{' -> '.join(map(str, allowed_seeds))} 循环")

    install_strict_fixed_seed_controller(
        trainer.env,
        allowed_seeds,
        mode="cycle",
        start_index=0,
        rng_seed=int(args.seed) + 2026,
    )

    min_novice_size = int(getattr(args, "stage2_min_novice_size", 2000))

    obs = reset_to_strict_stage_map(trainer.env, allowed_seeds[0])
    initial_map_cursor = 1

    return _train_main_loop(
        trainer,
        train_key,
        obs,
        log_path,
        args,
        allowed_seeds=allowed_seeds,
        min_novice_size=min_novice_size,
        map_cursor=initial_map_cursor,
    )


def _train_main_loop(
    trainer, train_key, obs: np.ndarray, log_path: Path, args,
    allowed_seeds: Optional[List[int]] = None,
    min_novice_size: int = 2000,
    map_cursor: int = 0,
) -> int:
    sl = trainer.sample_log
    ul = trainer.update_log
    stage2_map_pass_counts: Dict[int, int] = {}
    missing_allowed_seed_warning_emitted = False

    critic_warmup_steps = int(getattr(args, "stage2_critic_warmup_steps", 3000))
    stage2_rl_warmup_steps = int(getattr(args, "stage2_rl_warmup_steps", 1000))
    stage2_rl_ramp_steps = int(getattr(args, "stage2_rl_ramp_steps", 2000))
    stage2_bc_decay = bool(getattr(args, "stage2_bc_decay", True))
    stage2_bc_freeze_steps = int(getattr(args, "stage2_bc_freeze_steps", 3000))
    lambda_bc_init = float(args.lambda_bc)
    lambda_bc_min = lambda_bc_init * 0.1
    current_lambda_bc = lambda_bc_init
    lambda_rl_target = float(getattr(args, "lambda_rl", 0.0))
    current_lambda_rl = lambda_rl_target
    warmup_key_offset = int(getattr(args, "stage2_warmup_key_offset", 10 ** 7))
    max_warmup_env_steps = int(getattr(args, "stage2_max_warmup_env_steps", critic_warmup_steps * 10))

    if stage2_bc_decay and not _has_set_lambda_bc(trainer.algorithm):
        stage2_bc_decay = False
    if not _has_set_lambda_rl(trainer.algorithm):
        lambda_rl_target = 0.0
        current_lambda_rl = 0.0

    iter_key_fn = create_iter_key_fn(train_key, trainer.sample_per_iteration, trainer.update_per_iteration)

    try:
        trainer.progress.unpause()
        trainer.progress.total = trainer.total_step
        trainer.progress.n = int(sl.sample_step)
        trainer.progress.refresh()
    except Exception:
        pass

    trainer.stage2_started = True
    trainer.last_metrics = dict(getattr(trainer, "last_metrics", {}) or {})
    trainer.stage2_initial_human_size = len(trainer.buffer.human_buffer)
    trainer.stage2_initial_novice_size = len(trainer.buffer.novice_buffer)
    print(f"📊 Stage2 | Human0={trainer.stage2_initial_human_size} | Novice0={trainer.stage2_initial_novice_size}")

    current_episode_seed = _get_running_map_seed(trainer.env, fallback=None)

    obs, _ = warmup_novice_buffer(
        trainer=trainer, iter_key_fn=iter_key_fn, obs=obs,
        min_novice_size=min_novice_size, max_env_steps=max_warmup_env_steps,
        warmup_key_offset=warmup_key_offset,
        allowed_seeds=allowed_seeds,
    )

    current_episode_seed = _get_running_map_seed(trainer.env, fallback=current_episode_seed)

    if critic_warmup_steps > 0 and _has_critic_only_update(trainer.algorithm):
        print(f"\n🔥 Q 热身：{critic_warmup_steps} 次 critic 更新...")
        warmup_key = jax.random.fold_in(train_key, 99999)
        warmup_critic_updates = 0
        warmup_env_steps_q = 0

        while warmup_critic_updates < critic_warmup_steps:
            try:
                if len(trainer.buffer) < trainer.batch_size:
                    if warmup_env_steps_q >= max_warmup_env_steps:
                        break
                    s_keys, _ = iter_key_fn(warmup_env_steps_q + warmup_key_offset + 1)

                    seed_before_step = _get_running_map_seed(trainer.env, fallback=current_episode_seed)
                    obs = trainer.sample(s_keys[0], obs)
                    seed_after_step = _get_running_map_seed(trainer.env, fallback=seed_before_step)
                    current_episode_seed = seed_after_step

                    warmup_env_steps_q += 1
                    continue

                batch = trainer.buffer.sample(trainer.batch_size, to_jax=True)
                warmup_key, c_key = jax.random.split(warmup_key)
                with JAX_UPDATE_LOCK:
                    _critic_update(trainer.algorithm, c_key, batch)
                warmup_critic_updates += 1

                if warmup_critic_updates % 500 == 0:
                    running_seed = _get_running_map_seed(trainer.env, fallback=current_episode_seed)
                    running_seed_str = int(running_seed) if running_seed is not None else "unknown"
                    print(
                        f"   Q热身: {warmup_critic_updates}/{critic_warmup_steps}"
                        f" | seed={running_seed_str}"
                    )
            except Exception as e:
                print(f"   ⚠️  Q热身异常: {e}")
                break

        print(f"   ✅ Q热身完成: {warmup_critic_updates} 次更新")

    # [FIX-3] warmup 结束后，日志计数 + 算法内部 step 一起归零
    warmup_sample_steps = int(getattr(sl, "sample_step", 0))
    warmup_update_steps = int(getattr(ul, "update_step", 0)) if hasattr(ul, "update_step") else 0
    if warmup_sample_steps or warmup_update_steps:
        print(
            f"   🔄 Stage2 warmup 已完成 | env_steps={warmup_sample_steps} | updates={warmup_update_steps} | 正式训练计数归零"
        )
    sl.sample_step = 0
    if hasattr(ul, "update_step"):
        ul.update_step = 0

    # [FIX-3] 关键修正：同步重置算法内部 step，否则 actor_delay / target_update_delay / delay_alpha_update 会错位
    algo_step_reset_ok = _try_reset_algorithm_internal_step(trainer.algorithm, 0)
    if algo_step_reset_ok:
        print("   🔄 algorithm.state.step 已同步归零（actor_delay / target_delay 从 0 开始计数）")
    else:
        print("   ⚠️  algorithm.state.step 归零失败（state 可能无 step 字段或结构不支持写回），delay 计数可能漂移")

    try:
        trainer.progress.n = 0
        trainer.progress.refresh()
    except Exception:
        pass

    bc_boost_scheduler = BCBoostScheduler(
        config=BCBoostConfig(
            burst_steps=int(getattr(args, "stage2_bc_boost_burst_steps", 20)),
            burst_batch_size=int(getattr(args, "stage2_bc_boost_burst_batch_size", 64)),
            recent_steps=int(getattr(args, "stage2_bc_boost_recent_steps", 200)),
            lambda_bc=(
                None if getattr(args, "stage2_bc_boost_lambda_bc", None) is None
                else float(args.stage2_bc_boost_lambda_bc)
            ),
            lr=(
                None if getattr(args, "stage2_bc_boost_lr", None) is None
                else float(args.stage2_bc_boost_lr)
            ),
            ema_alpha=float(getattr(args, "stage2_bc_boost_ema_alpha", 0.2)),
            trigger_every=int(getattr(args, "stage2_bc_boost_trigger_every", 200)),
        ),
        steps_per_tick=int(getattr(args, "stage2_bc_boost_steps_per_tick", 1)),
    )
    print(
        "   🚀 BC-Boost:"
        f" trigger_every={bc_boost_scheduler.config.trigger_every}"
        f" | burst={bc_boost_scheduler.config.burst_steps}"
        f" | burst_bs={bc_boost_scheduler.config.burst_batch_size}"
        f" | ema={bc_boost_scheduler.config.ema_alpha:.2f}"
        f" | lambda_bc={'follow-stage2' if bc_boost_scheduler.config.lambda_bc is None else bc_boost_scheduler.config.lambda_bc}"
        f" | lr={'follow-actor_lr' if bc_boost_scheduler.config.lr is None else bc_boost_scheduler.config.lr}"
    )
    logger = getattr(trainer, "logger", None)

    last_step = -1
    stall_count = stall_alerts = 0
    last_print_step = int(sl.sample_step)
    print_interval = 500

    save_every = getattr(trainer, "save_policy_every", 0)
    next_save_step = None
    if isinstance(save_every, int) and save_every > 0:
        next_save_step = ((int(sl.sample_step) // save_every) + 1) * save_every
    eval_every = int(getattr(args, "eval_no_takeover_every", 0))
    next_eval_step = None
    if eval_every > 0:
        next_eval_step = ((int(sl.sample_step) // eval_every) + 1) * eval_every

    ep_steps_acc = ep_takeover = 0
    last_ep_count = int(getattr(sl, "sample_episode", 0))

    current_policy_mode = str(getattr(trainer.algorithm, "policy_mode", getattr(args, "policy_mode", "hybrid_dacer"))).lower()
    current_lambda_rl = float(getattr(trainer.algorithm, "lambda_rl", lambda_rl_target))
    show_rl_actor_logs = (current_policy_mode != "pvp_paper" and current_lambda_rl > 0.0)
    rl_warmup_active = (
        stage2_rl_warmup_steps > 0
        and _has_bc_only_update(trainer.algorithm)
        and show_rl_actor_logs
    )
    rl_warmup_announced_end = False
    rl_ramp_announced_end = False
    rl_warmup_key = jax.random.fold_in(train_key, 88888)

    if rl_warmup_active and show_rl_actor_logs:
        print(f"   🛡️  RL Warmup: 前 {stage2_rl_warmup_steps} 步屏蔽 RL loss")
        print(f"   🔀 RL Ramp: warmup 结束后 {stage2_rl_ramp_steps} 步内 full update 概率从 0% → 100%")

    _stage2_flush_monitor(trainer)

    print(f"\n=== Stage2 主循环 (0 → {trainer.total_step}) ===")

    try:
        while int(sl.sample_step) < int(trainer.total_step):
            current_step = int(sl.sample_step)

            if (
                getattr(trainer, "use_value_guidance", False)
                and not getattr(trainer, "_guidance_armed", False)
                and current_step >= int(getattr(trainer, "_guidance_arm_step", 0))
            ):
                trainer.arm_value_guidance(True)
                print(f"UPV-GDS guidance armed @ sample_step={current_step}")
                if logger is not None:
                    logger.add_scalar("guidance/armed", 1.0, current_step)

            if stage2_bc_decay:
                if current_step < stage2_bc_freeze_steps:
                    current_lambda_bc = lambda_bc_init
                else:
                    decay_total = max(1, trainer.total_step - stage2_bc_freeze_steps)
                    decay_ratio = min(1.0, (current_step - stage2_bc_freeze_steps) / decay_total)
                    current_lambda_bc = lambda_bc_init * (1.0 - decay_ratio) + lambda_bc_min * decay_ratio
                try:
                    trainer.algorithm.set_lambda_bc(current_lambda_bc)
                except Exception:
                    pass
            else:
                current_lambda_bc = lambda_bc_init

            if show_rl_actor_logs and _has_set_lambda_rl(trainer.algorithm):
                if current_step < stage2_rl_warmup_steps:
                    current_lambda_rl = 0.0
                elif stage2_rl_ramp_steps > 0 and current_step < stage2_rl_warmup_steps + stage2_rl_ramp_steps:
                    ramp_progress = (current_step - stage2_rl_warmup_steps) / max(1, stage2_rl_ramp_steps)
                    current_lambda_rl = lambda_rl_target * min(max(ramp_progress, 0.0), 1.0)
                else:
                    current_lambda_rl = lambda_rl_target
                try:
                    trainer.algorithm.set_lambda_rl(current_lambda_rl)
                except Exception:
                    pass
            else:
                current_lambda_rl = 0.0 if current_policy_mode == "pvp_paper" else lambda_rl_target

            sample_keys, update_keys = iter_key_fn(sl.sample_step)

            for i in range(trainer.sample_per_iteration):
                seed_before_step = _get_running_map_seed(trainer.env, fallback=current_episode_seed)

                if getattr(args, "print_seed_every_step", False):
                    running_seed_str = int(seed_before_step) if seed_before_step is not None else "unknown"
                    print(f"[Stage2-Step] step={int(sl.sample_step):5d} | running_seed={running_seed_str}")

                obs = trainer.sample(sample_keys[i], obs)

                seed_after_step = _get_running_map_seed(trainer.env, fallback=seed_before_step)
                current_episode_seed = seed_after_step

                ep_steps_acc += 1
                if float(getattr(trainer, "prev_intervention", 0.0)) > 0.5:
                    ep_takeover += 1

                cur_ep = int(getattr(sl, "sample_episode", 0))
                if cur_ep > last_ep_count:
                    completed_seed = int(seed_before_step) if seed_before_step is not None else -1
                    next_seed = None

                    if allowed_seeds:
                        next_seed = int(allowed_seeds[map_cursor % len(allowed_seeds)])
                        map_cursor = (map_cursor + 1) % len(allowed_seeds)

                    next_seed_str = int(next_seed) if next_seed is not None else "unknown"

                    print(
                        f"🏁 [Stage2-Episode] seed={completed_seed}"
                        f" | len={ep_steps_acc}"
                        f" | takeover_rate={ep_takeover / max(ep_steps_acc, 1):.3f}"
                        f" | next_seed={next_seed_str}"
                    )

                    if logger is not None:
                        log_step = int(sl.sample_step)
                        logger.add_scalar("episode/length", ep_steps_acc, log_step)
                        logger.add_scalar("episode/takeover_rate", ep_takeover / max(ep_steps_acc, 1), log_step)
                        if seed_before_step is not None:
                            logger.add_scalar("episode/seed", float(seed_before_step), log_step)

                    if allowed_seeds:
                        if completed_seed in allowed_seeds:
                            stage2_map_pass_counts[completed_seed] = stage2_map_pass_counts.get(completed_seed, 0) + 1
                            _save_stage2_demo_snapshot(
                                trainer, log_path, completed_seed, stage2_map_pass_counts[completed_seed]
                            )
                    elif not missing_allowed_seed_warning_emitted:
                        print("⚠️ allowed_seeds 未配置，Stage2 快照保存已禁用")
                        missing_allowed_seed_warning_emitted = True

                    ep_steps_acc = ep_takeover = 0
                    last_ep_count = cur_ep

                    if next_seed is not None:
                        trainer.prev_intervention = 0.0
                        auto_reset_seed = _get_running_map_seed(trainer.env, fallback=next_seed)
                        if auto_reset_seed != next_seed:
                            print(
                                f"   ⚠️ 自动 reset seed={auto_reset_seed} 与预期 {next_seed} 不一致，执行显式纠正"
                            )
                            obs = reset_to_strict_stage_map(trainer.env, next_seed)
                            current_episode_seed = next_seed
                        else:
                            current_episode_seed = auto_reset_seed
                    else:
                        current_episode_seed = _get_running_map_seed(trainer.env, fallback=current_episode_seed)

                if next_save_step is not None and int(sl.sample_step) >= next_save_step:
                    try:
                        pkl_name = trainer.policy_pkl_template.format(
                            sample_step=next_save_step, update_step=ul.update_step
                        )
                        trainer.algorithm.save_policy(str(trainer.log_path / pkl_name))
                    except Exception:
                        pass
                    next_save_step += save_every

            post_sample_step = int(sl.sample_step)
            in_rl_warmup = rl_warmup_active and post_sample_step < stage2_rl_warmup_steps

            if not in_rl_warmup and rl_warmup_active:
                steps_since_warmup = post_sample_step - stage2_rl_warmup_steps
                in_rl_ramp = (stage2_rl_ramp_steps > 0 and steps_since_warmup < stage2_rl_ramp_steps)
            else:
                steps_since_warmup = 0
                in_rl_ramp = False

            if not in_rl_warmup and not in_rl_ramp:
                stage2_new_human = int(getattr(trainer, "stage2_new_human_total", 0))
                bc_boost_scheduler.maybe_trigger(trainer, stage2_new_human, trigger_every=int(args.stage2_bc_boost_trigger_every))

            if int(sl.sample_step) % 5000 == 0:
                gc.collect()

            # ---- 更新 ----
            for i in range(trainer.update_per_iteration):
                if in_rl_warmup:
                    bc_ok = False
                    critic_ok = False
                    combined_direct_metrics: Dict[str, float] = {}
                    next_update_step = int(ul.update_step) + 1

                    if _has_critic_only_update(trainer.algorithm) and len(trainer.buffer) >= trainer.batch_size:
                        try:
                            full_batch = trainer.buffer.sample(trainer.batch_size, to_jax=True)
                            rl_warmup_key, crit_key = jax.random.split(rl_warmup_key)
                            with JAX_UPDATE_LOCK:
                                critic_metrics = _critic_update(trainer.algorithm, crit_key, full_batch)
                            combined_direct_metrics.update(_host_metric_dict(critic_metrics))
                            critic_ok = True
                        except Exception as e:
                            print(f"⚠️ warmup critic failed: {e}")

                    human_buf_size = len(trainer.buffer.human_buffer)
                    min_human_bs = 8
                    if human_buf_size >= min_human_bs:
                        try:
                            human_bs = min(human_buf_size, trainer.batch_size)
                            human_batch = _sample_human_bc_batch(trainer, human_bs, to_jax=True)
                            rl_warmup_key, bc_key = jax.random.split(rl_warmup_key)
                            with JAX_UPDATE_LOCK:
                                metrics = _bc_update(trainer.algorithm, bc_key, human_batch)
                            combined_direct_metrics.update(_host_metric_dict(metrics))
                            bc_ok = True
                        except Exception as e:
                            print(f"⚠️ warmup BC failed: {e}")
                            combined_direct_metrics["policy/bc_loss"] = float("nan")
                    elif not critic_ok:
                        combined_direct_metrics["policy/bc_loss"] = float("nan")

                    did_full_update = False
                    if not bc_ok and not critic_ok:
                        try:
                            with JAX_UPDATE_LOCK:
                                trainer.update(update_keys[i])
                            did_full_update = True
                            if logger is not None:
                                logger.add_scalar("stage2/direct_bc_applied", 0.0, next_update_step)
                                logger.add_scalar("stage2/direct_critic_applied", 0.0, next_update_step)
                                logger.add_scalar("stage2/full_update_applied", 1.0, next_update_step)
                                logger.add_scalar("stage2/warmup_fallback_full", 1.0, next_update_step)
                        except Exception as e:
                            print(f"⚠️ warmup fallback full update failed: {e}")
                            if logger is not None:
                                logger.add_scalar("stage2/direct_bc_applied", 0.0, next_update_step)
                                logger.add_scalar("stage2/direct_critic_applied", 0.0, next_update_step)
                                logger.add_scalar("stage2/full_update_applied", 0.0, next_update_step)
                                logger.add_scalar("stage2/warmup_fallback_full", 0.0, next_update_step)
                                logger.add_scalar("stage2/update_skipped", 1.0, next_update_step)
                    else:
                        trainer.last_metrics = dict(getattr(trainer, "last_metrics", {}) or {})
                        trainer.last_metrics.update(combined_direct_metrics)
                        if logger is not None:
                            _log_metric_dict(logger, combined_direct_metrics, next_update_step)
                            logger.add_scalar("stage2/direct_bc_applied", float(bc_ok), next_update_step)
                            logger.add_scalar("stage2/direct_critic_applied", float(critic_ok), next_update_step)
                            logger.add_scalar("stage2/full_update_applied", 0.0, next_update_step)

                    if hasattr(ul, "update_step") and not did_full_update and (bc_ok or critic_ok):
                        ul.update_step = next_update_step

                else:
                    if in_rl_ramp:
                        rl_prob = steps_since_warmup / stage2_rl_ramp_steps
                        rl_warmup_key, gate_key = jax.random.split(rl_warmup_key)
                        do_full = bool(jax.device_get(jax.random.uniform(gate_key) < rl_prob))
                    else:
                        do_full = True

                    if do_full:
                        next_update_step = int(ul.update_step) + 1
                        if logger is not None:
                            logger.add_scalar("stage2/direct_bc_applied", 0.0, next_update_step)
                            logger.add_scalar("stage2/direct_critic_applied", 0.0, next_update_step)
                            logger.add_scalar("stage2/full_update_applied", 1.0, next_update_step)
                        with JAX_UPDATE_LOCK:
                            trainer.update(update_keys[i])
                    else:
                        ramp_bc_ok = False
                        ramp_critic_ok = False
                        combined_direct_metrics = {}
                        next_update_step = int(ul.update_step) + 1
                        if _has_critic_only_update(trainer.algorithm) and len(trainer.buffer) >= trainer.batch_size:
                            try:
                                full_batch = trainer.buffer.sample(trainer.batch_size, to_jax=True)
                                rl_warmup_key, crit_key = jax.random.split(rl_warmup_key)
                                with JAX_UPDATE_LOCK:
                                    critic_metrics = _critic_update(trainer.algorithm, crit_key, full_batch)
                                combined_direct_metrics.update(_host_metric_dict(critic_metrics))
                                ramp_critic_ok = True
                            except Exception as e:
                                print(f"⚠️ ramp critic failed: {e}")

                        human_buf_size = len(trainer.buffer.human_buffer)
                        if human_buf_size >= 8:
                            try:
                                human_bs = min(human_buf_size, trainer.batch_size)
                                human_batch = _sample_human_bc_batch(trainer, human_bs, to_jax=True)
                                rl_warmup_key, bc_key = jax.random.split(rl_warmup_key)
                                with JAX_UPDATE_LOCK:
                                    ramp_metrics = _bc_update(trainer.algorithm, bc_key, human_batch)
                                combined_direct_metrics.update(_host_metric_dict(ramp_metrics))
                                ramp_bc_ok = True
                            except Exception as e:
                                print(f"⚠️ ramp BC failed: {e}")

                        if not ramp_bc_ok and not ramp_critic_ok:
                            try:
                                with JAX_UPDATE_LOCK:
                                    trainer.update(update_keys[i])
                                if logger is not None:
                                    logger.add_scalar("stage2/direct_bc_applied", 0.0, next_update_step)
                                    logger.add_scalar("stage2/direct_critic_applied", 0.0, next_update_step)
                                    logger.add_scalar("stage2/full_update_applied", 1.0, next_update_step)
                                    logger.add_scalar("stage2/ramp_fallback_full", 1.0, next_update_step)
                            except Exception as e:
                                print(f"⚠️ ramp fallback full update failed: {e}")
                                if logger is not None:
                                    logger.add_scalar("stage2/direct_bc_applied", 0.0, next_update_step)
                                    logger.add_scalar("stage2/direct_critic_applied", 0.0, next_update_step)
                                    logger.add_scalar("stage2/full_update_applied", 0.0, next_update_step)
                                    logger.add_scalar("stage2/update_skipped", 1.0, next_update_step)
                        elif hasattr(ul, "update_step"):
                            trainer.last_metrics = dict(getattr(trainer, "last_metrics", {}) or {})
                            trainer.last_metrics.update(combined_direct_metrics)
                            if logger is not None:
                                _log_metric_dict(logger, combined_direct_metrics, next_update_step)
                                logger.add_scalar("stage2/direct_bc_applied", float(ramp_bc_ok), next_update_step)
                                logger.add_scalar("stage2/direct_critic_applied", float(ramp_critic_ok), next_update_step)
                                logger.add_scalar("stage2/full_update_applied", 0.0, next_update_step)
                            ul.update_step = next_update_step

            if not in_rl_warmup and not in_rl_ramp:
                boost_metrics = bc_boost_scheduler.tick(trainer)
                if boost_metrics:
                    tagged = {f"bc_boost/{k}": float(v) for k, v in boost_metrics.items()}
                    trainer.last_metrics = dict(getattr(trainer, "last_metrics", {}) or {})
                    trainer.last_metrics.update(tagged)
                    if logger is not None:
                        for tag, value in tagged.items():
                            logger.add_scalar(tag, value, current_step)

            if logger is not None:
                logger.add_scalar("bc_boost/pending_steps", float(bc_boost_scheduler.pending_steps), current_step)

            trainer.progress.update(trainer.sample_per_iteration)

            current_step = int(sl.sample_step)
            if current_step == last_step:
                stall_count += 1
                if stall_count >= 10:
                    stall_alerts += 1
                    stall_count = 0
                    if stall_alerts >= 3:
                        raise RuntimeError("训练进度连续停滞")
            else:
                stall_count = stall_alerts = 0
            last_step = current_step

            if current_step - last_print_step >= print_interval:
                pct = (current_step / trainer.total_step) * 100.0
                human_total = len(trainer.buffer.human_buffer)
                novice_total = len(trainer.buffer.novice_buffer)
                stage2_human = int(getattr(trainer, "stage2_new_human_total", 0))
                stage2_novice = novice_total - trainer.stage2_initial_novice_size

                if show_rl_actor_logs and in_rl_warmup:
                    phase_str = " [BC+Critic Warmup]"
                elif show_rl_actor_logs and in_rl_ramp:
                    ramp_pct = min(100.0, steps_since_warmup / max(1, stage2_rl_ramp_steps) * 100)
                    phase_str = f" [RL Ramp {ramp_pct:.0f}%]"
                else:
                    phase_str = ""

                last_metrics = getattr(trainer, "last_metrics", None) or {}
                loss_parts = []
                metric_items = [("policy/bc_loss", "bc")]
                if show_rl_actor_logs:
                    metric_items.append(("policy/rl_loss", "rl"))
                metric_items.extend([
                    ("pv1_loss", "pv1"),
                    ("pv/q_h_mean", "q_h"),
                    ("pv/q_n_mean", "q_n"),
                    ("pv/margin", "mgn"),
                ])
                for k, label in metric_items:
                    v = last_metrics.get(k)
                    if v is not None:
                        loss_parts.append(f"{label}={float(v):.4f}")
                loss_str = " | ".join(loss_parts)

                running_seed = _get_running_map_seed(trainer.env, fallback=current_episode_seed)
                running_seed_str = int(running_seed) if running_seed is not None else "unknown"

                print(
                    f"[Stage2]{phase_str} {current_step:5d}/{trainer.total_step}"
                    f" ({pct:5.1f}%) | seed={running_seed_str}"
                    f" | λbc={current_lambda_bc:.2f}"
                    f" | λrl={current_lambda_rl:.3f}"
                    f" | +h_seen={stage2_human:4d} +n={stage2_novice:4d}"
                    f" | {loss_str}"
                )

                if show_rl_actor_logs and rl_warmup_active and not rl_warmup_announced_end and not in_rl_warmup:
                    rl_warmup_announced_end = True
                    print(f"\n🎯 RL Warmup 结束 (step={current_step})，进入 RL 渐入期 ({stage2_rl_ramp_steps} 步)")

                if show_rl_actor_logs and rl_warmup_active and not rl_ramp_announced_end and not in_rl_warmup and not in_rl_ramp:
                    rl_ramp_announced_end = True
                    print(f"\n🚀 RL Ramp 结束 (step={current_step})，切换到 100% full update")

                last_print_step = current_step

                if logger is not None:
                    logger.add_scalar("buffer/human_total", human_total, current_step)
                    logger.add_scalar("buffer/novice_total", novice_total, current_step)
                    logger.add_scalar("buffer/human_seen_total", stage2_human, current_step)
                    logger.add_scalar("stage2/lambda_bc", current_lambda_bc, current_step)
                    logger.add_scalar("stage2/lambda_rl", current_lambda_rl, current_step)
                    if show_rl_actor_logs:
                        logger.add_scalar("stage2/in_rl_warmup", float(in_rl_warmup), current_step)
                        logger.add_scalar("stage2/in_rl_ramp", float(in_rl_ramp), current_step)
                        if in_rl_ramp:
                            logger.add_scalar("stage2/rl_ramp_prob", float(steps_since_warmup / max(1, stage2_rl_ramp_steps)), current_step)
                    metric_keys = [
                        "policy/bc_loss",
                        "policy/actor_update_applied",
                        "policy/steps_since_actor_update",
                        "policy/actor_delay",
                        "policy/actor_lr",
                        "policy/action_diff_det_nonexpert",
                        "policy/pre_takeover_rate",
                        "policy/q_gain_nonexpert",
                        "pv1_loss", "pv2_loss",
                        "pv/loss_ratio",
                        "pv/q_h_mean", "pv/q_n_mean", "pv/margin",
                        "pv/q1_h_mean", "pv/q1_n_mean",
                        "pv/q2_h_mean", "pv/q2_n_mean",
                        "q/range",
                        "critic/td_effective_count",
                        "critic/td_effective_frac",
                        "critic/pt_td_leak_count",
                        "critic/demo_td_leak_count",
                        "critic/target_update_applied",
                        "critic/target_update_delay",
                        "critic/pre_takeover_frac",
                        "critic/intervention_frac",
                        "td_mask_mean", "intervention_frac",
                        "total_q1_loss", "total_q2_loss",
                    ]
                    if show_rl_actor_logs:
                        metric_keys.insert(1, "policy/rl_loss")
                    for k in metric_keys:
                        v = last_metrics.get(k)
                        if v is not None:
                            tag = k.replace("/", "_")
                            logger.add_scalar(f"stage2/{tag}", float(v), current_step)
                    _log_metric_dict(logger, last_metrics, current_step, prefix="stage2_metrics")

            if current_step % 1000 == 0 and logger is not None:
                logger.flush()

            if next_eval_step is not None and current_step >= next_eval_step:
                if getattr(trainer, "inline_no_takeover_eval_disabled", False):
                    next_eval_step = None
                    continue
                try:
                    eval_metrics = _run_no_takeover_eval(
                        trainer,
                        trainer.algorithm,
                        args,
                        current_step=current_step,
                        prefix="eval_no_takeover",
                        num_episodes=int(getattr(args, "eval_no_takeover_episodes", 0)),
                        allowed_seeds=allowed_seeds,
                    )
                    trainer.last_metrics = dict(getattr(trainer, "last_metrics", {}) or {})
                    trainer.last_metrics.update(eval_metrics)
                except Exception as e:
                    trainer.inline_no_takeover_eval_disabled = True
                    next_eval_step = None
                    print(
                        f"⚠️ no-takeover eval disabled after failure @ step={current_step}: {e} "
                        f"| 请改用 scripts/eval_pvp_policies_fixed_v3.py 做离线评估"
                    )
                if next_eval_step is not None:
                    next_eval_step += eval_every

    finally:
        final_step = int(sl.sample_step)
        final_path = log_path / f"final_policy_{final_step}.pkl"
        try:
            trainer.algorithm.save_policy(str(final_path))
            print(f"💾 Final policy @ step {final_step}")
        except Exception as e:
            print(f"⚠️  Final policy save failed: {e}")

        try:
            meta = {
                "sample_step": final_step,
                "update_step": int(ul.update_step) if hasattr(ul, "update_step") else 0,
                "lambda_bc": float(current_lambda_bc),
                "lambda_pv": float(getattr(args, "lambda_pv", 0.0)),
                "stage2_rl_warmup_steps": int(stage2_rl_warmup_steps),
                "stage2_rl_ramp_steps": int(stage2_rl_ramp_steps),
                "actor_lr": float(getattr(args, "actor_lr", 0.0)),
                "actor_delay": int(getattr(args, "actor_delay", 1)),
                "target_update_delay": int(getattr(args, "target_update_delay", 1)),
                "use_value_guidance": bool(getattr(args, "use_value_guidance", False)),
                "guidance_lambda0": float(getattr(args, "guidance_lambda0", 0.0)),
                "guidance_beta_unc": float(getattr(args, "guidance_beta_unc", 0.0)),
                "guidance_p_decay": float(getattr(args, "guidance_p_decay", 0.0)),
                "guidance_grad_clip": float(getattr(args, "guidance_grad_clip", 0.0)),
                "guidance_mode": str(getattr(args, "guidance_mode", "proxy_value")),
                "guidance_arm_step": int(getattr(args, "guidance_arm_step", 0)),
            }
            with open(log_path / f"final_policy_{final_step}.meta.pkl", "wb") as f:
                pickle.dump(meta, f)
        except Exception:
            pass

        if not getattr(trainer, "inline_no_takeover_eval_disabled", False):
            try:
                _run_no_takeover_eval(
                    trainer,
                    trainer.algorithm,
                    args,
                    current_step=final_step,
                    prefix="eval_no_takeover_final",
                    num_episodes=int(getattr(args, "eval_no_takeover_episodes", 0)),
                    allowed_seeds=allowed_seeds,
                )
            except Exception as e:
                print(f"⚠️  Final no-takeover eval failed: {e}")

        try:
            logger_obj = getattr(trainer, "logger", None)
            if logger_obj is not None:
                logger_obj.flush()
                logger_obj.close()
        except Exception:
            pass

        try:
            _save_final_log_bundle(
                log_path=log_path,
                trainer=trainer,
                args=args,
                final_step=final_step,
                current_lambda_bc=current_lambda_bc,
            )
        except Exception as e:
            print(f"⚠️  Final bundle save failed: {e}")

        try:
            eval_env = getattr(trainer, "no_takeover_eval_env", None)
            if eval_env is not None:
                eval_env.close()
                trainer.no_takeover_eval_env = None
        except Exception:
            pass

    return int(sl.sample_step)


def _safe_copy_path(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _save_final_log_bundle(
    log_path: Path,
    trainer,
    args: argparse.Namespace,
    final_step: int,
    current_lambda_bc: float,
) -> Path:
    bundle_dir = log_path / f"final_bundle_step_{final_step}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with open(bundle_dir / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    with open(bundle_dir / "buffer.pkl", "wb") as f:
        pickle.dump(trainer.buffer, f)

    for name in [
        f"final_policy_{final_step}.pkl",
        f"final_policy_{final_step}.meta.pkl",
        "stage1b_offline_bc_policy.pkl",
    ]:
        src = log_path / name
        if src.exists():
            _safe_copy_path(src, bundle_dir / name)

    for dname in ["tb", "data"]:
        src = log_path / dname
        if src.exists():
            _safe_copy_path(src, bundle_dir / dname)

    with open(bundle_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write("PVP-DACER final bundle\n")
        f.write("=" * 60 + "\n")
        f.write(f"log_path: {log_path}\n")
        f.write(f"final_step: {final_step}\n")
        f.write(f"lambda_bc_final: {current_lambda_bc:.6f}\n")
        f.write(f"human_buffer: {len(getattr(trainer.buffer, 'human_buffer', []))}\n")
        f.write(f"novice_buffer: {len(getattr(trainer.buffer, 'novice_buffer', []))}\n")
        f.write(f"saved_at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"📦 Final bundle saved: {bundle_dir}")
    return bundle_dir


# ============================================================================
# 三阶段总驱动
# ============================================================================

def run_three_stage_training(
    args: argparse.Namespace, trainer, algorithm: PVPDACER,
    log_path: Path, train_key,
) -> int:
    print(f"\n🚀 三阶段训练")
    if getattr(args, "demo_root", None) or getattr(args, "demo_dir", None) or getattr(args, "demo_file", None):
        print("   1a: 使用已有演示数据，跳过在线演示收集")
    else:
        print(f"   1a: 演示收集 ({args.demo_passes_per_map} 遍/地图)")
    print(f"   1b: 离线BC/DBP预训练 {args.stage1b_updates} 次")
    if (
        str(getattr(args, "policy_mode", "hybrid_dacer")).lower() == "pvp_paper"
        or float(getattr(args, "lambda_rl", 0.0)) <= 0.0
    ):
        print(f"   2:  PVP+BC {args.total_step} 步")
    else:
        print(f"   2:  PVP+BC {args.total_step} 步 (RL warmup {args.stage2_rl_warmup_steps} 步 + ramp {args.stage2_rl_ramp_steps} 步)")

    if getattr(args, "demo_root", None):
        print(f"   📂 demo_root: {args.demo_root}")

    if not args.force_demo_collection and not args.demo_dir and not getattr(args, "demo_root", None):
        default_root = Path(__file__).resolve().parents[1] / "data"
        candidates = sorted(default_root.glob("metadrive_data_seed*"))
        if len(candidates) == 1:
            args.demo_dir = str(candidates[0])
            print(f"   ⚠️  自动检测 demo 目录: {args.demo_dir}")
        elif len(candidates) > 1:
            raise RuntimeError(
                "检测到多个 demo 目录，请显式指定 --demo_dir 或 --demo_root，避免误用旧实验数据：\n" +
                "\n".join(str(p) for p in candidates)
            )

    def _stage1b_and_stage2(algo):
        _, algo1b = run_stage1b_bc_updates(trainer, algo, args, log_path)
        return run_stage2_training(trainer, algo1b, log_path, args, train_key)

    if len(trainer.buffer.human_buffer) > 0 or len(trainer.buffer.novice_buffer) > 0:
        print("✅ trainer.buffer 已有 demo 数据，跳过自动重载")
        return _stage1b_and_stage2(algorithm)

    demo_root = getattr(args, "demo_root", None)
    if demo_root:
        root_path = Path(demo_root)
        if not root_path.is_absolute():
            root_path = Path.cwd() / root_path
        if not root_path.exists():
            raise RuntimeError(f"❌ --demo_root 路径不存在: {root_path}")

        print(f"\n📂 使用 --demo_root 递归加载所有演示数据: {root_path}")
        demo_loaded = load_all_demos_recursive(str(root_path), trainer)
        if not demo_loaded:
            raise RuntimeError(f"❌ 从 --demo_root={root_path} 加载演示数据失败")
        print("✅ demo_root 递归加载成功，跳过 Stage1a")
        return _stage1b_and_stage2(algorithm)

    if args.skip_demo_collection:
        if not args.demo_file and not args.demo_dir:
            raise RuntimeError("--skip_demo_collection 需要 --demo_file 或 --demo_dir")

        demo_loaded = False
        if args.demo_dir:
            dp = Path(args.demo_dir)
            if not dp.is_absolute():
                dp = Path.cwd() / dp
            if dp.exists():
                demo_loaded = load_demo_data_directory(str(dp), trainer)
        elif args.demo_file:
            fp = Path(args.demo_file)
            if not fp.is_absolute():
                fp = Path.cwd() / fp
            if fp.exists():
                demo_loaded = load_integrated_demo_data(str(fp), trainer)

        if not demo_loaded:
            raise RuntimeError("加载 demo 数据失败")
        return _stage1b_and_stage2(algorithm)

    if args.force_demo_collection:
        run_stage1a_demo_collection(trainer, args, log_path)
        return _stage1b_and_stage2(algorithm)

    candidate_roots: List[Path] = []
    seen_roots: set = set()
    for root in [args.demo_dir, args.data_dir]:
        if not root:
            continue
        path = Path(root) if Path(root).is_absolute() else Path.cwd() / root
        key = str(path)
        if key not in seen_roots:
            seen_roots.add(key)
            candidate_roots.append(path)

    for root in candidate_roots:
        merged = root / "stage1a_merged_demo.pkl"
        if merged.exists():
            if load_integrated_demo_data(str(merged), trainer):
                print("✅ 合并数据加载成功，跳过 Stage1a")
                return _stage1b_and_stage2(algorithm)

    if args.demo_dir and os.path.exists(args.demo_dir):
        if load_demo_data_directory(args.demo_dir, trainer):
            print("✅ Demo 目录加载成功，跳过 Stage1a")
            return _stage1b_and_stage2(algorithm)

    if args.demo_file and Path(args.demo_file).exists():
        if load_integrated_demo_data(args.demo_file, trainer):
            print("✅ Demo 文件加载成功，跳过 Stage1a")
            return _stage1b_and_stage2(algorithm)

    run_stage1a_demo_collection(trainer, args, log_path)
    return _stage1b_and_stage2(algorithm)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PVP-DACER Training")

    # 网络
    parser.add_argument("--hidden_num", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Critic/Value MLP 宽度；MetaDrive obs 约 260 维，64 太窄，默认 256")
    parser.add_argument("--diffusion_steps", type=int, default=20)
    parser.add_argument("--diffusion_hidden_dim", type=int, default=128,
                        help="Diffusion policy MLP 宽度；默认 128")

    # 训练基础
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--total_step", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha_lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_policy_every", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--log_dir", type=str, default=None)

    # PVP / DACER
    parser.add_argument("--lambda_pv", type=float, default=1.0,
                        help="PV anchor loss 权重；PVP 原论文默认 2.0，此处 1.0 折中")
    parser.add_argument("--B", type=float, default=2.0,
                        help="novice Q 压制幅度；PVP 原论文默认 2.0")
    parser.add_argument("--lambda_bc", type=float, default=12.0, help="Stage2 lambda_bc 初始值（衰减到 0.1 倍）")
    parser.add_argument("--lambda_qreg", type=float, default=0.05,
                        help="Q L2 regularization strength inside PV anchor loss")
    parser.add_argument("--policy_mode", type=str, default="hybrid_dacer",
                        choices=["pvp_paper", "hybrid_dacer"],
                        help="pvp_paper: policy only learns expert-mask diffusion BC")
    parser.add_argument("--lambda_rl", type=float, default=None,
                        help="hybrid_dacer 中 actor 的 RL loss 权重；默认 hybrid=0.1, pvp_paper=0.0")
    parser.add_argument("--lambda_reg", type=float, default=2.0,
                        help="hybrid_dacer 中非专家样本行为正则项权重")
    parser.add_argument("--pre_takeover_window", type=int, default=0,
                        help="Optional ablation only: number of steps before takeover to relabel as soft context. Main method uses 0.")
    parser.add_argument("--pre_takeover_bc_coef", type=float, default=0.0,
                        help="Optional soft-context BC weight for pre-takeover samples. Main method keeps this 0.")
    parser.add_argument("--pre_takeover_pv_coef", type=float, default=0.0,
                        help="Optional soft-context PV ablation weight. Main method keeps this 0 because pre-takeover is not a rejected-action pair.")
    parser.add_argument("--critic_objective", type=str, default="cost",
                        choices=["cost", "risk", "lower", "value", "higher"],
                        help="critic 语义；cost/risk/lower 表示 smaller score is better")
    parser.add_argument("--use_energy_rank", action="store_true", default=False,
                        help="启用 DOVE-ER EnergyRank diffusion-prior reshaping")
    parser.add_argument("--lambda_er", type=float, default=0.0,
                        help="EnergyRank loss 权重")
    parser.add_argument("--er_margin", type=float, default=0.05,
                        help="EnergyRank margin: E_pos + margin < E_neg")
    parser.add_argument("--er_min_action_gap", type=float, default=0.03,
                        help="忽略正负动作过近的 EnergyRank pair")
    parser.add_argument("--er_positive_action", type=str, default="behavior",
                        choices=["behavior", "human"],
                        help="EnergyRank 正样本动作")
    parser.add_argument("--er_use_primal_dual", action="store_true", default=False,
                        help="把 EnergyRank 作为约束 V_E <= epsilon_E，用 primal-dual 自适应 eta_E")
    parser.add_argument("--er_budget", type=float, default=0.0,
                        help="epsilon_E: EnergyRank 平均约束违反预算")
    parser.add_argument("--er_dual_lr", type=float, default=1e-3,
                        help="rho_E: EnergyRank dual multiplier 学习率")
    parser.add_argument("--er_eta_init", type=float, default=-1.0,
                        help="eta_E 初值；负数表示使用 lambda_er")
    parser.add_argument("--er_adaptive_margin", action="store_true", default=False,
                        help="使用 critic gap 校准 m_E(s)，默认关闭，仅作为扩展/消融")
    parser.add_argument("--er_margin_alpha", type=float, default=0.25)
    parser.add_argument("--er_margin_min", type=float, default=0.01)
    parser.add_argument("--er_margin_max", type=float, default=0.20)
    parser.add_argument("--lambda_pv_constraint", type=float, default=0.0,
                        help="critic-side preference constraint hinge 权重")
    parser.add_argument("--pv_constraint_margin", type=float, default=0.1,
                        help="m_C: C(a+)+m_C <= C(a-) 或 Q(a+) >= Q(a-)+m_Q")
    parser.add_argument("--pv_use_primal_dual", action="store_true", default=False,
                        help="critic-side preference constraint 使用 primal-dual")
    parser.add_argument("--pv_constraint_budget", type=float, default=0.0)
    parser.add_argument("--pv_dual_lr", type=float, default=1e-3)
    parser.add_argument("--pv_eta_init", type=float, default=-1.0,
                        help="eta_C 初值；负数表示使用 lambda_pv_constraint")
    parser.add_argument("--rl_gain_clip", type=float, default=0.5,
                        help="hybrid_dacer 中 non-expert Q-gain 的裁剪范围")
    parser.add_argument("--actor_lr", type=float, default=1.5e-4,
                        help="actor 单独学习率；默认 1.5e-4")
    parser.add_argument("--actor_delay", type=int, default=3,
                        help="actor 更新间隔：每 N 次 critic/full update 更新 1 次 actor")
    parser.add_argument("--target_update_delay", type=int, default=2,
                        help="target critics 软更新间隔；与 actor_delay 解耦")
    parser.add_argument("--action_noise_coef", type=float, default=0.0)
    parser.add_argument("--capacity_novice", type=int, default=500000)
    parser.add_argument("--capacity_human", type=int, default=500000)
    parser.add_argument("--stage2_batch_novice_frac", type=float, default=0.75,
                        help="Stage2 full update batch 中 novice/autonomous 样本占比")
    parser.add_argument("--stage2_batch_intervention_frac", type=float, default=0.15,
                        help="Stage2 full update batch 中真实 online intervention 样本占比")
    parser.add_argument("--stage2_batch_pre_takeover_frac", type=float, default=0.0,
                        help="Optional ablation only. Main EnergyRank/PV pair training does not sample pre-takeover as rejected-action supervision.")
    parser.add_argument("--stage2_batch_demo_frac", type=float, default=0.10,
                        help="Stage2 full update batch 中 Stage1a demo 样本占比")
    parser.add_argument("--merge_action_semantics", action="store_true", default=False,
                        help="Data semantics ablation: replace novice/human actions with behavior action")
    parser.add_argument("--disable_stop_td_mask", action="store_true", default=False,
                        help="Data semantics ablation: do not mask TD at takeover/demo boundaries")
    parser.add_argument("--pv_on_all_expert_data", action="store_true", default=False,
                        help="Data semantics ablation: mark demo/pre-takeover/expert data as PV intervention samples")
    parser.add_argument("--allow_missing_novice_action", action="store_true", default=False,
                        help="兼容旧数据开关：缺少 action_novice/action_agent/agent_action 时回退到 action_behavior；默认直接失败以保护 PV/EnergyRank pair 语义")
    parser.add_argument("--use_value_guidance", action="store_true",
                        help="Stage2 启用 UPV-GDS proxy-value-guided diffusion sampling")
    parser.add_argument("--guidance_lambda0", type=float, default=0.5,
                        help="UPV-GDS guidance 基础强度")
    parser.add_argument("--guidance_beta_unc", type=float, default=1.0,
                        help="UPV-GDS critic uncertainty 调节系数")
    parser.add_argument("--guidance_p_decay", type=float, default=1.0,
                        help="UPV-GDS 时间调度幂次: eta_t = alpha_bar_t ** p")
    parser.add_argument("--guidance_grad_clip", type=float, default=1.0,
                        help="UPV-GDS Q 梯度逐元素裁剪范围")
    parser.add_argument("--guidance_mode", type=str, default="proxy_value",
                        choices=["proxy_value", "random_grad", "none", "reverse_grad"],
                        help="UPV-GDS guidance 方向: proxy_value 为符号修正后的主方法, random_grad 负对照, none 零修正")
    parser.add_argument("--guidance_kappa", type=float, default=1.0)
    parser.add_argument("--guidance_q_agg", type=str, default="conservative",
                        choices=["min", "mean", "single_q1", "lcb", "ucb", "conservative"],
                        help="conservative: cost critic uses UCB(C), value critic uses LCB(Q)")
    parser.add_argument("--guidance_injection", type=str, default="clean_x0",
                        choices=["clean_x0", "latent_mean"])
    parser.add_argument("--guidance_schedule", type=str, default="noise_level",
                        choices=["noise_level", "alpha_cumprod"])
    parser.add_argument("--guidance_target", type=str, default="x0",
                        choices=["x0", "xt"])
    parser.add_argument("--guidance_step_interval", type=int, default=1)
    parser.add_argument("--guidance_arm_step", type=int, default=2000,
                        help="Stage2 正式训练 sample_step 达到该值后启用 guidance")

    # 环境
    parser.add_argument("--controller", type=str, default="steering_wheel",
                        choices=["keyboard", "joystick", "gamepad", "steering_wheel", "xbox"])
    parser.add_argument("--manual_control", action="store_true")
    parser.add_argument("--use_render", action="store_true", default=False,
                        help="启用环境渲染；无显示环境下保持关闭以进行 headless 训练")
    parser.add_argument("--start_seed", type=int, default=TRAIN_MAP_SEEDS[0],
                        help="training map pool starts at seed 100")
    parser.add_argument("--num_scenarios", type=int, default=len(TRAIN_MAP_SEEDS),
                        help="number of training maps (fixed at 20: seeds 100–119)")
    parser.add_argument("--traffic_density", type=float, default=0.06)
    parser.add_argument("--env_seed", type=int, default=0)

    # 数据
    parser.add_argument("--demo_file", type=str, default=None,
                        help="单个演示数据文件路径")
    parser.add_argument("--demo_dir", type=str, default=None,
                        help="演示数据目录（递归搜索 seed*.pkl）")
    parser.add_argument("--demo_root", type=str, default=None,
                        help="★ 演示数据根目录：自动递归扫描所有子文件夹中的 .pkl 文件并全部加载到预训练")
    parser.add_argument("--skip_demo_collection", action="store_true")
    parser.add_argument("--force_demo_collection", action="store_true")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--demo_passes_per_map", type=int, default=5)

    # Stage1b
    parser.add_argument("--stage1b_updates", type=int, default=25_000,
                        help="Stage1b 离线 BC 总步数；diffusion policy 收敛需要较多步，默认 50k")
    parser.add_argument("--stage1b_lambda_bc", type=float, default=50.0)
    parser.add_argument("--stage1b_debug_overfit_fixed_batch", action="store_true", default=False)
    parser.add_argument("--stage1b_debug_fixed_batch_size", type=int, default=64)
    parser.add_argument("--allow_stage1b_update_fallback", action="store_true", default=False)

    # Stage2
    parser.add_argument("--stage2_critic_warmup_steps", type=int, default=2000)
    parser.add_argument("--stage2_max_warmup_env_steps", type=int, default=None)
    parser.add_argument("--stage2_warmup_key_offset", type=int, default=10**7)
    parser.add_argument("--stage2_rl_warmup_steps", type=int, default=1000,
                        help="Stage2 RL warmup 步数：此期间只做 BC+Critic，屏蔽 RL loss")
    parser.add_argument("--stage2_rl_ramp_steps", type=int, default=2000,
                        help="RL warmup 结束后的渐入步数：full update 概率从 0%% 线性增到 100%%")
    parser.add_argument("--stage2_bc_decay", dest="stage2_bc_decay", action="store_true", default=True)
    parser.add_argument("--no_stage2_bc_decay", dest="stage2_bc_decay", action="store_false")
    parser.add_argument("--stage2_bc_freeze_steps", type=int, default=3000)
    parser.add_argument("--stage2_min_novice_size", type=int, default=1500)
    parser.add_argument("--stage2_bc_boost_burst_steps", type=int, default=20)
    parser.add_argument("--stage2_bc_boost_burst_batch_size", type=int, default=64)
    parser.add_argument("--stage2_bc_boost_recent_steps", type=int, default=200)
    parser.add_argument("--stage2_bc_boost_trigger_every", type=int, default=200,
                        help="每积累多少条新专家样本触发一次 BC-Boost；默认 200 以降低触发敏感度")
    parser.add_argument("--stage2_bc_boost_steps_per_tick", type=int, default=1)
    parser.add_argument("--stage2_bc_boost_lambda_bc", type=float, default=None,
                        help="BC-Boost 的 lambda_bc；默认跟随当前 Stage2 lambda_bc")
    parser.add_argument("--stage2_bc_boost_lr", type=float, default=None,
                        help="BC-Boost 的学习率；默认跟随当前 actor_lr")
    parser.add_argument("--stage2_bc_boost_ema_alpha", type=float, default=0.2,
                        help="BC-Boost 写回主策略时的 EMA 混合系数")

    parser.add_argument("--print_seed_every_step", action="store_true", default=False,
                        help="每个 env step 都打印当前地图 seed（调试用，日志很多）")
    parser.add_argument("--eval_no_takeover_every", type=int, default=0,
                        help="训练进程内无接管评估间隔；默认 0 关闭，推荐使用 scripts/eval_pvp_policies_fixed_v3.py 离线评估")
    parser.add_argument("--eval_no_takeover_episodes", type=int, default=0,
                        help="训练进程内无接管评估 episode 数；默认 0 关闭")

    args = parser.parse_args()
    configured_train_maps = tuple(range(args.start_seed, args.start_seed + args.num_scenarios))
    if configured_train_maps != TRAIN_MAP_SEEDS:
        parser.error("training maps are fixed to seeds 100–119 (--start_seed 100 --num_scenarios 20)")
    if args.stage2_max_warmup_env_steps is None:
        args.stage2_max_warmup_env_steps = args.stage2_critic_warmup_steps * 10
    if args.lambda_rl is None:
        args.lambda_rl = 0.1 if str(args.policy_mode).lower() == "hybrid_dacer" else 0.0
    if args.actor_lr is None:
        args.actor_lr = args.lr / 5.0
    GUIDANCE_MODE_IDS = {"proxy_value": 0, "random_grad": 1, "none": 2, "reverse_grad": 3}
    GUIDANCE_Q_AGG_IDS = {"min": 0, "mean": 1, "single_q1": 2, "lcb": 3, "ucb": 4, "conservative": 5}
    GUIDANCE_INJECTION_IDS = {"clean_x0": 0, "latent_mean": 1}
    GUIDANCE_SCHEDULE_IDS = {"noise_level": 0, "alpha_cumprod": 1}
    GUIDANCE_TARGET_IDS = {"x0": 0, "xt": 1}
    args.guidance_mode_id = GUIDANCE_MODE_IDS[args.guidance_mode]
    args.guidance_q_agg_id = GUIDANCE_Q_AGG_IDS[args.guidance_q_agg]
    args.guidance_injection_id = GUIDANCE_INJECTION_IDS[args.guidance_injection]
    args.guidance_schedule_id = GUIDANCE_SCHEDULE_IDS[args.guidance_schedule]
    args.guidance_target_id = GUIDANCE_TARGET_IDS[args.guidance_target]
    args.critic_objective_id = 1.0 if args.critic_objective in ("cost", "risk", "lower") else 0.0
    if args.pre_takeover_window > 0 or args.pre_takeover_bc_coef > 0.0 or args.pre_takeover_pv_coef > 0.0 or args.stage2_batch_pre_takeover_frac > 0.0:
        print("⚠️  pre-takeover soft-context ablation enabled; this is not part of the main DOVEER EnergyRank pair objective.")
    if args.use_energy_rank and args.lambda_er <= 0.0:
        args.lambda_er = 1.0
    require_jax_gpu_ready()

    assert args.start_step >= 0
    assert args.total_step >= args.start_step
    assert args.batch_size > 0
    assert args.lambda_pv >= 0
    assert args.B >= 0
    assert args.lambda_rl >= 0
    assert args.lambda_reg >= 0
    assert args.pre_takeover_window >= 0
    assert args.pre_takeover_bc_coef >= 0
    assert args.pre_takeover_pv_coef >= 0
    assert args.lambda_er >= 0
    assert args.er_budget >= 0
    assert args.er_dual_lr >= 0
    assert args.er_margin_min >= 0
    assert args.er_margin_max >= args.er_margin_min
    assert args.lambda_pv_constraint >= 0
    assert args.pv_constraint_margin >= 0
    assert args.pv_constraint_budget >= 0
    assert args.pv_dual_lr >= 0
    args.er_eta_init_resolved = None if args.er_eta_init < 0 else float(args.er_eta_init)
    args.pv_eta_init_resolved = None if args.pv_eta_init < 0 else float(args.pv_eta_init)
    assert args.er_margin >= 0
    assert args.er_min_action_gap >= 0
    assert args.rl_gain_clip > 0
    assert args.actor_lr > 0
    assert args.actor_delay >= 1
    assert args.target_update_delay >= 1
    assert args.stage2_batch_novice_frac >= 0
    assert args.stage2_batch_intervention_frac >= 0
    assert args.stage2_batch_pre_takeover_frac >= 0
    assert args.stage2_batch_demo_frac >= 0
    assert (
        args.stage2_batch_novice_frac
        + args.stage2_batch_intervention_frac
        + args.stage2_batch_pre_takeover_frac
        + args.stage2_batch_demo_frac
    ) > 0
    assert args.demo_passes_per_map >= 1
    assert args.stage1b_lambda_bc > 0
    assert args.stage2_min_novice_size >= 0
    assert args.stage2_bc_freeze_steps >= 0
    assert args.stage2_rl_warmup_steps >= 0
    assert args.stage2_rl_ramp_steps >= 0
    assert args.stage2_bc_boost_burst_steps >= 0
    assert args.stage2_bc_boost_burst_batch_size > 0
    assert args.stage2_bc_boost_recent_steps >= 0
    assert args.stage2_bc_boost_trigger_every >= 0
    assert args.stage2_bc_boost_steps_per_tick >= 1
    assert 0.0 <= args.stage2_bc_boost_ema_alpha <= 1.0
    assert args.guidance_lambda0 >= 0
    assert args.guidance_beta_unc >= 0
    assert args.guidance_p_decay >= 0
    assert args.guidance_grad_clip > 0
    assert args.guidance_kappa >= 0
    assert args.guidance_step_interval >= 0
    assert args.guidance_arm_step >= 0
    assert args.eval_no_takeover_every >= 0
    assert args.eval_no_takeover_episodes >= 0
    assert not (args.skip_demo_collection and args.force_demo_collection), \
        "skip_demo_collection 和 force_demo_collection 不能同时启用"

    sep = "=" * 70
    print(f"\n{sep}")
    print("PVP-DACER Training")
    print(sep)
    print(f"  Seed={args.seed} | Total={args.total_step:,} | Batch={args.batch_size} | LR={args.lr}")
    print(f"  Net: hidden={args.hidden_dim}x{args.hidden_num} | diff_hidden={args.diffusion_hidden_dim} | diff_steps={args.diffusion_steps}")
    print(f"  Stage1b: lambda_bc={args.stage1b_lambda_bc} | updates={args.stage1b_updates:,}")
    if str(args.policy_mode).lower() == "pvp_paper":
        print(f"  Stage2:  lambda_bc={args.lambda_bc}→{args.lambda_bc*0.1} | mode={args.policy_mode} (隐藏 RL actor warmup/ramp 日志)")
    else:
        print(
            f"  Stage2:  lambda_bc={args.lambda_bc}→{args.lambda_bc*0.1}"
            f" | lambda_rl={args.lambda_rl} | lambda_reg={args.lambda_reg}"
            f" | actor_lr={args.actor_lr}"
            f" | actor_delay={args.actor_delay}"
            f" | target_delay={args.target_update_delay}"
            f" | pre_context_window={args.pre_takeover_window}"
            f" | pre_context_BC={args.pre_takeover_bc_coef} | pre_context_PV={args.pre_takeover_pv_coef}"
            f" | critic_objective={args.critic_objective}"
            f" | lambda_er={args.lambda_er}"
            f" | ER_PD={args.er_use_primal_dual} epsE={args.er_budget} eta_lr={args.er_dual_lr}"
            f" | PV_constraint={args.lambda_pv_constraint} PV_PD={args.pv_use_primal_dual}"
            f" | RL warmup={args.stage2_rl_warmup_steps} | RL ramp={args.stage2_rl_ramp_steps}"
            f" | mode={args.policy_mode}"
        )
    print(
        "  Stage2 batch mix:"
        f" novice={args.stage2_batch_novice_frac:.2f}"
        f" | intervention={args.stage2_batch_intervention_frac:.2f}"
        f" | pre={args.stage2_batch_pre_takeover_frac:.2f}"
        f" | demo={args.stage2_batch_demo_frac:.2f}"
        f" | eval_no_takeover_every={args.eval_no_takeover_every}"
        f" | eval_eps={args.eval_no_takeover_episodes}"
    )
    print(f"  PVP: lambda_pv={args.lambda_pv} | B={args.B}")
    print(
        "  UPV-GDS:"
        f" enabled={args.use_value_guidance}"
        f" | lambda0={args.guidance_lambda0}"
        f" | beta_unc={args.guidance_beta_unc}"
        f" | p_decay={args.guidance_p_decay}"
        f" | grad_clip={args.guidance_grad_clip}"
        f" | mode={args.guidance_mode}"
        f" | q_agg={args.guidance_q_agg}"
        f" | kappa={args.guidance_kappa}"
        f" | injection={args.guidance_injection}"
        f" | schedule={args.guidance_schedule}"
        f" | target={args.guidance_target}"
        f" | step_interval={args.guidance_step_interval}"
        f" | arm_step={args.guidance_arm_step}"
    )
    if args.demo_root:
        print(f"  Demo root: {args.demo_root} (递归加载)")
    elif args.demo_dir:
        print(f"  Demo dir: {args.demo_dir}")
    elif args.demo_file:
        print(f"  Demo file: {args.demo_file}")
    print(sep + "\n")

    try:
        clear_jax_cache()
    except Exception:
        pass

    master_rng, _ = seeding(int(args.seed))
    _, _, _, init_network_seed, train_seed = map(int, master_rng.integers(0, 2**32 - 1, 5))
    init_network_key = jax.random.key(init_network_seed)
    train_key = jax.random.key(train_seed)

    env, obs_dim, act_dim = create_human_in_the_loop_pvp_env(
        env_seed=args.env_seed,
        controller=args.controller,
        manual_control=args.manual_control,
        use_render=args.use_render,
        start_seed=args.start_seed,
        num_scenarios=args.num_scenarios,
        traffic_density=args.traffic_density,
    )

    hidden_sizes = [args.hidden_dim] * args.hidden_num
    diffusion_hidden_sizes = [args.diffusion_hidden_dim] * args.hidden_num

    def mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    agent, params = create_dacer_net(
        init_network_key, obs_dim, act_dim,
        hidden_sizes, diffusion_hidden_sizes, mish,
        num_timesteps=args.diffusion_steps,
        action_noise_coef=args.action_noise_coef,
    )

    clear_jax_cache()
    algorithm = PVPDACER(
        agent=agent, params=params,
        gamma=args.gamma, tau=args.tau, lr=args.lr, alpha_lr=args.alpha_lr,
        delay_alpha_update=10000, delay_update=2,
        actor_delay=args.actor_delay,
        target_update_delay=args.target_update_delay,
        reward_scale=1.0,
        num_samples=200,
        actor_lr=args.actor_lr,
        lambda_pv=args.lambda_pv, B=args.B, lambda_bc=args.lambda_bc,
        reward_free=True,
        lambda_qreg=args.lambda_qreg,
        policy_mode=args.policy_mode,
        lambda_rl=args.lambda_rl,
        lambda_reg=args.lambda_reg,
        rl_gain_clip=args.rl_gain_clip,
        pre_takeover_bc_coef=args.pre_takeover_bc_coef,
        pre_takeover_pv_coef=args.pre_takeover_pv_coef,
        critic_objective=args.critic_objective,
        lambda_er=args.lambda_er,
        er_margin=args.er_margin,
        er_min_action_gap=args.er_min_action_gap,
        er_positive_action=args.er_positive_action,
        er_use_primal_dual=args.er_use_primal_dual,
        er_budget=args.er_budget,
        er_dual_lr=args.er_dual_lr,
        er_eta_init=args.er_eta_init_resolved,
        er_adaptive_margin=args.er_adaptive_margin,
        er_margin_alpha=args.er_margin_alpha,
        er_margin_min=args.er_margin_min,
        er_margin_max=args.er_margin_max,
        lambda_pv_constraint=args.lambda_pv_constraint,
        pv_constraint_margin=args.pv_constraint_margin,
        pv_use_primal_dual=args.pv_use_primal_dual,
        pv_constraint_budget=args.pv_constraint_budget,
        pv_dual_lr=args.pv_dual_lr,
        pv_eta_init=args.pv_eta_init_resolved,
    )

    _detect_pvpdacer_methods(algorithm)

    buffer = PVPBalancedDualBuffer(
        obs_shape=(obs_dim,), action_shape=(act_dim,),
        capacity_novice=args.capacity_novice, capacity_human=args.capacity_human,
        novice_fraction=args.stage2_batch_novice_frac,
        intervention_fraction=args.stage2_batch_intervention_frac,
        pre_takeover_fraction=args.stage2_batch_pre_takeover_frac,
        demo_fraction=args.stage2_batch_demo_frac,
    )

    project_root = Path(__file__).resolve().parents[1]
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_id = f"pvp_dacer_{timestamp}_s{args.seed}"

    if args.log_dir:
        log_root = Path(args.log_dir)
        log_path = log_root / run_id
    else:
        log_path = project_root / "logs" / run_id

    log_path.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir) if args.data_dir else (log_path / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    args.log_dir = str(log_path)
    args.data_dir = str(data_dir)

    tb_writer = _create_tb_logger(log_path)
    tb_logger = TBLoggerAdapter(tb_writer)
    tb_logger.add_scalars(
        "hparams/lambda",
        {
            "lambda_bc_stage2_init": float(args.lambda_bc),
            "lambda_bc_stage2_min": float(args.lambda_bc * 0.1),
            "lambda_pv": float(args.lambda_pv),
            "B": float(args.B),
            "lambda_rl": float(args.lambda_rl),
            "lambda_reg": float(args.lambda_reg),
            "pre_takeover_bc_coef": float(args.pre_takeover_bc_coef),
            "pre_takeover_pv_coef": float(args.pre_takeover_pv_coef),
            "critic_objective_id": float(args.critic_objective_id),
            "use_energy_rank": 1.0 if args.use_energy_rank else 0.0,
            "lambda_er": float(args.lambda_er),
            "er_margin": float(args.er_margin),
            "er_min_action_gap": float(args.er_min_action_gap),
            "er_use_primal_dual": 1.0 if args.er_use_primal_dual else 0.0,
            "er_budget": float(args.er_budget),
            "er_dual_lr": float(args.er_dual_lr),
            "er_adaptive_margin": 1.0 if args.er_adaptive_margin else 0.0,
            "er_margin_alpha": float(args.er_margin_alpha),
            "er_margin_min": float(args.er_margin_min),
            "er_margin_max": float(args.er_margin_max),
            "lambda_pv_constraint": float(args.lambda_pv_constraint),
            "pv_constraint_margin": float(args.pv_constraint_margin),
            "pv_use_primal_dual": 1.0 if args.pv_use_primal_dual else 0.0,
            "pv_constraint_budget": float(args.pv_constraint_budget),
            "pv_dual_lr": float(args.pv_dual_lr),
            "rl_gain_clip": float(args.rl_gain_clip),
            "actor_lr": float(args.actor_lr),
            "batch_novice_frac": float(args.stage2_batch_novice_frac),
            "batch_intervention_frac": float(args.stage2_batch_intervention_frac),
            "batch_pre_takeover_frac": float(args.stage2_batch_pre_takeover_frac),
            "batch_demo_frac": float(args.stage2_batch_demo_frac),
            "allow_missing_novice_action": 1.0 if args.allow_missing_novice_action else 0.0,
            "use_value_guidance": 1.0 if args.use_value_guidance else 0.0,
            "guidance_lambda0": float(args.guidance_lambda0),
            "guidance_beta_unc": float(args.guidance_beta_unc),
            "guidance_p_decay": float(args.guidance_p_decay),
            "guidance_grad_clip": float(args.guidance_grad_clip),
            "guidance_mode_id": float(args.guidance_mode_id),
            "guidance_q_agg_id": float(args.guidance_q_agg_id),
            "guidance_kappa": float(args.guidance_kappa),
            "guidance_injection_id": float(args.guidance_injection_id),
            "guidance_schedule_id": float(args.guidance_schedule_id),
            "guidance_target_id": float(args.guidance_target_id),
            "guidance_step_interval": float(args.guidance_step_interval),
        },
        step=0,
    )
    tb_logger.add_scalars(
        "hparams/stage2",
        {
            "rl_warmup_steps": float(args.stage2_rl_warmup_steps),
            "rl_ramp_steps": float(args.stage2_rl_ramp_steps),
            "critic_warmup_steps": float(args.stage2_critic_warmup_steps),
            "actor_delay": float(args.actor_delay),
            "target_update_delay": float(args.target_update_delay),
            "pre_takeover_window": float(args.pre_takeover_window),
            "bc_decay_enabled": 1.0 if args.stage2_bc_decay else 0.0,
            "reward_free": 1.0,
            "guidance_arm_step": float(args.guidance_arm_step),
            "eval_no_takeover_every": float(args.eval_no_takeover_every),
            "eval_no_takeover_episodes": float(args.eval_no_takeover_episodes),
        },
        step=0,
    )

    if args.manual_control:
        env = SharedControlMonitor(
            env,
            folder=str(data_dir / "recorded_data"),
            prefix=f"pvp_{args.controller}",
            save_freq=10**9,
            pre_takeover_window=int(args.pre_takeover_window),
        )

        install_strict_fixed_seed_controller(
            env,
            get_fixed_stage_map_seeds(),
            mode="cycle",
            start_index=0,
            rng_seed=int(args.seed) + 999,
        )

    trainer = PVPOffPolicyTrainer(
        env=env, algorithm=algorithm, buffer=buffer,
        log_path=log_path, batch_size=args.batch_size,
        start_step=args.start_step, total_step=args.total_step,
        sample_per_iteration=1, update_per_iteration=1,
        save_policy_every=args.save_policy_every,
        warmup_with="random", intervention_callback=None,
        reward_free=True,
    )
    trainer.logger = tb_logger
    trainer.pre_takeover_window = int(args.pre_takeover_window)
    trainer.merge_action_semantics = bool(args.merge_action_semantics)
    trainer.disable_stop_td_mask = bool(args.disable_stop_td_mask)
    trainer.pv_on_all_expert_data = bool(args.pv_on_all_expert_data)
    trainer.allow_missing_novice_action = bool(args.allow_missing_novice_action)
    trainer.no_takeover_eval_env = None
    trainer.no_takeover_eval_allowed_seeds = ()
    _install_stage2_human_counter(trainer)

    try:
        trainer.map_seed_rng = np.random.default_rng(int(args.seed) + 2026)
    except Exception:
        trainer.map_seed_rng = None
    trainer.fixed_stage_map_seeds = get_fixed_stage_map_seeds()

    with open(log_path / "args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    dummy_data = Experience.create_example(obs_dim, act_dim, trainer.batch_size)
    trainer.setup(dummy_data)

    integrated_demo = data_dir / "integrated_demo_data.pkl"
    if not args.force_demo_collection and integrated_demo.exists() and args.start_step == 0:
        try:
            with open(integrated_demo, "rb") as f:
                demo_data = pickle.load(f)
            manager = DemoDataManager()
            if manager.load_demo_data_to_trainer(trainer, demo_data):
                print(f"✅ 集成演示加载成功: human={len(trainer.buffer.human_buffer)}")
        except Exception as e:
            print(f"⚠️  集成演示加载失败: {e}")

    assert hasattr(trainer, "sample_human_demo"), "trainer.sample_human_demo not found"
    assert hasattr(trainer.buffer, "sample_human_only"), "buffer.sample_human_only not found"

    final_step = run_three_stage_training(args, trainer, algorithm, log_path, train_key)

    if final_step > 0:
        print(f"\n{'=' * 70}")
        print(f"🎉 训练完成 | step={final_step} | {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   📁 {log_path}")
        print(f"   📊 tensorboard --logdir {log_path / 'tb'}")
    else:
        print("\n⏭️  训练完成（Stage2 未执行或 total_step=0）")


if __name__ == "__main__":
    main()
