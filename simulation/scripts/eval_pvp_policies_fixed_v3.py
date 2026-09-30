"""
PVP-DACER 批量策略评估脚本 v4（严格对齐训练环境）

修正重点：
  - 严格对齐训练脚本的环境配置与 reset 行为
  - horizon 默认 1000（与训练一致）
  - 环境配置与训练一致（训练地图 100–119）
  - vehicle_config 与训练一致
  - 默认 use_render=False，便于 headless 离线评估
  - 默认执行 no-takeover 评估，专门测自主闭环：
      manual_control=False
      enable_takeover=False
      out_of_route_done=True
      crash_done=True
  - 移植训练里的严格固定 seed 控制：
      allowed_scenarios
      禁用 sequential_reset
      reset 后同步 current_seed / last_reset_info
  - 过滤 .meta.pkl
  - 每个 episode 打印终止原因
"""

import os
import sys
import re
import json
import csv
import time
import pickle
import argparse
import traceback
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("JAX_ENABLE_X64", "false")
os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "float32")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.5")
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

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "float32")
jax.config.update("jax_enable_x64", False)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from _local_bootstrap import ensure_project_imports
except ModuleNotFoundError:
    from scripts._local_bootstrap import ensure_project_imports
_PROJECT_ROOT = str(ensure_project_imports(__file__))

from relax_env.human_in_the_loop_env import HumanInTheLoopEnv
from relax.network.dacer import create_dacer_net
from relax.utils.jax_utils import random_key_from_data
from map_splits import TEST_MAP_SEEDS, TRAIN_MAP_SEEDS

EVAL_MAPS_DEFAULT = list(TEST_MAP_SEEDS)
ACT_DIM = 2
GUIDANCE_MODE_IDS = {"proxy_value": 0, "random_grad": 1, "none": 2, "reverse_grad": 3}
GUIDANCE_TARGET_IDS = {"x0": 0, "xt": 1}
GUIDANCE_Q_AGG_IDS = {"min": 0, "mean": 1, "single_q1": 2, "lcb": 3, "ucb": 4, "conservative": 5}
GUIDANCE_INJECTION_IDS = {"clean_x0": 0, "latent_mean": 1}
GUIDANCE_SCHEDULE_IDS = {"noise_level": 0, "alpha_cumprod": 1}
CRITIC_OBJECTIVE_IDS = {
    "value": 0,
    "higher": 0,
    "higher_is_better": 0,
    "cost": 1,
    "risk": 1,
    "lower": 1,
    "lower_is_better": 1,
}


def require_jax_gpu_ready() -> Tuple[str, List[Any]]:
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
            f"JAX GPU backend is required for this project, but active backend is {backend!r}. "
            "Do not run this evaluation with CPU backend."
        )
    return backend, devices


def _create_tb_writer(tb_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
        tb_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(log_dir=str(tb_dir))
    except Exception:
        pass
    try:
        from tensorboardX import SummaryWriter
        tb_dir.mkdir(parents=True, exist_ok=True)
        return SummaryWriter(logdir=str(tb_dir))
    except Exception:
        return None


def _tb_add_scalar(writer, tag: str, value: Any, step: int) -> None:
    if writer is None:
        return
    try:
        value_f = float(value)
        if np.isfinite(value_f):
            writer.add_scalar(tag, value_f, global_step=int(step))
    except Exception:
        pass


def _tb_safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(name)).strip("_") or "policy"


def sanitize_sigma_ref(raw_sigma_ref: Any) -> jax.Array:
    raw_sigma_ref = jnp.asarray(raw_sigma_ref, dtype=jnp.float32)
    finite_positive = jnp.logical_and(jnp.isfinite(raw_sigma_ref), raw_sigma_ref > 0.0)
    return jnp.where(finite_positive, raw_sigma_ref, jnp.float32(1.0))


# ============================================================================
# Utility
# ============================================================================

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))


def load_run_args(log_dir: Path) -> Dict[str, Any]:
    candidates: List[Path] = []
    root_args = log_dir / "args.json"
    if root_args.exists():
        candidates.append(root_args)

    bundle_candidates = sorted(log_dir.glob("final_bundle_step_*/args.json"))
    if bundle_candidates:
        def _bundle_step(p: Path) -> int:
            m = re.search(r"final_bundle_step_(\d+)", str(p))
            return int(m.group(1)) if m else -1
        candidates.extend(sorted(bundle_candidates, key=_bundle_step, reverse=True))

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"Loaded training args from: {path}")
            return data
        except Exception as e:
            print(f"WARNING: failed to read {path}: {e}")

    print("WARNING: args.json not found, fallback to default network/env config")
    return {}


def find_run_args_dir(preferred_dir: Path, policy_files: Optional[List[Path]] = None) -> Path:
    """Pick the nearest directory that contains training args.json."""
    candidates: List[Path] = []

    def _add(path: Path):
        try:
            path = path.resolve()
        except Exception:
            pass
        if path not in candidates:
            candidates.append(path)

    if preferred_dir.is_file():
        _add(preferred_dir.parent)
    else:
        _add(preferred_dir)

    for policy in policy_files or []:
        _add(policy.parent)
        _add(policy.parent.parent)

    for path in candidates:
        if (path / "args.json").exists():
            return path
        try:
            if any(path.glob("final_bundle_step_*/args.json")):
                return path
        except Exception:
            pass

    return candidates[0] if candidates else preferred_dir


def build_agent(obs_dim: int, run_args: Dict[str, Any]):
    hidden_num = int(run_args.get("hidden_num", 2))
    hidden_dim = int(run_args.get("hidden_dim", 256))
    diffusion_hidden_dim = int(run_args.get("diffusion_hidden_dim", 128))
    diffusion_steps = int(run_args.get("diffusion_steps", 20))
    action_noise_coef = float(run_args.get("action_noise_coef", 0.0))

    hidden_sizes = [hidden_dim] * hidden_num
    diff_hidden_sizes = [diffusion_hidden_dim] * hidden_num

    init_key = jax.random.key(0)
    agent, _ = create_dacer_net(
        init_key, obs_dim, ACT_DIM,
        hidden_sizes, diff_hidden_sizes, mish,
        num_timesteps=diffusion_steps,
        action_noise_coef=action_noise_coef,
    )
    return agent


# ============================================================================
# Environment creation (strictly aligned with training Stage2)
# ============================================================================

def create_eval_env(
    horizon: int = 1000,
    run_args: Optional[Dict[str, Any]] = None,
    use_render: bool = False,
) -> HumanInTheLoopEnv:
    """
    Align with training script:
      create_human_in_the_loop_pvp_env(...)
      + Stage2 extra config updates.
    """
    run_args = run_args or {}

    env_config = {
        # training-aligned basic config
        "start_seed": int(run_args.get("start_seed", TRAIN_MAP_SEEDS[0])),
        "num_scenarios": int(run_args.get("num_scenarios", len(TRAIN_MAP_SEEDS))),
        "traffic_density": float(run_args.get("traffic_density", 0.06)),
        "horizon": int(horizon),
        "controller": str(run_args.get("controller", "steering_wheel")),
        "vehicle_config": {
            "show_lidar": False,
            "show_side_detector": False,
            "show_lane_line_detector": False,
        },
        "show_logo": False,
        "show_fps": True,

        # Stage2-aligned runtime behavior
        "manual_control": False,
        "enable_takeover": False,
        "out_of_route_done": True,
        "crash_done": True,

        # 外部评估默认禁用接管，专门测试自主闭环表现
        "use_render": bool(use_render),
    }

    return HumanInTheLoopEnv(env_config)


def infer_dt(env: HumanInTheLoopEnv, cli_dt: Optional[float]) -> float:
    if cli_dt is not None:
        return float(cli_dt)

    candidates = []
    try:
        cfg = _safe_local_attr(env, "config", None)
        if isinstance(cfg, dict):
            for key in ("dt", "time_step", "physics_world_step_size"):
                if key in cfg:
                    candidates.append(cfg[key])
    except Exception:
        pass

    try:
        engine = getattr(env, "engine", None)
        if engine is not None:
            for key in ("dt", "time_step", "physics_world_step_size"):
                if hasattr(engine, key):
                    candidates.append(getattr(engine, key))
    except Exception:
        pass

    for v in candidates:
        try:
            v = float(v)
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            pass

    print("WARNING: unable to infer dt automatically, fallback to 0.02")
    return 0.02


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


# ============================================================================
# Strict fixed-seed controller (ported from training)
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


def install_strict_eval_seed_controller(env, allowed_seeds: List[int]) -> None:
    allowed_seeds = [int(x) for x in allowed_seeds]
    if not allowed_seeds:
        raise ValueError("allowed_seeds cannot be empty")
    _apply_fixed_seed_config(env, allowed_seeds)
    _disable_env_sequential_reset(env)


def reset_to_strict_eval_map(env, seed: int, allowed_seeds: Optional[List[int]] = None):
    seed = int(seed)
    if allowed_seeds is None:
        allowed_seeds = [seed]
    allowed_seeds = [int(x) for x in allowed_seeds]

    if seed not in allowed_seeds:
        raise ValueError(f"seed={seed} is not in allowed_seeds={allowed_seeds}")

    _apply_fixed_seed_config(env, allowed_seeds)
    _disable_env_sequential_reset(env)

    obs, info = env.reset(seed=seed)
    _sync_env_current_seed(env, seed)

    actual_seed = None
    for key in ("current_seed", "seed", "scenario_seed", "map_seed"):
        if key in info:
            try:
                actual_seed = int(info[key])
                break
            except Exception:
                pass

    if actual_seed is not None and actual_seed != seed:
        print(f"WARNING: seed mismatch, requested={seed}, actual={actual_seed}")

    return obs, info


# ============================================================================
# Info extraction
# ============================================================================

def _safe_float(info: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in info:
            try:
                v = info[key]
                if isinstance(v, (list, tuple, np.ndarray)):
                    arr = np.asarray(v, dtype=np.float32)
                    if arr.size == 1:
                        return float(arr.reshape(-1)[0])
                    continue
                return float(v)
            except Exception:
                continue
    return None


def get_speed_mps(info: Dict[str, Any]) -> float:
    speed = _safe_float(info, "speed")
    if speed is not None:
        return speed

    speed_kmh = _safe_float(info, "speed_kmh")
    if speed_kmh is not None:
        return speed_kmh / 3.6

    velocity = info.get("velocity", None)
    if velocity is not None:
        try:
            arr = np.asarray(velocity, dtype=np.float32)
            if arr.ndim == 0:
                return float(arr)
            if arr.size >= 2:
                return float(np.linalg.norm(arr[:2]))
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
        except Exception:
            pass

    return 0.0


# ============================================================================
# Policy load / action adapter
# ============================================================================

def load_policy_payload(policy_pkl: Path) -> Optional[Any]:
    try:
        with open(policy_pkl, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"ERROR: failed to load {policy_pkl.name}: {e}")
        return None


def _extract_log_alpha_from_payload(payload: Any):
    try:
        if hasattr(payload, "log_alpha"):
            return payload.log_alpha
    except Exception:
        pass
    if isinstance(payload, dict) and "log_alpha" in payload:
        return payload["log_alpha"]
    if isinstance(payload, (list, tuple)) and len(payload) >= 2:
        return payload[1]
    return None


def _extract_policy_only_from_payload(payload: Any):
    try:
        if hasattr(payload, "policy"):
            return payload.policy
    except Exception:
        pass
    if isinstance(payload, dict) and "policy" in payload:
        return payload["policy"]
    if isinstance(payload, (list, tuple)) and len(payload) >= 1:
        return payload[0]
    return payload


def _candidate_policy_objs(payload: Any) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    seen = set()

    def _add(tag: str, obj: Any):
        k = id(obj)
        if k in seen:
            return
        seen.add(k)
        out.append((tag, obj))

    _add("raw", payload)
    policy_only = _extract_policy_only_from_payload(payload)
    _add("policy_only", policy_only)

    log_alpha = _extract_log_alpha_from_payload(payload)
    if log_alpha is not None:
        try:
            _add("policy_plus_log_alpha", (policy_only, log_alpha))
        except Exception:
            pass

    return out


def _extract_guidance_components_from_payload(payload: Any):
    policy_only = _extract_policy_only_from_payload(payload)
    log_alpha = _extract_log_alpha_from_payload(payload)
    if log_alpha is None:
        return None

    online_q1 = online_q2 = None
    target_q1 = target_q2 = None
    mean_q1_std = mean_q2_std = None
    sigma_ref = None
    version = "legacy"
    critic_objective_id = -1

    if isinstance(payload, dict):
        online_q1 = payload.get("q1")
        online_q2 = payload.get("q2")
        target_q1 = payload.get("target_q1", payload.get("q1"))
        target_q2 = payload.get("target_q2", payload.get("q2"))
        mean_q1_std = payload.get("mean_q1_std", None)
        mean_q2_std = payload.get("mean_q2_std", None)
        sigma_ref = payload.get("sigma_ref", None)
        version = str(payload.get("version", version))
        if "critic_objective_id" in payload:
            try:
                critic_objective_id = int(float(payload["critic_objective_id"]) > 0.5)
            except Exception:
                critic_objective_id = -1
        elif "critic_objective" in payload:
            critic_objective_id = CRITIC_OBJECTIVE_IDS.get(str(payload["critic_objective"]).lower(), -1)
    elif isinstance(payload, (list, tuple)) and len(payload) >= 4:
        online_q1 = payload[2]
        online_q2 = payload[3]
        target_q1 = payload[2]
        target_q2 = payload[3]

    if target_q1 is None or target_q2 is None:
        return None
    if online_q1 is None or online_q2 is None:
        online_q1, online_q2 = target_q1, target_q2

    if sigma_ref is None and mean_q1_std is not None and mean_q2_std is not None:
        sigma_ref = 0.5 * (jnp.asarray(mean_q1_std) + jnp.asarray(mean_q2_std))
    if sigma_ref is None:
        sigma_ref = jnp.float32(1.0)

    sigma_ref = sanitize_sigma_ref(sigma_ref)
    return {
        "policy_obj": (policy_only, log_alpha),
        "q_params": (target_q1, target_q2),
        "target_q_params": (target_q1, target_q2),
        "online_q_params": (online_q1, online_q2),
        "sigma_ref": sigma_ref,
        "version": version,
        "critic_objective_id": critic_objective_id,
        "has_target_critics": isinstance(payload, dict) and "target_q1" in payload and "target_q2" in payload,
    }


# ============================================================================
# JIT inference
# ============================================================================

def _build_jit_action_fn(agent, obs_mode: str, guidance_cfg: Optional[Dict[str, Any]] = None):
    guidance_cfg = guidance_cfg or {}
    use_guidance = bool(guidance_cfg.get("use_value_guidance", False))
    if int(guidance_cfg.get("rerank_topk", 0) or 0) > 1:
        return None

    if use_guidance and obs_mode == "batched":
        lambda_0 = jnp.float32(guidance_cfg["lambda_0"])
        beta_unc = jnp.float32(guidance_cfg["beta_unc"])
        p_decay = jnp.float32(guidance_cfg["p_decay"])
        grad_clip = jnp.float32(guidance_cfg["grad_clip"])
        guidance_mode = jnp.int32(guidance_cfg["guidance_mode_id"])
        guidance_target = jnp.int32(guidance_cfg["guidance_target_id"])
        guidance_step_interval = jnp.int32(guidance_cfg["guidance_step_interval"])
        guidance_q_agg = jnp.int32(guidance_cfg["guidance_q_agg_id"])
        guidance_kappa = jnp.float32(guidance_cfg.get("guidance_kappa", 1.0))
        guidance_injection = jnp.int32(guidance_cfg.get("guidance_injection_id", 0))
        guidance_schedule = jnp.int32(guidance_cfg.get("guidance_schedule_id", 0))
        critic_objective = jnp.float32(guidance_cfg.get("critic_objective_id", 1))
        project_guidance = bool(guidance_cfg.get("final_gate_project_guidance", bool(guidance_cfg.get("use_final_gate", False))))
        max_action_shift = jnp.float32(guidance_cfg.get("final_gate_max_action_shift", -1.0))

        @jax.jit
        def _jit_action(policy_obj, q_params, sigma_ref, obs):
            obs_batched = obs[None, :]
            key = random_key_from_data(obs_batched)
            anchor = agent.get_deterministic_action(policy_obj, obs_batched) if project_guidance else None
            act, info = agent.get_deterministic_action_guided_with_metrics(
                key, policy_obj, q_params, obs_batched, sigma_ref,
                lambda_0, beta_unc, p_decay, grad_clip, guidance_mode,
                guidance_target, guidance_step_interval, guidance_q_agg,
                guidance_kappa, guidance_injection, guidance_schedule,
                critic_objective,
                anchor_action=anchor,
                max_action_shift=max_action_shift,
            )
            return act[0], info
        return _jit_action

    if use_guidance:
        lambda_0 = jnp.float32(guidance_cfg["lambda_0"])
        beta_unc = jnp.float32(guidance_cfg["beta_unc"])
        p_decay = jnp.float32(guidance_cfg["p_decay"])
        grad_clip = jnp.float32(guidance_cfg["grad_clip"])
        guidance_mode = jnp.int32(guidance_cfg["guidance_mode_id"])
        guidance_target = jnp.int32(guidance_cfg["guidance_target_id"])
        guidance_step_interval = jnp.int32(guidance_cfg["guidance_step_interval"])
        guidance_q_agg = jnp.int32(guidance_cfg["guidance_q_agg_id"])
        guidance_kappa = jnp.float32(guidance_cfg.get("guidance_kappa", 1.0))
        guidance_injection = jnp.int32(guidance_cfg.get("guidance_injection_id", 0))
        guidance_schedule = jnp.int32(guidance_cfg.get("guidance_schedule_id", 0))
        critic_objective = jnp.float32(guidance_cfg.get("critic_objective_id", 1))
        project_guidance = bool(guidance_cfg.get("final_gate_project_guidance", bool(guidance_cfg.get("use_final_gate", False))))
        max_action_shift = jnp.float32(guidance_cfg.get("final_gate_max_action_shift", -1.0))

        @jax.jit
        def _jit_action(policy_obj, q_params, sigma_ref, obs):
            key = random_key_from_data(obs)
            anchor = agent.get_deterministic_action(policy_obj, obs) if project_guidance else None
            act, info = agent.get_deterministic_action_guided_with_metrics(
                key, policy_obj, q_params, obs, sigma_ref,
                lambda_0, beta_unc, p_decay, grad_clip, guidance_mode,
                guidance_target, guidance_step_interval, guidance_q_agg,
                guidance_kappa, guidance_injection, guidance_schedule,
                critic_objective,
                anchor_action=anchor,
                max_action_shift=max_action_shift,
            )
            if act.ndim == 2:
                act = act[0]
            return act, info
        return _jit_action

    if obs_mode == "batched":
        @jax.jit
        def _jit_action(policy_obj, obs):
            obs_batched = obs[None, :]
            act = agent.get_deterministic_action(policy_obj, obs_batched)
            return act[0]
        return _jit_action
    else:
        @jax.jit
        def _jit_action(policy_obj, obs):
            act = agent.get_deterministic_action(policy_obj, obs)
            if act.ndim == 2:
                act = act[0]
            return act
        return _jit_action



def _build_fast_final_gate_jit_fn(agent, guidance_cfg: Dict[str, Any]):
    """Single XLA program for UPV-GDS + Final Gate.

    The old Python path computed vanilla action, guided action, critic bounds,
    and diffusion-prior energy as separate Python/JAX calls at every env step.
    That forces many GPU synchronizations and makes final-gate evaluation much
    slower than vanilla/guided-only rollout.  This function fuses the whole gate
    into one jitted call per environment step.
    """
    lambda_0 = jnp.float32(guidance_cfg["lambda_0"])
    beta_unc = jnp.float32(guidance_cfg["beta_unc"])
    p_decay = jnp.float32(guidance_cfg["p_decay"])
    grad_clip = jnp.float32(guidance_cfg["grad_clip"])
    guidance_mode = jnp.int32(guidance_cfg["guidance_mode_id"])
    guidance_target = jnp.int32(guidance_cfg["guidance_target_id"])
    guidance_step_interval = jnp.int32(guidance_cfg["guidance_step_interval"])
    guidance_q_agg = jnp.int32(guidance_cfg["guidance_q_agg_id"])
    guidance_kappa = jnp.float32(guidance_cfg.get("guidance_kappa", 1.0))
    guidance_injection = jnp.int32(guidance_cfg.get("guidance_injection_id", 0))
    guidance_schedule = jnp.int32(guidance_cfg.get("guidance_schedule_id", 0))
    critic_objective = jnp.float32(guidance_cfg.get("critic_objective_id", 1))
    project_guidance = bool(guidance_cfg.get("final_gate_project_guidance", True))

    mode = str(guidance_cfg.get("final_gate_mode", "proxy_prior"))
    compare = str(guidance_cfg.get("final_gate_proxy_compare", "robust"))
    lower_is_better = int(guidance_cfg.get("critic_objective_id", 1)) == 1
    uses_prior = mode in ("proxy_prior", "prior_only")
    uses_proxy = mode in ("proxy_prior", "proxy_only")

    final_gate_margin = jnp.float32(guidance_cfg.get("final_gate_margin", 0.0))
    proxy_min_gain = jnp.float32(guidance_cfg.get("final_gate_min_proxy_gain", 0.0))
    kappa = jnp.float32(guidance_cfg.get("final_gate_kappa", 1.0))
    beta_prior = jnp.float32(guidance_cfg.get("final_gate_beta_prior", 1.0))
    rho_shift = jnp.float32(guidance_cfg.get("final_gate_rho_shift", 0.0))
    rho_sat = jnp.float32(guidance_cfg.get("final_gate_rho_sat", 0.0))
    max_action_shift = jnp.float32(guidance_cfg.get("final_gate_max_action_shift", 0.2))
    tau_prior = jnp.float32(guidance_cfg.get("final_gate_tau_prior", 0.02))
    tau_u = jnp.float32(guidance_cfg.get("final_gate_tau_u", 1.0))
    energy_alpha = jnp.float32(guidance_cfg.get("final_gate_energy_alpha", 1.0))
    energy_samples = max(1, int(guidance_cfg.get("final_gate_energy_samples", 4)))
    energy_timesteps = jnp.asarray(
        np.linspace(0, max(int(agent.num_timesteps) - 1, 0), energy_samples, dtype=np.int32),
        dtype=jnp.int32,
    )
    energy_indices = jnp.arange(energy_samples, dtype=jnp.int32)
    compare_mode_id = jnp.float32({
        "robust": 0.0,
        "tex_lcb": 4.0,
        "strict": 4.0,
        "mean": 1.0,
        "guided_ucb_vs_vanilla_mean": 2.0,
        "mean_penalty": 3.0,
    }.get(compare, -1.0))

    def _q_bounds_jax(q_params, obs_b, act_b):
        q1m, _ = agent.q(q_params[0], obs_b, act_b)
        q2m, _ = agent.q(q_params[1], obs_b, act_b)
        q1 = jnp.ravel(q1m)[0]
        q2 = jnp.ravel(q2m)[0]
        qbar = jnp.float32(0.5) * (q1 + q2)
        lcb = jnp.minimum(q1, q2)
        ucb = jnp.maximum(q1, q2)
        uq = ucb - lcb
        return lcb, ucb, qbar, uq

    @jax.jit
    def _final_gate_action(policy_obj, q_params, sigma_ref, obs):
        obs_b = obs[None, :]
        key = random_key_from_data(obs_b)

        vanilla_b = agent.get_deterministic_action(policy_obj, obs_b)
        guided_b, guidance_info = agent.get_deterministic_action_guided_with_metrics(
            key,
            policy_obj,
            q_params,
            obs_b,
            sigma_ref,
            lambda_0,
            beta_unc,
            p_decay,
            grad_clip,
            guidance_mode,
            guidance_target,
            guidance_step_interval,
            guidance_q_agg,
            guidance_kappa,
            guidance_injection,
            guidance_schedule,
            critic_objective,
            anchor_action=vanilla_b if project_guidance else None,
            max_action_shift=max_action_shift,
        )
        vanilla = jnp.clip(jnp.reshape(vanilla_b, (-1, ACT_DIM))[0], -1.0, 1.0)
        guided = jnp.clip(jnp.reshape(guided_b, (-1, ACT_DIM))[0], -1.0, 1.0)
        vanilla_b = vanilla[None, :]
        guided_b = guided[None, :]

        lcb_g, ucb_g, qbar_g, uq_g = _q_bounds_jax(q_params, obs_b, guided_b)
        lcb_0, ucb_0, qbar_0, uq_0 = _q_bounds_jax(q_params, obs_b, vanilla_b)

        if lower_is_better:
            proxy_gain_lcb = lcb_0 - lcb_g
            proxy_gain_robust = lcb_0 - ucb_g
            proxy_gain_mean = qbar_0 - qbar_g
            proxy_gain_mean_penalty = proxy_gain_mean - jnp.float32(0.5) * kappa * uq_g
            if compare == "robust":
                proxy_gain = proxy_gain_robust
            elif compare in ("tex_lcb", "strict"):
                proxy_gain = proxy_gain_lcb
            elif compare == "mean":
                proxy_gain = proxy_gain_mean
            elif compare == "guided_ucb_vs_vanilla_mean":
                proxy_gain = qbar_0 - ucb_g
            elif compare == "mean_penalty":
                proxy_gain = proxy_gain_mean_penalty
            else:
                proxy_gain = proxy_gain_lcb
        else:
            proxy_gain_lcb = lcb_g - lcb_0
            proxy_gain_robust = lcb_g - ucb_0
            proxy_gain_mean = qbar_g - qbar_0
            proxy_gain_mean_penalty = proxy_gain_mean - jnp.float32(0.5) * kappa * uq_g
            if compare == "robust":
                proxy_gain = proxy_gain_robust
            elif compare in ("tex_lcb", "strict"):
                proxy_gain = proxy_gain_lcb
            elif compare == "mean":
                proxy_gain = proxy_gain_mean
            elif compare == "guided_ucb_vs_vanilla_mean":
                proxy_gain = lcb_g - qbar_0
            elif compare == "mean_penalty":
                proxy_gain = proxy_gain_mean_penalty
            else:
                proxy_gain = proxy_gain_lcb

        if uses_prior:
            base_key = random_key_from_data(obs_b)

            def _one_energy(idx, t):
                noise = jax.random.normal(jax.random.fold_in(base_key, idx + jnp.int32(7919)), vanilla_b.shape)
                x0_noisy = agent.diffusion.q_sample(t, vanilla_b, noise)
                xg_noisy = agent.diffusion.q_sample(t, guided_b, noise)
                pred0 = agent.predict_noise(policy_obj, obs_b, t, x0_noisy)
                predg = agent.predict_noise(policy_obj, obs_b, t, xg_noisy)
                e0 = jnp.mean((pred0 - noise) ** 2)
                eg = jnp.mean((predg - noise) ** 2)
                return e0, eg

            e0_losses, eg_losses = jax.vmap(_one_energy)(energy_indices, energy_timesteps)
            energy_0 = energy_alpha * jnp.mean(e0_losses)
            energy_g = energy_alpha * jnp.mean(eg_losses)
            energy_shift = energy_g - energy_0
            prior_improvement = -energy_shift
        else:
            energy_0 = jnp.float32(0.0)
            energy_g = jnp.float32(0.0)
            energy_shift = jnp.float32(0.0)
            prior_improvement = jnp.float32(0.0)

        shift = jnp.linalg.norm(guided - vanilla)
        shift_cost = jnp.float32(0.5) * shift * shift
        sat_cost = jnp.mean((jnp.maximum(jnp.abs(guided) - jnp.float32(0.98), 0.0) > 0.0).astype(jnp.float32))
        if mode == "proxy_only":
            raw_score = proxy_gain
        elif mode == "prior_only":
            raw_score = beta_prior * prior_improvement
        else:
            raw_score = proxy_gain + beta_prior * prior_improvement
        score = raw_score - rho_shift * shift_cost - rho_sat * sat_cost

        finite_ok = jnp.all(jnp.isfinite(guided))
        bound_ok = jnp.all(jnp.abs(guided) <= 1.0 + 1e-6)
        shift_ok = shift <= max_action_shift
        prior_ok = (energy_shift <= tau_prior) if uses_prior else jnp.asarray(True)
        proxy_ok = (proxy_gain >= proxy_min_gain) if uses_proxy else jnp.asarray(True)
        uncertainty_ok = jnp.logical_or(tau_u < 0.0, uq_g <= tau_u)
        score_ok = score >= final_gate_margin
        hard_ok = finite_ok & bound_ok & shift_ok & prior_ok & proxy_ok & uncertainty_ok
        accepted = hard_ok
        action = jnp.where(accepted, guided, vanilla)

        stats = {
            "final_gate/accepted": accepted.astype(jnp.float32),
            "final_gate/score": score,
            "final_gate/proxy_gain": proxy_gain,
            "final_gate/delta_c": proxy_gain,
            "final_gate/delta_e": energy_shift,
            "final_gate/energy_shift": energy_shift,
            "final_gate/prior_improvement": prior_improvement,
            "final_gate/prior_delta": prior_improvement,
            "final_gate/uses_proxy": jnp.float32(1.0 if uses_proxy else 0.0),
            "final_gate/uses_prior": jnp.float32(1.0 if uses_prior else 0.0),
            "final_gate/proxy_compare_mode_id": compare_mode_id,
            "final_gate/proxy_gain_lcb": proxy_gain_lcb,
            "final_gate/proxy_gain_robust": proxy_gain_robust,
            "final_gate/proxy_gain_mean": proxy_gain_mean,
            "final_gate/proxy_gain_mean_penalty": proxy_gain_mean_penalty,
            "final_gate/energy_guided": energy_g,
            "final_gate/energy_vanilla": energy_0,
            "final_gate/tau_prior": tau_prior,
            "final_gate/tau_u": tau_u,
            "final_gate/prior_ok": prior_ok.astype(jnp.float32),
            "final_gate/uncertainty_ok": uncertainty_ok.astype(jnp.float32),
            "final_gate/proxy_ok": proxy_ok.astype(jnp.float32),
            "final_gate/proxy_min_gain": proxy_min_gain,
            "final_gate/bound_ok": bound_ok.astype(jnp.float32),
            "final_gate/action_shift": shift,
            "final_gate/qbar_guided": qbar_g,
            "final_gate/qbar_vanilla": qbar_0,
            "final_gate/uq_guided": uq_g,
            "final_gate/uq_vanilla": uq_0,
            "final_gate/hard_ok": hard_ok.astype(jnp.float32),
            "final_gate/score_ok": score_ok.astype(jnp.float32),
            "final_gate/shift_ok": shift_ok.astype(jnp.float32),
            "final_gate/finite_ok": finite_ok.astype(jnp.float32),
            "final_gate/reject_low_score": ((~accepted) & hard_ok & (~score_ok)).astype(jnp.float32),
            "final_gate/reject_proxy": ((~accepted) & jnp.asarray(uses_proxy) & finite_ok & bound_ok & shift_ok & prior_ok & uncertainty_ok & (~proxy_ok)).astype(jnp.float32),
            "final_gate/reject_bound": ((~accepted) & finite_ok & (~bound_ok)).astype(jnp.float32),
            "final_gate/reject_shift": ((~accepted) & finite_ok & bound_ok & (~shift_ok)).astype(jnp.float32),
            "final_gate/reject_prior": ((~accepted) & finite_ok & bound_ok & shift_ok & (~prior_ok)).astype(jnp.float32),
            "final_gate/reject_uncertainty": ((~accepted) & finite_ok & bound_ok & shift_ok & prior_ok & proxy_ok & (~uncertainty_ok)).astype(jnp.float32),
            "final_gate/reject_nonfinite": ((~accepted) & (~finite_ok)).astype(jnp.float32),
        }
        stats.update(guidance_info)
        return action, stats

    return _final_gate_action

def _build_topk_rerank_jit_fn(agent, guidance_cfg: Dict[str, Any]):
    topk = max(2, int(guidance_cfg.get("rerank_topk", 2)))
    q_agg_id = int(guidance_cfg.get("guidance_q_agg_id", 0))
    lower_is_better = int(guidance_cfg.get("critic_objective_id", 1)) == 1

    @jax.jit
    def _topk_rerank_action(policy_obj, q_params, obs):
        obs_b = obs[None, :]
        obs_k = jnp.repeat(obs_b, topk, axis=0)
        key = random_key_from_data(obs_b)
        acts = agent.get_action(key, policy_obj, obs_k)
        acts = jnp.clip(acts, -1.0, 1.0)
        q1m, _ = agent.q(q_params[0], obs_k, acts)
        q2m, _ = agent.q(q_params[1], obs_k, acts)
        if q_agg_id == 1:
            score = 0.5 * (q1m + q2m)
        elif q_agg_id == 2:
            score = q1m
        elif q_agg_id == 3:
            score = jnp.minimum(q1m, q2m)
        elif q_agg_id == 4:
            score = jnp.maximum(q1m, q2m)
        elif q_agg_id == 5:
            score = jnp.maximum(q1m, q2m) if lower_is_better else jnp.minimum(q1m, q2m)
        else:
            score = jnp.minimum(q1m, q2m)
        score = jnp.ravel(score)
        best_idx = jnp.argmin(score) if lower_is_better else jnp.argmax(score)
        return acts[best_idx]

    return _topk_rerank_action


def resolve_action_adapter(
    agent,
    policy_payload: Any,
    sample_obs: np.ndarray,
    guidance_cfg: Optional[Dict[str, Any]] = None,
):
    sample_obs = np.asarray(sample_obs, dtype=np.float32)
    guidance_cfg = guidance_cfg or {}
    use_guidance = bool(guidance_cfg.get("use_value_guidance", False))
    rerank_topk = int(guidance_cfg.get("rerank_topk", 0) or 0)
    obs_variants = [
        ("single", jnp.asarray(sample_obs, dtype=jnp.float32)),
        ("batched", jnp.asarray(sample_obs[None, :], dtype=jnp.float32)),
    ]
    last_error: Optional[Exception] = None

    if rerank_topk > 1:
        components = _extract_guidance_components_from_payload(policy_payload)
        if components is None:
            print("   ERROR: top-k rerank requires policy/log_alpha plus q critics in checkpoint")
            return None
        ckpt_objective_id = int(components.get("critic_objective_id", -1))
        cli_objective_id = int(guidance_cfg.get("critic_objective_id", 1))
        if ckpt_objective_id >= 0 and ckpt_objective_id != cli_objective_id:
            print(
                "   WARNING: checkpoint critic_objective does not match --critic_objective "
                f"(checkpoint={ckpt_objective_id}, cli={cli_objective_id}); top-k rerank may be invalid"
            )
        policy_obj = components["policy_obj"]
        critic_params_source = str(guidance_cfg.get("critic_params", "target"))
        if critic_params_source == "online":
            q_params = components["online_q_params"]
        else:
            q_params = components["target_q_params"]
        for obs_mode, obs_val in obs_variants:
            try:
                topk_rerank_jit_fn = _build_topk_rerank_jit_fn(agent, guidance_cfg)
                adapter = {
                    "policy_obj": policy_obj,
                    "policy_tag": f"topk_rerank_{rerank_topk}",
                    "obs_mode": obs_mode,
                    "jit_fn": None,
                    "topk_rerank_jit_fn": topk_rerank_jit_fn,
                    "use_topk_rerank": True,
                    "use_final_gate": False,
                    "q_params": q_params,
                    "guidance_cfg": guidance_cfg,
                }
                out_np = det_action(agent, adapter, sample_obs)
                if out_np.shape != (ACT_DIM,) or not np.all(np.isfinite(out_np)):
                    continue
                print(
                    f"Action adapter resolved | policy=topk_rerank | obs={obs_mode} | "
                    f"K={rerank_topk} | JIT=yes | critic_params={critic_params_source}"
                )
                return adapter
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            print(f"   ERROR: top-k action adapter validation failed: {type(last_error).__name__}: {last_error}")
        return None

    if use_guidance:
        components = _extract_guidance_components_from_payload(policy_payload)
        if components is None:
            print("   ERROR: guided eval requires policy/log_alpha plus q critics in checkpoint")
            return None
        if not components["has_target_critics"]:
            print("   WARNING: checkpoint has no target critics; falling back to saved q1/q2 for guided eval")
        if components["version"] == "legacy":
            print("   WARNING: legacy checkpoint has no saved mean_q_std; sigma_ref fallback may be approximate")
        ckpt_objective_id = int(components.get("critic_objective_id", -1))
        cli_objective_id = int(guidance_cfg.get("critic_objective_id", 1))
        if ckpt_objective_id >= 0 and ckpt_objective_id != cli_objective_id:
            print(
                "   WARNING: checkpoint critic_objective does not match --critic_objective "
                f"(checkpoint={ckpt_objective_id}, cli={cli_objective_id}); guidance/gate results may be invalid"
            )

        policy_obj = components["policy_obj"]
        critic_params_source = str(guidance_cfg.get("critic_params", "target"))
        if critic_params_source == "online":
            q_params = components["online_q_params"]
        else:
            q_params = components["target_q_params"]
        sigma_ref = components["sigma_ref"]
        for obs_mode, obs_val in obs_variants:
            try:
                key = random_key_from_data(obs_val)
                out = agent.get_deterministic_action_guided(
                    key,
                    policy_obj,
                    q_params,
                    obs_val,
                    sigma_ref,
                    jnp.float32(guidance_cfg["lambda_0"]),
                    jnp.float32(guidance_cfg["beta_unc"]),
                    jnp.float32(guidance_cfg["p_decay"]),
                    jnp.float32(guidance_cfg["grad_clip"]),
                    jnp.int32(guidance_cfg["guidance_mode_id"]),
                    jnp.int32(guidance_cfg["guidance_target_id"]),
                    jnp.int32(guidance_cfg["guidance_step_interval"]),
                    jnp.int32(guidance_cfg["guidance_q_agg_id"]),
                    jnp.float32(guidance_cfg.get("guidance_kappa", 1.0)),
                    jnp.int32(guidance_cfg.get("guidance_injection_id", 0)),
                    jnp.int32(guidance_cfg.get("guidance_schedule_id", 0)),
                    jnp.float32(guidance_cfg.get("critic_objective_id", 1)),
                )
                out_np = np.asarray(jax.device_get(out), dtype=np.float32)
                if out_np.ndim == 2 and out_np.shape[0] == 1:
                    out_np = out_np[0]
                if out_np.ndim != 1 or out_np.shape[-1] != ACT_DIM:
                    continue
                if not np.all(np.isfinite(out_np)):
                    continue

                jit_fn = _build_jit_action_fn(agent, obs_mode, guidance_cfg)
                warmup_out = jit_fn(policy_obj, q_params, sigma_ref, jnp.asarray(sample_obs, dtype=jnp.float32))
                warmup_act = warmup_out[0] if isinstance(warmup_out, tuple) else warmup_out
                warmup_out_np = np.asarray(jax.device_get(warmup_act), dtype=np.float32)
                if warmup_out_np.shape != (ACT_DIM,) or not np.all(np.isfinite(warmup_out_np)):
                    print("WARNING: guided JIT output validation failed, fallback to non-JIT mode")
                    jit_fn = None

                vanilla_jit_fn = None
                final_gate_jit_fn = None
                if bool(guidance_cfg.get("use_final_gate", False)):
                    try:
                        vanilla_jit_fn = _build_jit_action_fn(agent, obs_mode, None)
                        vanilla_warm = vanilla_jit_fn(
                            policy_obj,
                            jnp.asarray(sample_obs, dtype=jnp.float32),
                        )
                        vanilla_warm_np = np.asarray(jax.device_get(vanilla_warm), dtype=np.float32)
                        if vanilla_warm_np.shape != (ACT_DIM,) or not np.all(np.isfinite(vanilla_warm_np)):
                            print("WARNING: final-gate vanilla JIT output validation failed, fallback to non-JIT mode")
                            vanilla_jit_fn = None
                    except Exception as e:
                        print(f"WARNING: final-gate vanilla JIT warmup failed: {type(e).__name__}: {e}")
                        vanilla_jit_fn = None

                    try:
                        final_gate_jit_fn = _build_fast_final_gate_jit_fn(agent, guidance_cfg)
                        fg_warm_action, fg_warm_stats = final_gate_jit_fn(
                            policy_obj,
                            q_params,
                            sigma_ref,
                            jnp.asarray(sample_obs, dtype=jnp.float32),
                        )
                        fg_warm_np = np.asarray(jax.device_get(fg_warm_action), dtype=np.float32)
                        if fg_warm_np.shape != (ACT_DIM,) or not np.all(np.isfinite(fg_warm_np)):
                            print("WARNING: final-gate fused JIT output validation failed, fallback to Python gate")
                            final_gate_jit_fn = None
                        else:
                            # Force compilation and metric-shape validation during adapter setup,
                            # not inside the first environment step.
                            _ = jax.device_get(fg_warm_stats)
                    except Exception as e:
                        print(f"WARNING: final-gate fused JIT warmup failed: {type(e).__name__}: {e}")
                        final_gate_jit_fn = None

                print(
                    f"Action adapter resolved | policy=guided:{components['version']} | obs={obs_mode} | "
                    f"JIT={'yes' if jit_fn else 'no'} | sigma_ref={float(jax.device_get(sigma_ref)):.6g}"
                    f" | critic_params={critic_params_source}"
                    f" | final_gate_vanilla_JIT={'yes' if vanilla_jit_fn else 'no'}"
                    f" | final_gate_fused_JIT={'yes' if final_gate_jit_fn else 'no'}"
                )
                return {
                    "policy_obj": policy_obj,
                    "policy_tag": "guided",
                    "obs_mode": obs_mode,
                    "jit_fn": jit_fn,
                    "vanilla_jit_fn": vanilla_jit_fn,
                    "final_gate_jit_fn": final_gate_jit_fn,
                    "use_value_guidance": True,
                    "use_final_gate": bool(guidance_cfg.get("use_final_gate", False)),
                    "q_params": q_params,
                    "sigma_ref": sigma_ref,
                    "guidance_cfg": guidance_cfg,
                }
            except Exception as e:
                last_error = e
                continue
        if last_error is not None:
            print(f"   ERROR: guided action adapter validation failed: {type(last_error).__name__}: {last_error}")
        return None

    for policy_tag, policy_obj in _candidate_policy_objs(policy_payload):
        for obs_mode, obs_val in obs_variants:
            try:
                out = agent.get_deterministic_action(policy_obj, obs_val)
                out_np = np.asarray(jax.device_get(out), dtype=np.float32)

                if out_np.ndim == 2 and out_np.shape[0] == 1:
                    out_np = out_np[0]

                if out_np.ndim != 1 or out_np.shape[-1] != ACT_DIM:
                    continue
                if not np.all(np.isfinite(out_np)):
                    continue

                jit_fn = _build_jit_action_fn(agent, obs_mode)
                obs_jax = jnp.asarray(sample_obs, dtype=jnp.float32)
                warmup_out = jit_fn(policy_obj, obs_jax)
                warmup_out_np = np.asarray(jax.device_get(warmup_out), dtype=np.float32)

                if warmup_out_np.shape != (ACT_DIM,) or not np.all(np.isfinite(warmup_out_np)):
                    print("WARNING: JIT output validation failed, fallback to non-JIT mode")
                    jit_fn = None

                print(
                    f"Action adapter resolved | policy={policy_tag} | obs={obs_mode} | "
                    f"JIT={'yes' if jit_fn else 'no'}"
                )
                return {
                    "policy_obj": policy_obj,
                    "policy_tag": policy_tag,
                    "obs_mode": obs_mode,
                    "jit_fn": jit_fn,
                }
            except Exception as e:
                last_error = e
                continue

    if last_error is not None:
        print(f"   ERROR: action adapter validation failed: {type(last_error).__name__}: {last_error}")
    return None


def _as_action_1d(action: Any) -> np.ndarray:
    arr = np.asarray(jax.device_get(action), dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    return np.clip(arr.reshape(-1), -1.0, 1.0)


def _q_bounds(agent, q_params, obs_b, act_b, kappa: float) -> Tuple[float, float, float, float]:
    q1m, _ = agent.q(q_params[0], obs_b, act_b)
    q2m, _ = agent.q(q_params[1], obs_b, act_b)
    q1 = float(jax.device_get(jnp.ravel(q1m)[0]))
    q2 = float(jax.device_get(jnp.ravel(q2m)[0]))
    qbar = 0.5 * (q1 + q2)
    # TeX notation: C_LCB=min_i C_i, C_UCB=max_i C_i,
    # U_psi=C_UCB-C_LCB=|C1-C2|.  kappa is retained in the CLI only for
    # legacy score ablations and no longer changes the TeX LCB/UCB bounds.
    lcb = min(q1, q2)
    ucb = max(q1, q2)
    uq = ucb - lcb
    return lcb, ucb, qbar, uq


def _scalar_from_jax(value: Any) -> float:
    try:
        arr = np.asarray(jax.device_get(value), dtype=np.float64)
        return float(np.mean(arr))
    except Exception:
        try:
            return float(value)
        except Exception:
            return float("nan")


def _append_guidance_stats(adapter: Dict[str, Any], guidance_info: Optional[Dict[str, Any]], extra: Optional[Dict[str, float]] = None) -> None:
    if not guidance_info:
        return
    # In long rollouts, collecting dozens of scalar diagnostics every single
    # env step can dominate runtime because every scalar conversion may sync
    # the GPU.  The action itself is still computed every step; this only
    # subsamples diagnostic bookkeeping for CSV/TensorBoard summaries.
    if not bool(adapter.get("_collect_step_stats", True)):
        return
    stats: Dict[str, float] = {}
    for k, v in guidance_info.items():
        stats[k] = _scalar_from_jax(v)
    if extra:
        for k, v in extra.items():
            stats[k] = float(v)
    adapter.setdefault("_episode_gate_stats", []).append(stats)


def _diffusion_prior_energy(
    agent,
    policy_obj,
    obs_b,
    act_b,
    *,
    samples: int,
    alpha: float,
) -> float:
    samples = max(1, int(samples))
    timesteps = np.linspace(0, max(int(agent.num_timesteps) - 1, 0), samples, dtype=np.int32)
    base_key = random_key_from_data(obs_b)
    losses: List[float] = []
    for idx, t_i in enumerate(timesteps):
        t = jnp.asarray(int(t_i), dtype=jnp.int32)
        noise = jax.random.normal(jax.random.fold_in(base_key, int(idx) + 7919), act_b.shape)
        x_noisy = agent.diffusion.q_sample(t, act_b, noise)
        pred = agent.predict_noise(policy_obj, obs_b, t, x_noisy)
        loss = jnp.mean((pred - noise) ** 2)
        losses.append(float(jax.device_get(loss)))
    return float(alpha) * float(np.mean(losses))


def _diffusion_prior_energy_pair(
    agent,
    policy_obj,
    obs_b,
    act0_b,
    actg_b,
    *,
    samples: int,
    alpha: float,
) -> Tuple[float, float]:
    """Estimate E(s,a0) and E(s,ag) with matched diffusion noise.

    The final gate in the TeX formulation compares the relative prior
    feasibility E(s,ag)-E(s,a0).  Using the same timestep/noise samples for
    both actions reduces Monte-Carlo variance and mirrors the training-time
    EnergyRank pair evaluation.
    """
    samples = max(1, int(samples))
    timesteps = np.linspace(0, max(int(agent.num_timesteps) - 1, 0), samples, dtype=np.int32)
    base_key = random_key_from_data(obs_b)
    e0_losses: List[float] = []
    eg_losses: List[float] = []
    for idx, t_i in enumerate(timesteps):
        t = jnp.asarray(int(t_i), dtype=jnp.int32)
        noise = jax.random.normal(jax.random.fold_in(base_key, int(idx) + 7919), act0_b.shape)

        x0_noisy = agent.diffusion.q_sample(t, act0_b, noise)
        xg_noisy = agent.diffusion.q_sample(t, actg_b, noise)

        pred0 = agent.predict_noise(policy_obj, obs_b, t, x0_noisy)
        predg = agent.predict_noise(policy_obj, obs_b, t, xg_noisy)

        e0_losses.append(float(jax.device_get(jnp.mean((pred0 - noise) ** 2))))
        eg_losses.append(float(jax.device_get(jnp.mean((predg - noise) ** 2))))

    return (
        float(alpha) * float(np.mean(e0_losses)),
        float(alpha) * float(np.mean(eg_losses)),
    )


def _dove_final_gate_action(agent, adapter: Dict[str, Any], obs: np.ndarray) -> np.ndarray:
    cfg = adapter["guidance_cfg"]
    obs_jax = jnp.asarray(obs, dtype=jnp.float32)
    obs_b = obs_jax[None, :]
    key = random_key_from_data(obs_b)

    fast_final_gate_fn = adapter.get("final_gate_jit_fn")
    if fast_final_gate_fn is not None:
        action_jax, stats = fast_final_gate_fn(
            adapter["policy_obj"],
            adapter["q_params"],
            adapter["sigma_ref"],
            obs_jax,
        )
        _append_guidance_stats(adapter, stats)
        return _as_action_1d(action_jax)

    # IMPORTANT: final gate needs both vanilla and guided actions.
    # Reuse pre-warmed JIT samplers. Otherwise JAX/XLA may recompile
    # the diffusion lax.scan inside every closed-loop env step.
    vanilla_jit_fn = adapter.get("vanilla_jit_fn")
    if vanilla_jit_fn is not None:
        vanilla_jax = vanilla_jit_fn(adapter["policy_obj"], obs_jax)
    else:
        vanilla_jax = agent.get_deterministic_action(adapter["policy_obj"], obs_b)

    guided_jit_fn = adapter.get("jit_fn")
    if guided_jit_fn is not None:
        guided_out = guided_jit_fn(
            adapter["policy_obj"],
            adapter["q_params"],
            adapter["sigma_ref"],
            obs_jax,
        )
        if isinstance(guided_out, tuple):
            guided_jax, guidance_info = guided_out
        else:
            guided_jax, guidance_info = guided_out, {}
    else:
        guided_jax, guidance_info = agent.get_deterministic_action_guided_with_metrics(
            key,
            adapter["policy_obj"],
            adapter["q_params"],
            obs_b,
            adapter["sigma_ref"],
            jnp.float32(cfg["lambda_0"]),
            jnp.float32(cfg["beta_unc"]),
            jnp.float32(cfg["p_decay"]),
            jnp.float32(cfg["grad_clip"]),
            jnp.int32(cfg["guidance_mode_id"]),
            jnp.int32(cfg["guidance_target_id"]),
            jnp.int32(cfg["guidance_step_interval"]),
            jnp.int32(cfg["guidance_q_agg_id"]),
            jnp.float32(cfg.get("guidance_kappa", 1.0)),
            jnp.int32(cfg.get("guidance_injection_id", 0)),
            jnp.int32(cfg.get("guidance_schedule_id", 0)),
            jnp.float32(cfg.get("critic_objective_id", 1)),
            anchor_action=vanilla_jax if bool(cfg.get("final_gate_project_guidance", True)) else None,
            max_action_shift=jnp.float32(cfg.get("final_gate_max_action_shift", -1.0)),
        )

    vanilla = _as_action_1d(vanilla_jax)
    guided = _as_action_1d(guided_jax)

    vanilla_b = jnp.asarray(vanilla[None, :], dtype=jnp.float32)
    guided_b = jnp.asarray(guided[None, :], dtype=jnp.float32)
    kappa = float(cfg.get("final_gate_kappa", 1.0))
    lcb_g, ucb_g, qbar_g, uq_g = _q_bounds(agent, adapter["q_params"], obs_b, guided_b, kappa)
    lcb_0, ucb_0, qbar_0, uq_0 = _q_bounds(agent, adapter["q_params"], obs_b, vanilla_b, kappa)

    compare = str(cfg.get("final_gate_proxy_compare", "robust"))
    lower_is_better = int(cfg.get("critic_objective_id", 1)) == 1
    if lower_is_better:
        # TeX Final Gate uses the conservative LCB-to-UCB risk improvement
        # Delta C = C_LCB(s,a0) - C_UCB(s,ag).  The older LCB-to-LCB variant
        # is retained as tex_lcb/strict for compatibility ablations.
        proxy_gain_lcb = lcb_0 - lcb_g
        proxy_gain_robust = lcb_0 - ucb_g
        proxy_gain_mean = qbar_0 - qbar_g
        proxy_gain_mean_penalty = proxy_gain_mean - 0.5 * kappa * uq_g
        if compare == "robust":
            proxy_gain = proxy_gain_robust
        elif compare in ("tex_lcb", "strict"):
            proxy_gain = proxy_gain_lcb
        elif compare == "mean":
            proxy_gain = proxy_gain_mean
        elif compare == "guided_ucb_vs_vanilla_mean":
            proxy_gain = qbar_0 - ucb_g
        elif compare == "mean_penalty":
            proxy_gain = proxy_gain_mean_penalty
        else:
            proxy_gain = proxy_gain_lcb
    else:
        proxy_gain_lcb = lcb_g - lcb_0
        proxy_gain_robust = lcb_g - ucb_0
        proxy_gain_mean = qbar_g - qbar_0
        proxy_gain_mean_penalty = proxy_gain_mean - 0.5 * kappa * uq_g
        if compare == "robust":
            proxy_gain = proxy_gain_robust
        elif compare in ("tex_lcb", "strict"):
            proxy_gain = proxy_gain_lcb
        elif compare == "mean":
            proxy_gain = proxy_gain_mean
        elif compare == "guided_ucb_vs_vanilla_mean":
            proxy_gain = lcb_g - qbar_0
        elif compare == "mean_penalty":
            proxy_gain = proxy_gain_mean_penalty
        else:
            proxy_gain = proxy_gain_lcb
    compare_mode_id = {
        "robust": 0.0,  # TeX-aligned conservative LCB(a0)-UCB(ag) rule
        "tex_lcb": 4.0,  # legacy optimistic LCB(a0)-LCB(ag) rule
        "strict": 4.0,
        "mean": 1.0,
        "guided_ucb_vs_vanilla_mean": 2.0,
        "mean_penalty": 3.0,
    }.get(compare, -1.0)

    mode = str(cfg.get("final_gate_mode", "proxy_prior"))
    if mode in ("proxy_prior", "prior_only"):
        energy_0, energy_g = _diffusion_prior_energy_pair(
            agent,
            adapter["policy_obj"],
            obs_b,
            vanilla_b,
            guided_b,
            samples=int(cfg.get("final_gate_energy_samples", 4)),
            alpha=float(cfg.get("final_gate_energy_alpha", 1.0)),
        )
        # TeX Delta E = E(s, a_g) - E(s, a_0).  Since lower diffusion energy
        # means better prior support, the scalar score uses the corresponding
        # improvement/bonus E(s, a_0) - E(s, a_g).
        energy_shift = energy_g - energy_0
        prior_improvement = -energy_shift
    else:
        energy_g = 0.0
        energy_0 = 0.0
        energy_shift = 0.0
        prior_improvement = 0.0
    uses_prior = mode in ("proxy_prior", "prior_only")
    uses_proxy = mode in ("proxy_prior", "proxy_only")
    tau_prior = float(cfg.get("final_gate_tau_prior", 0.02))
    prior_ok = bool((not uses_prior) or (energy_shift <= tau_prior))
    shift = float(np.linalg.norm(guided - vanilla))
    shift_cost = 0.5 * shift * shift
    sat_cost = float(np.mean(np.maximum(np.abs(guided) - 0.98, 0.0) > 0.0))
    if mode == "proxy_only":
        raw_score = proxy_gain
    elif mode == "prior_only":
        raw_score = float(cfg.get("final_gate_beta_prior", 1.0)) * prior_improvement
    else:
        raw_score = proxy_gain + float(cfg.get("final_gate_beta_prior", 1.0)) * prior_improvement
    score = raw_score - float(cfg.get("final_gate_rho_shift", 0.0)) * shift_cost - float(cfg.get("final_gate_rho_sat", 0.0)) * sat_cost
    finite_ok = bool(np.all(np.isfinite(guided)))
    bound_ok = bool(np.all(np.abs(guided) <= 1.0 + 1e-6))
    shift_ok = bool(shift <= float(cfg.get("final_gate_max_action_shift", 0.2)))
    proxy_min_gain = float(cfg.get("final_gate_min_proxy_gain", 0.0))
    # TeX Final Gate is a conjunction of feasibility checks.  The risk check is
    # Delta C >= kappa_C, represented by final_gate_min_proxy_gain.  The scalar
    # score is retained as a diagnostic/legacy ablation signal, but the default
    # paper path must not reject an action that already satisfies the explicit
    # risk, prior, shift, uncertainty, finite, and bound checks.
    proxy_ok = bool((not uses_proxy) or (proxy_gain >= proxy_min_gain))
    tau_u = float(cfg.get("final_gate_tau_u", float("inf")))
    uncertainty_ok = bool((tau_u < 0.0) or (uq_g <= tau_u))
    score_ok = bool(score >= float(cfg.get("final_gate_margin", 0.0)))
    hard_ok = finite_ok and bound_ok and shift_ok and prior_ok and proxy_ok and uncertainty_ok
    accepted = bool(hard_ok)

    stats = {
        "final_gate/accepted": float(accepted),
        "final_gate/score": float(score),
        "final_gate/proxy_gain": float(proxy_gain),
        # TeX notation aliases: Delta C is the selected proxy-risk gain.
        # TeX Delta E = E(s,ag)-E(s,a0).  prior_improvement = -Delta E is the
        # scalar-score bonus; prior_delta is kept only as a legacy/deprecated
        # alias for prior_improvement.
        "final_gate/delta_c": float(proxy_gain),
        "final_gate/delta_e": float(energy_shift),
        "final_gate/energy_shift": float(energy_shift),
        "final_gate/prior_improvement": float(prior_improvement),
        "final_gate/prior_delta": float(prior_improvement),  # legacy deprecated alias
        "final_gate/uses_proxy": float(uses_proxy),
        "final_gate/uses_prior": float(uses_prior),
        "final_gate/proxy_compare_mode_id": float(compare_mode_id),
        "final_gate/proxy_gain_lcb": float(proxy_gain_lcb),
        "final_gate/proxy_gain_robust": float(proxy_gain_robust),
        "final_gate/proxy_gain_mean": float(proxy_gain_mean),
        "final_gate/proxy_gain_mean_penalty": float(proxy_gain_mean_penalty),
        "final_gate/energy_guided": float(energy_g),
        "final_gate/energy_vanilla": float(energy_0),
        "final_gate/tau_prior": float(tau_prior),
        "final_gate/tau_u": float(tau_u),
        "final_gate/prior_ok": float(prior_ok),
        "final_gate/uncertainty_ok": float(uncertainty_ok),
        "final_gate/proxy_ok": float(proxy_ok),
        "final_gate/proxy_min_gain": float(proxy_min_gain),
        "final_gate/bound_ok": float(bound_ok),
        "final_gate/action_shift": float(shift),
        "final_gate/qbar_guided": float(qbar_g),
        "final_gate/qbar_vanilla": float(qbar_0),
        "final_gate/uq_guided": float(uq_g),
        "final_gate/uq_vanilla": float(uq_0),
        "final_gate/hard_ok": float(hard_ok),
        "final_gate/score_ok": float(score_ok),
        "final_gate/shift_ok": float(shift_ok),
        "final_gate/finite_ok": float(finite_ok),
        "final_gate/reject_low_score": float((not accepted) and hard_ok and (not score_ok)),
        "final_gate/reject_proxy": float((not accepted) and uses_proxy and finite_ok and bound_ok and shift_ok and prior_ok and uncertainty_ok and (not proxy_ok)),
        "final_gate/reject_bound": float((not accepted) and finite_ok and (not bound_ok)),
        "final_gate/reject_shift": float((not accepted) and finite_ok and bound_ok and (not shift_ok)),
        "final_gate/reject_prior": float((not accepted) and finite_ok and bound_ok and shift_ok and (not prior_ok)),
        "final_gate/reject_uncertainty": float((not accepted) and finite_ok and bound_ok and shift_ok and prior_ok and proxy_ok and (not uncertainty_ok)),
        "final_gate/reject_nonfinite": float((not accepted) and (not finite_ok)),
    }
    for k, v in guidance_info.items():
        stats[k] = _scalar_from_jax(v)
    adapter.setdefault("_episode_gate_stats", []).append(stats)
    return guided if accepted else vanilla


def det_action(agent, adapter: Dict[str, Any], obs: np.ndarray) -> np.ndarray:
    step_idx = int(adapter.get("_episode_action_step", 0))
    cfg_for_stride = adapter.get("guidance_cfg", {}) or {}
    stride = int(cfg_for_stride.get("step_diagnostics_stride", 1) or 0)
    adapter["_collect_step_stats"] = bool(stride > 0 and (step_idx % stride == 0))
    adapter["_episode_action_step"] = step_idx + 1

    obs_jax = jnp.asarray(obs, dtype=jnp.float32)
    jit_fn = adapter.get("jit_fn")
    use_guidance = bool(adapter.get("use_value_guidance", False))
    use_topk_rerank = bool(adapter.get("use_topk_rerank", False))
    use_final_gate = bool(adapter.get("use_final_gate", False))

    if use_final_gate and use_guidance and not use_topk_rerank:
        return _dove_final_gate_action(agent, adapter, obs)

    if use_topk_rerank:
        topk_rerank_jit_fn = adapter.get("topk_rerank_jit_fn")
        if topk_rerank_jit_fn is not None:
            act_jax = topk_rerank_jit_fn(adapter["policy_obj"], adapter["q_params"], obs_jax)
            return _as_action_1d(act_jax)
        cfg = adapter["guidance_cfg"]
        topk = max(2, int(cfg.get("rerank_topk", 2)))
        q_agg_id = int(cfg.get("guidance_q_agg_id", 0))
        lower_is_better = int(cfg.get("critic_objective_id", 1)) == 1
        if adapter["obs_mode"] == "batched":
            obs_for_policy = obs_jax[None, :]
        else:
            obs_for_policy = obs_jax
        obs_b = obs_jax[None, :]
        base_key = random_key_from_data(obs_b)
        best_action = None
        best_score = None
        for idx in range(topk):
            key_i = jax.random.fold_in(base_key, idx)
            act_jax = agent.get_action(key_i, adapter["policy_obj"], obs_for_policy)
            act = np.asarray(jax.device_get(act_jax), dtype=np.float32)
            if act.ndim == 2 and act.shape[0] == 1:
                act = act[0]
            act = np.clip(act, -1.0, 1.0)
            act_b = jnp.asarray(act[None, :], dtype=jnp.float32)
            q1m, _ = agent.q(adapter["q_params"][0], obs_b, act_b)
            q2m, _ = agent.q(adapter["q_params"][1], obs_b, act_b)
            if q_agg_id == 1:
                score = 0.5 * (q1m + q2m)
            elif q_agg_id == 2:
                score = q1m
            elif q_agg_id == 3:
                score = jnp.minimum(q1m, q2m)
            elif q_agg_id == 4:
                score = jnp.maximum(q1m, q2m)
            elif q_agg_id == 5:
                score = jnp.maximum(q1m, q2m) if lower_is_better else jnp.minimum(q1m, q2m)
            else:
                score = jnp.minimum(q1m, q2m)
            score_f = float(jax.device_get(jnp.ravel(score)[0]))
            if best_score is None:
                best_score = score_f
                best_action = act
            else:
                better = score_f < best_score if lower_is_better else score_f > best_score
                if better:
                    best_score = score_f
                    best_action = act
        return np.asarray(best_action, dtype=np.float32)

    if jit_fn is not None:
        if use_guidance:
            jit_out = jit_fn(adapter["policy_obj"], adapter["q_params"], adapter["sigma_ref"], obs_jax)
            if isinstance(jit_out, tuple):
                act_jax, guidance_info = jit_out
                _append_guidance_stats(adapter, guidance_info)
            else:
                act_jax = jit_out
        else:
            act_jax = jit_fn(adapter["policy_obj"], obs_jax)
        act = np.asarray(jax.device_get(act_jax), dtype=np.float32)
    else:
        if adapter["obs_mode"] == "batched":
            obs_jax = obs_jax[None, :]
        if use_guidance:
            cfg = adapter["guidance_cfg"]
            key = random_key_from_data(obs_jax)
            act, guidance_info = agent.get_deterministic_action_guided_with_metrics(
                key,
                adapter["policy_obj"],
                adapter["q_params"],
                obs_jax,
                adapter["sigma_ref"],
                jnp.float32(cfg["lambda_0"]),
                jnp.float32(cfg["beta_unc"]),
                jnp.float32(cfg["p_decay"]),
                jnp.float32(cfg["grad_clip"]),
                jnp.int32(cfg["guidance_mode_id"]),
                jnp.int32(cfg["guidance_target_id"]),
                jnp.int32(cfg["guidance_step_interval"]),
                jnp.int32(cfg["guidance_q_agg_id"]),
                jnp.float32(cfg.get("guidance_kappa", 1.0)),
                jnp.int32(cfg.get("guidance_injection_id", 0)),
                jnp.int32(cfg.get("guidance_schedule_id", 0)),
                jnp.float32(cfg.get("critic_objective_id", 1)),
            )
            _append_guidance_stats(adapter, guidance_info)
        else:
            act = agent.get_deterministic_action(adapter["policy_obj"], obs_jax)
        act = np.asarray(jax.device_get(act), dtype=np.float32)
        if act.ndim == 2 and act.shape[0] == 1:
            act = act[0]

    return np.clip(act, -1.0, 1.0)


def guidance_dry_run(agent, adapter: Dict[str, Any], obs: np.ndarray) -> Optional[Dict[str, float]]:
    if not bool(adapter.get("use_value_guidance", False)):
        return None
    cfg = adapter["guidance_cfg"]
    obs_jax = jnp.asarray(obs, dtype=jnp.float32)
    if adapter["obs_mode"] == "batched":
        obs_jax = obs_jax[None, :]
    key = random_key_from_data(obs_jax)
    _, info = agent.get_deterministic_action_guided_with_metrics(
        key,
        adapter["policy_obj"],
        adapter["q_params"],
        obs_jax,
        adapter["sigma_ref"],
        jnp.float32(cfg["lambda_0"]),
        jnp.float32(cfg["beta_unc"]),
        jnp.float32(cfg["p_decay"]),
        jnp.float32(cfg["grad_clip"]),
        jnp.int32(cfg["guidance_mode_id"]),
        jnp.int32(cfg["guidance_target_id"]),
        jnp.int32(cfg["guidance_step_interval"]),
        jnp.int32(cfg["guidance_q_agg_id"]),
        jnp.float32(cfg.get("guidance_kappa", 1.0)),
        jnp.int32(cfg.get("guidance_injection_id", 0)),
        jnp.int32(cfg.get("guidance_schedule_id", 0)),
        jnp.float32(cfg.get("critic_objective_id", 1)),
    )
    return {k: float(jax.device_get(v)) for k, v in info.items()}


# ============================================================================
# Policy file scan
# ============================================================================

def list_policy_files(log_dir: Path) -> List[Path]:
    files: Dict[Path, Path] = {}
    patterns = [
        "policy-*.pkl",
        "policy_*.pkl",
        "final_policy_*.pkl",
        "stage1b_offline_bc_policy.pkl",
        "final_bundle_step_*/final_policy_*.pkl",
        "final_bundle_step_*/stage1b_offline_bc_policy.pkl",
    ]
    for pattern in patterns:
        for path in log_dir.glob(pattern):
            files[path.resolve()] = path

    def _step_key(path: Path) -> Tuple[int, str]:
        name = path.name
        if name == "stage1b_offline_bc_policy.pkl":
            return (-1, name)
        m = re.search(r"policy-(\d+)-", name)
        if m:
            return (int(m.group(1)), name)
        m = re.search(r"final_policy_(\d+)", name)
        if m:
            return (int(m.group(1)), name)
        m = re.search(r"policy_\D*(\d+)", name)
        if m:
            return (int(m.group(1)), name)
        bundle_m = re.search(r"final_bundle_step_(\d+)", str(path))
        if bundle_m:
            return (int(bundle_m.group(1)), name)
        return (10**18, name)

    result = sorted(files.values(), key=_step_key)
    result = [p for p in result if not p.name.endswith(".meta.pkl")]
    return result


def is_evaluable_policy_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix == ".pkl"
        and not path.name.endswith(".meta.pkl")
    )


def resolve_policy_selection(log_dir_arg: Path, policy_arg: Optional[str]) -> Tuple[Path, List[Path], bool]:
    """
    Returns (policy_root, policy_files, single_policy).

    policy_root is used for relative policy names and default CSV placement.
    """
    log_dir_arg = log_dir_arg.expanduser()

    if policy_arg:
        base_dir = log_dir_arg if log_dir_arg.is_dir() else log_dir_arg.parent
        policy_path = Path(policy_arg).expanduser()
        if not policy_path.is_absolute():
            policy_path = base_dir / policy_path
        policy_path = policy_path.resolve()
        if not is_evaluable_policy_file(policy_path):
            raise FileNotFoundError(f"not an evaluable policy file: {policy_path}")
        return policy_path.parent, [policy_path], True

    if is_evaluable_policy_file(log_dir_arg):
        policy_path = log_dir_arg.resolve()
        return policy_path.parent, [policy_path], True

    if not log_dir_arg.exists():
        raise FileNotFoundError(f"log_dir does not exist: {log_dir_arg}")
    if not log_dir_arg.is_dir():
        raise FileNotFoundError(f"log_dir is not a directory or policy .pkl: {log_dir_arg}")

    policy_files = list_policy_files(log_dir_arg)
    return log_dir_arg, policy_files, False


def policy_display_name(policy_pkl: Path, policy_root: Path) -> str:
    try:
        return str(policy_pkl.relative_to(policy_root)).replace("\\", "/")
    except ValueError:
        return policy_pkl.name


class EvalVideoRecorder:
    def __init__(
        self,
        output_path: Path,
        fps: int = 30,
        size: Tuple[int, int] = (1280, 720),
        scaling: float = 6.0,
        every_n_steps: int = 1,
    ):
        self.output_path = Path(output_path)
        self.fps = int(fps)
        self.size = (int(size[0]), int(size[1]))
        self.scaling = float(scaling)
        self.every_n_steps = max(int(every_n_steps), 1)
        self.writer = None
        self.frame_count = 0
        self.capture_count = 0
        self._warned = False

    def _open_writer(self):
        if self.writer is not None:
            return
        import imageio.v2 as imageio

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = imageio.get_writer(
            str(self.output_path),
            fps=max(self.fps, 1),
            macro_block_size=1,
        )

    def _normalize_topdown_frame(self, frame: Any) -> Optional[np.ndarray]:
        if frame is None:
            return None
        arr = np.asarray(frame)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3 or arr.shape[2] < 3:
            return None
        arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # MetaDrive topdown renderer returns BGR.
        return arr[:, :, ::-1].copy()

    def capture(self, env: HumanInTheLoopEnv, force: bool = False) -> None:
        self.capture_count += 1
        if not force and (self.capture_count % self.every_n_steps != 0):
            return

        try:
            frame = env.render(
                mode="topdown",
                film_size=self.size,
                screen_size=self.size,
                scaling=self.scaling,
                window=False,
            )
            frame = self._normalize_topdown_frame(frame)
            if frame is None:
                raise RuntimeError("topdown renderer returned no frame")
            self._open_writer()
            self.writer.append_data(frame)
            self.frame_count += 1
        except Exception as e:
            if not self._warned:
                print(f"   WARNING: video frame capture failed: {e}")
                self._warned = True

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        if self.frame_count > 0:
            print(f"   video saved: {self.output_path} ({self.frame_count} frames)")
        elif self.output_path.exists():
            print(f"   WARNING: video file created but no frames were recorded: {self.output_path}")


def make_video_path(policy_pkl: Path, map_seed: int, episode_index: int) -> Path:
    return policy_pkl.with_name(
        f"{policy_pkl.stem}_map{int(map_seed)}_ep{int(episode_index) + 1}.mp4"
    )


# ============================================================================
# Evaluation logic
# ============================================================================

def terminal_reason(info: Dict[str, Any], terminated: bool, truncated: bool) -> str:
    if bool(info.get("arrive_dest", False)):
        return "success"
    if bool(info.get("crash", False)):
        return "crash"
    if bool(info.get("out_of_route", False)):
        return "out_of_route"
    if truncated:
        return "timeout"
    if terminated:
        return "terminated_other"
    return "unknown"


def _mean_std(values: List[float]) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float32)
    return float(np.mean(arr)), float(np.std(arr))


def _info_flag(info: Dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in info:
            try:
                return 1.0 if float(info.get(key) or 0.0) > 0.5 else 0.0
            except Exception:
                return 1.0 if bool(info.get(key)) else 0.0
    return 0.0


def _latest_step_gate_stats(adapter: Dict[str, Any]) -> Dict[str, float]:
    stats = adapter.get("_episode_gate_stats", [])
    if not stats:
        return {}
    try:
        latest = stats[-1]
        return {str(k): float(v) for k, v in latest.items()}
    except Exception:
        return {}


def _gate_stat(stats: Dict[str, float], key: str) -> float:
    try:
        value = float(stats[key])
        return value if np.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _risk_bounds_from_qbar_uq(qbar: float, uq: float) -> Tuple[float, float]:
    if not np.isfinite(qbar) or not np.isfinite(uq):
        return float("nan"), float("nan")
    half = 0.5 * max(float(uq), 0.0)
    return float(qbar - half), float(qbar + half)


def _reset_and_verify_seed(env: HumanInTheLoopEnv, map_seed: int, allowed_maps: Optional[List[int]] = None):
    obs, info = reset_to_strict_eval_map(env, map_seed, allowed_seeds=allowed_maps or [map_seed])
    return obs, info


def eval_episode(
    agent,
    adapter,
    env,
    map_seed,
    dt,
    allowed_maps=None,
    verbose=False,
    video_recorder: Optional[EvalVideoRecorder] = None,
    step_records: Optional[List[Dict[str, Any]]] = None,
    policy_name: str = "",
    policy_path: str = "",
    episode_idx: int = 0,
    reliability_horizon: int = 25,
):
    obs, info = _reset_and_verify_seed(env, map_seed, allowed_maps=allowed_maps or [int(map_seed)])
    adapter["_episode_gate_stats"] = []
    adapter["_episode_action_step"] = 0
    adapter["_collect_step_stats"] = True
    if video_recorder is not None:
        video_recorder.capture(env, force=True)

    done = False
    ep_reward = 0.0
    ep_cost = 0.0
    ep_native_cost = 0.0
    ep_takeover_cost = 0.0
    ep_steps = 0

    speeds: List[float] = []
    laterals: List[float] = []
    accels: List[float] = []
    jerks: List[float] = []
    steering_cmds: List[float] = []
    accel_cmds: List[float] = []
    action_saturation_flags: List[float] = []
    inference_latencies: List[float] = []
    prev_spd: Optional[float] = None
    prev_acc: Optional[float] = None
    final_info: Dict[str, Any] = {}
    terminated = False
    truncated = False
    takeover_steps = 0
    first_takeover_step = -1
    episode_step_records: List[Dict[str, Any]] = []

    while not done:
        t_action = time.time()
        action = det_action(agent, adapter, obs)
        inference_latencies.append(float(time.time() - t_action))
        gate_step_stats = _latest_step_gate_stats(adapter)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size >= 1:
            steering_cmds.append(float(action_arr[0]))
        if action_arr.size >= 2:
            accel_cmds.append(float(action_arr[1]))
        action_saturation_flags.append(float(np.any(np.abs(action_arr) >= 0.98)))
        obs, reward, terminated, truncated, info = env.step(action)
        if video_recorder is not None:
            video_recorder.capture(env)
        done = bool(terminated) or bool(truncated)
        final_info = info
        ep_reward += float(reward)
        ep_cost += float(info.get("cost", info.get("native_cost", 0.0)) or 0.0)
        ep_native_cost += float(info.get("native_cost", 0.0) or 0.0)
        ep_takeover_cost += float(info.get("takeover_cost", 0.0) or 0.0)
        ep_steps += 1

        if step_records is not None:
            qbar_g = _gate_stat(gate_step_stats, "final_gate/qbar_guided")
            uq_g = _gate_stat(gate_step_stats, "final_gate/uq_guided")
            cmin_g, cmax_g = _risk_bounds_from_qbar_uq(qbar_g, uq_g)
            qbar_0 = _gate_stat(gate_step_stats, "final_gate/qbar_vanilla")
            uq_0 = _gate_stat(gate_step_stats, "final_gate/uq_vanilla")
            cmin_0, cmax_0 = _risk_bounds_from_qbar_uq(qbar_0, uq_0)
            accepted = _gate_stat(gate_step_stats, "final_gate/accepted")
            episode_step_records.append(
                {
                    "policy_name": policy_name,
                    "policy_path": policy_path,
                    "map_seed": int(map_seed),
                    "episode_idx": int(episode_idx),
                    "step_idx": int(ep_steps - 1),
                    "action_steer": float(action_arr[0]) if action_arr.size >= 1 else float("nan"),
                    "action_accel": float(action_arr[1]) if action_arr.size >= 2 else float("nan"),
                    "gate_accepted": accepted,
                    "Cmax_ag": cmax_g,
                    "Cmin_ag": cmin_g,
                    "Cmean_ag": qbar_g,
                    "D_ag": uq_g,
                    "Cmax_a0": cmax_0,
                    "Cmin_a0": cmin_0,
                    "Cmean_a0": qbar_0,
                    "D_a0": uq_0,
                    "delta_C": _gate_stat(gate_step_stats, "final_gate/delta_c"),
                    "proxy_gain": _gate_stat(gate_step_stats, "final_gate/proxy_gain"),
                    "energy_shift": _gate_stat(gate_step_stats, "final_gate/energy_shift"),
                    "prior_improvement": _gate_stat(gate_step_stats, "final_gate/prior_improvement"),
                    "action_shift": _gate_stat(gate_step_stats, "final_gate/action_shift"),
                    "proxy_ok": _gate_stat(gate_step_stats, "final_gate/proxy_ok"),
                    "prior_ok": _gate_stat(gate_step_stats, "final_gate/prior_ok"),
                    "uncertainty_ok": _gate_stat(gate_step_stats, "final_gate/uncertainty_ok"),
                    "bound_ok": _gate_stat(gate_step_stats, "final_gate/bound_ok"),
                    "shift_ok": _gate_stat(gate_step_stats, "final_gate/shift_ok"),
                    "speed_mps": get_speed_mps(info),
                    "route_completion": _safe_float(info, "route_completion", "completion", "progress", "complete_ratio"),
                    "collision_now": _info_flag(info, "crash", "vehicle_crash", "crash_vehicle"),
                    "offroad_now": _info_flag(info, "out_of_route", "out_of_road", "offroad"),
                    "cost_now": float(info.get("cost", info.get("native_cost", 0.0)) or 0.0),
                    "terminal_now": 1.0 if done else 0.0,
                }
            )
        if float(info.get("takeover", False)) > 0.5:
            takeover_steps += 1
            if first_takeover_step < 0:
                first_takeover_step = ep_steps

        spd = get_speed_mps(info)
        speeds.append(spd)

        lateral = _safe_float(info, "lateral_offset", "lateral")
        if lateral is not None:
            laterals.append(abs(lateral))

        if prev_spd is not None and dt > 0:
            acc = (spd - prev_spd) / dt
            accels.append(abs(float(acc)))
            if prev_acc is not None:
                jerk = (acc - prev_acc) / dt
                jerks.append(abs(float(jerk)))
            prev_acc = float(acc)
        prev_spd = spd

    reason = terminal_reason(final_info, bool(terminated), bool(truncated))

    if verbose:
        true_keys = [k for k, v in final_info.items()
                     if isinstance(v, (bool, int, float)) and v]
        print(
            f"      [DEBUG] reason={reason} | steps={ep_steps} | rew={ep_reward:.2f}"
            f" | terminated={terminated} truncated={truncated}"
            f" | info_true_keys={true_keys}"
        )

    if step_records is not None:
        success_final = 1.0 if reason == "success" else 0.0
        failure_final = 0.0 if reason == "success" else 1.0
        horizon = max(1, int(reliability_horizon))
        for idx, record in enumerate(episode_step_records):
            future_slice = episode_step_records[idx:min(len(episode_step_records), idx + horizon)]
            future_collision = any(float(item.get("collision_now", 0.0) or 0.0) > 0.5 for item in future_slice)
            future_fail = future_collision or any(
                float(item.get("offroad_now", 0.0) or 0.0) > 0.5
                for item in future_slice
            )
            future_fail = future_fail or (
                failure_final > 0.5
                and any(float(item.get("terminal_now", 0.0) or 0.0) > 0.5 for item in future_slice)
            )
            record["future_collision_H"] = 1.0 if future_collision else 0.0
            record["future_fail_H"] = 1.0 if future_fail else 0.0
            record["success_final"] = success_final
            record["episode_failure"] = failure_final
            record["terminal_reason"] = reason
            record["reliability_horizon"] = horizon
        step_records.extend(episode_step_records)

    lateral_mean, lateral_std = _mean_std(laterals)
    speed_mean, speed_std = _mean_std(speeds)
    accel_mean, accel_std = _mean_std(accels)
    jerk_mean, jerk_std = _mean_std(jerks)
    steering_var = float(np.var(steering_cmds)) if steering_cmds else float("nan")
    accel_cmd_var = float(np.var(accel_cmds)) if accel_cmds else float("nan")
    action_saturation_ratio = float(np.mean(action_saturation_flags)) if action_saturation_flags else float("nan")
    inference_latency_mean = float(np.mean(inference_latencies)) if inference_latencies else float("nan")
    inference_latency_p95 = float(np.percentile(inference_latencies, 95)) if inference_latencies else float("nan")
    closed_loop_distance = float(np.sum(speeds) * max(float(dt), 0.0)) if speeds else 0.0
    interventions_per_km = float(takeover_steps / max(closed_loop_distance / 1000.0, 1e-6))
    time_to_first_intervention = float((first_takeover_step if first_takeover_step >= 0 else ep_steps) * max(float(dt), 0.0))
    route_completion_val = _safe_float(final_info, "route_completion", "completion", "progress", "complete_ratio")
    if route_completion_val is None:
        route_completion_val = 1.0 if reason == "success" else 0.0
    gate_stats = list(adapter.get("_episode_gate_stats", []))

    def _stat_mean(key: str) -> float:
        vals = [float(item[key]) for item in gate_stats if key in item and np.isfinite(float(item[key]))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "reward": float(ep_reward),
        "cost": float(ep_cost),
        "native_cost": float(ep_native_cost),
        "takeover_cost": float(ep_takeover_cost),
        "length": float(ep_steps),
        "takeover_rate": float(takeover_steps / max(ep_steps, 1)),
        "intervention_count": float(takeover_steps),
        "interventions_per_km": interventions_per_km,
        "autonomous_steps": float(max(ep_steps - takeover_steps, 0)),
        "first_takeover_step": float(first_takeover_step if first_takeover_step >= 0 else ep_steps),
        "time_to_first_intervention": time_to_first_intervention,
        "reason": reason,
        "success": 1.0 if reason == "success" else 0.0,
        "crash": 1.0 if reason == "crash" else 0.0,
        "out_of_route": 1.0 if reason == "out_of_route" else 0.0,
        "timeout": 1.0 if reason == "timeout" else 0.0,
        "route_completion": float(route_completion_val),
        "closed_loop_distance": closed_loop_distance,
        "lateral_offset_mean": lateral_mean,
        "lateral_offset_std": lateral_std,
        "speed_mean": speed_mean,
        "speed_std": speed_std,
        "accel_mean": accel_mean,
        "accel_std": accel_std,
        "jerk_mean": jerk_mean,
        "jerk_std": jerk_std,
        "steering_variance": steering_var,
        "accel_cmd_variance": accel_cmd_var,
        "action_saturation_ratio": action_saturation_ratio,
        "inference_latency_mean": inference_latency_mean,
        "inference_latency_p95": inference_latency_p95,
        "final_gate_accept_rate": _stat_mean("final_gate/accepted"),
        "final_gate_score_mean": _stat_mean("final_gate/score"),
        "final_gate_proxy_gain_mean": _stat_mean("final_gate/proxy_gain"),
        "final_gate_delta_c_mean": _stat_mean("final_gate/delta_c"),
        "final_gate_delta_e_mean": _stat_mean("final_gate/delta_e"),
        "final_gate_uses_proxy_rate": _stat_mean("final_gate/uses_proxy"),
        "final_gate_uses_prior_rate": _stat_mean("final_gate/uses_prior"),
        "final_gate_proxy_compare_mode_id_mean": _stat_mean("final_gate/proxy_compare_mode_id"),
        "final_gate_proxy_gain_lcb_mean": _stat_mean("final_gate/proxy_gain_lcb"),
        "final_gate_proxy_gain_robust_mean": _stat_mean("final_gate/proxy_gain_robust"),
        "final_gate_proxy_gain_qmean_mean": _stat_mean("final_gate/proxy_gain_mean"),
        "final_gate_proxy_gain_mean_penalty_mean": _stat_mean("final_gate/proxy_gain_mean_penalty"),
        "final_gate_prior_improvement_mean": _stat_mean("final_gate/prior_improvement"),
        "final_gate_prior_delta_mean": _stat_mean("final_gate/prior_delta"),
        "final_gate_action_shift_mean": _stat_mean("final_gate/action_shift"),
        "final_gate_energy_shift_mean": _stat_mean("final_gate/energy_shift"),
        "final_gate_energy_guided_mean": _stat_mean("final_gate/energy_guided"),
        "final_gate_energy_vanilla_mean": _stat_mean("final_gate/energy_vanilla"),
        "final_gate_hard_ok_rate": _stat_mean("final_gate/hard_ok"),
        "final_gate_prior_ok_rate": _stat_mean("final_gate/prior_ok"),
        "final_gate_proxy_ok_rate": _stat_mean("final_gate/proxy_ok"),
        "final_gate_bound_ok_rate": _stat_mean("final_gate/bound_ok"),
        "final_gate_uncertainty_ok_rate": _stat_mean("final_gate/uncertainty_ok"),
        "final_gate_tau_prior_mean": _stat_mean("final_gate/tau_prior"),
        "final_gate_tau_u_mean": _stat_mean("final_gate/tau_u"),
        "final_gate_proxy_min_gain_mean": _stat_mean("final_gate/proxy_min_gain"),
        "final_gate_reject_low_score_rate": _stat_mean("final_gate/reject_low_score"),
        "final_gate_reject_proxy_rate": _stat_mean("final_gate/reject_proxy"),
        "final_gate_reject_bound_rate": _stat_mean("final_gate/reject_bound"),
        "final_gate_reject_shift_rate": _stat_mean("final_gate/reject_shift"),
        "final_gate_reject_prior_rate": _stat_mean("final_gate/reject_prior"),
        "final_gate_reject_uncertainty_rate": _stat_mean("final_gate/reject_uncertainty"),
        "final_gate_reject_nonfinite_rate": _stat_mean("final_gate/reject_nonfinite"),
        "guidance_q_grad_norm_mean": _stat_mean("guidance/q_grad_norm"),
        "guidance_q_grad_norm_preclip_mean": _stat_mean("guidance/q_grad_norm_preclip"),
        "guidance_grad_clip_frac_mean": _stat_mean("guidance/grad_clip_frac"),
        "guidance_grad_nan_frac_mean": _stat_mean("guidance/grad_nan_frac"),
        "guidance_lambda_mean": _stat_mean("guidance/lambda"),
        "guidance_uncertainty_mean": _stat_mean("guidance/sigma_q"),
        "guidance_x0_clip_frac_mean": _stat_mean("guidance/x0_clip_frac"),
        "guidance_q_mean": _stat_mean("guidance/q_mean"),
        "guidance_proxy_score_mean": _stat_mean("guidance/proxy_score_mean"),
        "guidance_action_shift_mean": _stat_mean("guidance/action_shift"),
        "guidance_anchor_project_rate_mean": _stat_mean("guidance/anchor_project_rate"),
        "guidance_anchor_shift_mean": _stat_mean("guidance/anchor_shift"),
        "guidance_max_action_shift_mean": _stat_mean("guidance/max_action_shift"),
        "guidance_clean_injection_rate": _stat_mean("guidance/clean_injection_rate"),
    }


def eval_map_multi(
    agent,
    adapter,
    env,
    map_seed,
    dt,
    num_episodes,
    allowed_maps=None,
    verbose=False,
    policy_pkl: Optional[Path] = None,
    record_video: bool = False,
    video_episodes: int = 1,
    video_fps: int = 30,
    video_size: Tuple[int, int] = (1280, 720),
    video_scaling: float = 6.0,
    video_every_n_steps: int = 1,
    step_records: Optional[List[Dict[str, Any]]] = None,
    policy_name: str = "",
    reliability_horizon: int = 25,
):
    results = []
    for ep_idx in range(num_episodes):
        video_recorder = None
        if record_video and policy_pkl is not None and ep_idx < max(int(video_episodes), 0):
            video_recorder = EvalVideoRecorder(
                make_video_path(policy_pkl, int(map_seed), ep_idx),
                fps=video_fps,
                size=video_size,
                scaling=video_scaling,
                every_n_steps=video_every_n_steps,
            )
        try:
            metrics = eval_episode(
                agent, adapter, env, map_seed, dt,
                allowed_maps=allowed_maps, verbose=verbose,
                video_recorder=video_recorder,
                step_records=step_records,
                policy_name=policy_name,
                policy_path=str(policy_pkl) if policy_pkl is not None else "",
                episode_idx=ep_idx,
                reliability_horizon=reliability_horizon,
            )
            results.append(metrics)
        finally:
            if video_recorder is not None:
                video_recorder.close()
    return results


def _aggregate_episodes(episodes):
    if not episodes:
        return {}
    keys = episodes[0].keys()
    agg = {}
    for k in keys:
        if k == "reason":
            continue
        vals = [ep[k] for ep in episodes if np.isfinite(ep[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
    return agg


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="PVP-DACER Policy Batch Evaluation v4")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="training log directory, or a single policy .pkl for one-policy evaluation")
    parser.add_argument("--policy", "--policy_file", dest="policy_file", type=str, default=None,
                        help="single policy .pkl to evaluate/visualize; relative paths are resolved under --log_dir")
    parser.add_argument("--output_csv", type=str, default=None, help="output CSV path")
    parser.add_argument("--tensorboard_dir", type=str, default=None,
                        help="TensorBoard output directory; default is <output_csv_dir>/tb")
    parser.add_argument("--maps", type=int, nargs="+", default=EVAL_MAPS_DEFAULT,
                        help="held-out evaluation maps (default: seeds 200–209)")
    parser.add_argument("--horizon", type=int, default=1000,
                        help="max steps per episode (default 1000, aligned with training)")
    parser.add_argument("--dt", type=float, default=None,
                        help="explicit simulation dt; default infer automatically, fallback 0.02")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="episodes per map, default 5")
    parser.add_argument("--use_render", dest="use_render", action="store_true",
                        help="enable render for visual inspection")
    parser.add_argument("--no_use_render", dest="use_render", action="store_false",
                        help="disable render")
    parser.set_defaults(use_render=False)
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="print per-episode termination reason")
    parser.add_argument("--record_video", action="store_true", default=False,
                        help="record mp4 video for the evaluated policy episode(s)")
    parser.add_argument("--video_episodes", type=int, default=1,
                        help="episodes per map to record when --record_video is set")
    parser.add_argument("--video_fps", type=int, default=30,
                        help="video fps for --record_video")
    parser.add_argument("--video_size", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"),
                        help="topdown video frame size, default 1280 720")
    parser.add_argument("--video_scaling", type=float, default=6.0,
                        help="MetaDrive topdown renderer scaling for video")
    parser.add_argument("--video_every_n_steps", type=int, default=1,
                        help="record one video frame every N env steps")
    parser.add_argument("--episode_output_csv", type=str, default=None,
                        help="optional per-episode CSV path. Use this for exact mean/std tables across episodes.")
    parser.add_argument("--step_output_csv", type=str, default=None,
                        help="optional per-step reliability CSV with critic/gate scores and future-H risk labels.")
    parser.add_argument("--reliability_horizon", type=int, default=25,
                        help="future-step horizon used for step_output_csv labels such as future_collision_H.")
    parser.add_argument("--use_value_guidance", action="store_true",
                        help="use deterministic UPV-GDS guided inference for evaluation")
    parser.add_argument("--guidance_lambda0", type=float, default=0.5,
                        help="UPV-GDS guidance base strength")
    parser.add_argument("--guidance_beta_unc", type=float, default=1.0,
                        help="UPV-GDS uncertainty gate coefficient")
    parser.add_argument("--guidance_p_decay", type=float, default=1.0,
                        help="UPV-GDS time schedule exponent")
    parser.add_argument("--guidance_grad_clip", type=float, default=1.0,
                        help="UPV-GDS global action/latent gradient norm clip")
    parser.add_argument("--guidance_kappa", type=float, default=1.0,
                        help="legacy uncertainty multiplier kept for old runs; TeX main LCB/UCB use exact min/max")
    parser.add_argument("--guidance_injection", type=str, default="clean_x0",
                        choices=["clean_x0", "latent_mean"],
                        help="where to inject guidance: paper-aligned clean action x0 or legacy reverse mean")
    parser.add_argument("--guidance_schedule", type=str, default="noise_level",
                        choices=["noise_level", "alpha_cumprod"],
                        help="guidance schedule: paper phi(t)=((t+1)/T)^p or legacy alpha_cumprod^p")
    parser.add_argument("--guidance_mode", type=str, default="proxy_value",
                        choices=["proxy_value", "random_grad", "none", "reverse_grad"],
                        help="guided direction: proxy_value corrected main method, reverse_grad opposite-signed diagnostic, random_grad/none controls")
    parser.add_argument("--guidance_target", type=str, default="x0",
                        choices=["x0", "xt"],
                        help="critic input for guidance: Tweedie clean action x0 or noisy latent xt")
    parser.add_argument("--guidance_step_interval", type=int, default=1,
                        help="1 means every denoising step; K>1 means every K steps; 0 means final step only")
    parser.add_argument("--guidance_critic_params", type=str, default="target",
                        choices=["target", "online"],
                        help="use target critics or online critics for guidance")
    parser.add_argument("--guidance_q_agg", type=str, default="conservative",
                        choices=["min", "mean", "single_q1", "lcb", "ucb", "conservative"],
                        help="aggregate double-Q for guidance: min, mean, single q1, exact LCB=min (TeX main), exact UCB=max, or objective-aware conservative target")
    parser.add_argument("--critic_objective", type=str, default="lower",
                        choices=["value", "higher", "higher_is_better", "cost", "risk", "lower", "lower_is_better"],
                        help="critic score semantics: lower/risk/cost means smaller score is better; value/higher means larger score is better")
    parser.add_argument("--rerank_topk", type=int, default=0,
                        help="Top-K critic reranking baseline. 0 disables it; K>1 samples K actions and chooses best score under --critic_objective.")
    parser.add_argument("--use_final_gate", action="store_true",
                        help="enable empirical surrogate acceptance gate after the UPV-GDS proposal")
    parser.add_argument("--final_gate_mode", type=str, default="proxy_prior",
                        choices=["proxy_prior", "proxy_only", "prior_only"],
                        help="final gate score terms: full proxy+prior, proxy-only ablation, or prior-only ablation")
    parser.add_argument("--final_gate_proxy_compare", type=str, default="robust",
                        choices=["tex_lcb", "strict", "robust", "mean", "guided_ucb_vs_vanilla_mean", "mean_penalty"],
                        help="proxy comparison for final gate: robust=TeX LCB(a0)-UCB(ag), tex_lcb/strict=legacy LCB(a0)-LCB(ag), mean, guided UCB vs vanilla mean, or mean-uncertainty penalty")
    parser.add_argument("--final_gate_margin", type=float, default=0.0,
                        help="legacy diagnostic score margin; TeX-aligned acceptance uses explicit risk/prior/shift/uncertainty checks")
    parser.add_argument("--final_gate_min_proxy_gain", type=float, default=0.0,
                        help="hard proxy-improvement threshold: accept only if Delta C is greater than this value")
    parser.add_argument("--final_gate_kappa", type=float, default=1.0,
                        help="legacy score-ablation multiplier; TeX final gate uses exact min/max and U_psi=|C1-C2|")
    parser.add_argument("--final_gate_beta_prior", type=float, default=1.0,
                        help="weight for diffusion-prior relative value in final gate")
    parser.add_argument("--final_gate_rho_shift", type=float, default=0.0,
                        help="quadratic action-shift penalty in final gate")
    parser.add_argument("--final_gate_rho_sat", type=float, default=0.0,
                        help="action saturation penalty in final gate")
    parser.add_argument("--final_gate_max_action_shift", type=float, default=0.2,
                        help="hard maximum L2 shift between guided and vanilla actions; TeX-aligned main default d_max=0.2 and UPV-GDS anchor projection radius")
    parser.add_argument("--final_gate_tau_u", type=float, default=1.0,
                        help="hard critic-disagreement threshold U_psi(s,a_guided)<=tau_U; set <0 to disable")
    parser.add_argument("--final_gate_project_guidance", type=int, default=1, choices=[0, 1],
                        help="project each clean guided step onto the vanilla-action L2 ball; this implements the surrogate trust-region proposal and is independent from the final accept/reject gate")
    parser.add_argument("--final_gate_energy_samples", type=int, default=4,
                        help="number of diffusion timesteps used to estimate denoising energy")
    parser.add_argument("--final_gate_energy_alpha", type=float, default=1.0,
                        help="scale applied to denoising-energy relative prior value")
    parser.add_argument("--final_gate_tau_prior", type=float, default=0.02,
                        help="hard empirical prior-compatibility budget: accept only if E(ag)-E(a0) <= tau_prior")
    parser.add_argument("--guidance_dry_run", action="store_true",
                        help="print one-state guided inference diagnostics and skip environment rollout")
    parser.add_argument("--step_diagnostics_stride", type=int, default=1,
                        help="collect expensive per-step guidance/final-gate diagnostics every N env steps; 1 preserves old behavior, 0 disables diagnostic collection")
    args = parser.parse_args()
    if not args.maps:
        parser.error("at least one held-out evaluation map is required")
    duplicate_maps = sorted({seed for seed in args.maps if args.maps.count(seed) > 1})
    invalid_maps = sorted(set(args.maps) - set(TEST_MAP_SEEDS))
    if duplicate_maps:
        parser.error(f"duplicate evaluation map seeds: {duplicate_maps}")
    if invalid_maps:
        parser.error(
            f"evaluation maps must come from the held-out pool {list(TEST_MAP_SEEDS)}; "
            f"invalid seeds: {invalid_maps}"
        )
    args.guidance_mode_id = GUIDANCE_MODE_IDS[args.guidance_mode]
    args.guidance_target_id = GUIDANCE_TARGET_IDS[args.guidance_target]
    args.guidance_q_agg_id = GUIDANCE_Q_AGG_IDS[args.guidance_q_agg]
    args.guidance_injection_id = GUIDANCE_INJECTION_IDS[args.guidance_injection]
    args.guidance_schedule_id = GUIDANCE_SCHEDULE_IDS[args.guidance_schedule]
    args.critic_objective_id = CRITIC_OBJECTIVE_IDS[args.critic_objective]
    if args.guidance_lambda0 < 0 or args.guidance_beta_unc < 0 or args.guidance_p_decay < 0 or args.guidance_grad_clip <= 0:
        raise ValueError("invalid guidance hyperparameters")
    if args.guidance_kappa < 0:
        raise ValueError("--guidance_kappa must be >= 0")
    if args.guidance_step_interval < 0:
        raise ValueError("--guidance_step_interval must be >= 0")
    if args.step_diagnostics_stride < 0:
        raise ValueError("--step_diagnostics_stride must be >= 0")
    if args.rerank_topk < 0:
        raise ValueError("--rerank_topk must be >= 0")
    if args.use_final_gate and not args.use_value_guidance:
        raise ValueError("--use_final_gate requires --use_value_guidance")
    if args.rerank_topk > 1 and args.use_final_gate:
        raise ValueError("--use_final_gate and --rerank_topk are mutually exclusive")
    if args.final_gate_mode in ("proxy_prior", "prior_only") and args.final_gate_energy_samples <= 0:
        raise ValueError("--final_gate_energy_samples must be > 0 when final gate uses the prior term")
    if args.final_gate_max_action_shift <= 0:
        raise ValueError("--final_gate_max_action_shift must be > 0")
    if args.final_gate_tau_prior < 0:
        raise ValueError("--final_gate_tau_prior must be >= 0")
    if args.reliability_horizon <= 0:
        raise ValueError("--reliability_horizon must be > 0")

    log_dir_input = Path(args.log_dir).expanduser()
    try:
        policy_root, policy_files, single_policy = resolve_policy_selection(log_dir_input, args.policy_file)
    except Exception as e:
        print(f"ERROR: {e}")
        return

    if not policy_files:
        print(f"ERROR: no evaluable policy files found under {log_dir_input}")
        return

    if args.output_csv:
        output_csv = Path(args.output_csv)
    elif single_policy:
        output_csv = policy_files[0].with_name(f"{policy_files[0].stem}_policy_eval_results_v4.csv")
    else:
        output_csv = policy_root / "policy_eval_results_v4.csv"
    tensorboard_dir = Path(args.tensorboard_dir) if args.tensorboard_dir else output_csv.parent / "tb"

    run_args_dir = find_run_args_dir(log_dir_input, policy_files)
    run_args = load_run_args(run_args_dir)
    jax_backend, jax_devices = require_jax_gpu_ready()

    sep = "=" * 72
    print(f"\n{sep}")
    print("PVP-DACER Policy Batch Evaluation v4 (strict training alignment)")
    print(f"  input        : {log_dir_input}")
    print(f"  policy_root  : {policy_root}")
    print(f"  single_policy: {single_policy}")
    print(f"  maps         : {args.maps}")
    print(f"  num_episodes : {args.num_episodes}")
    print(f"  horizon      : {args.horizon}")
    print(f"  use_render   : {args.use_render}")
    print(f"  record_video : {args.record_video}")
    print(f"  value_guided : {args.use_value_guidance}")
    print(f"  rerank_topk  : {args.rerank_topk}")
    print(f"  final_gate   : {args.use_final_gate}")
    print(f"  critic_objective: {args.critic_objective}")
    print(f"  step_diag_stride: {args.step_diagnostics_stride}")
    if args.step_output_csv:
        print(f"  step_output  : {args.step_output_csv}")
        print(f"  reliability_H: {args.reliability_horizon}")
    if args.use_value_guidance:
        print(
            f"  guidance     : lambda0={args.guidance_lambda0} beta_unc={args.guidance_beta_unc} "
            f"p_decay={args.guidance_p_decay} grad_clip={args.guidance_grad_clip} kappa={args.guidance_kappa} "
            f"mode={args.guidance_mode} target={args.guidance_target} step_interval={args.guidance_step_interval} "
            f"injection={args.guidance_injection} schedule={args.guidance_schedule} "
            f"critic_params={args.guidance_critic_params} q_agg={args.guidance_q_agg}"
        )
    if args.use_final_gate:
        print(
            f"  gate         : mode={args.final_gate_mode} margin={args.final_gate_margin} "
            f"compare={args.final_gate_proxy_compare} "
            f"kappa={args.final_gate_kappa} beta_prior={args.final_gate_beta_prior} "
            f"rho_shift={args.final_gate_rho_shift} rho_sat={args.final_gate_rho_sat} "
            f"max_shift={args.final_gate_max_action_shift} tau_prior={args.final_gate_tau_prior} tau_u={args.final_gate_tau_u} "
            f"project_guidance={bool(args.final_gate_project_guidance)} min_proxy_gain={args.final_gate_min_proxy_gain} "
            f"energy_samples={args.final_gate_energy_samples}"
        )
    if args.record_video:
        print(f"  video        : fps={args.video_fps} size={tuple(args.video_size)} every={args.video_every_n_steps} episodes/map={args.video_episodes}")
    print(f"  verbose      : {args.verbose}")
    print(f"  output       : {output_csv}")
    print(f"  TensorBoard  : {tensorboard_dir}")
    print(f"  JAX backend  : {jax_backend}")
    print(f"  JAX devices  : {jax_devices}")
    print()
    print("  Env config (training-aligned):")
    print(f"    start_seed       = {run_args.get('start_seed', TRAIN_MAP_SEEDS[0])}")
    print(f"    num_scenarios    = {run_args.get('num_scenarios', len(TRAIN_MAP_SEEDS))}")
    print(f"    traffic_density  = {run_args.get('traffic_density', 0.06)}")
    print(f"    horizon          = {args.horizon}")
    print(f"    use_render       = {args.use_render}")
    print(f"    manual_control   = False")
    print(f"    enable_takeover  = False")
    print(f"    out_of_route_done= True")
    print(f"    crash_done       = True")
    print(f"    allowed_maps     = {[int(m) for m in args.maps]}")
    print(f"{sep}\n")

    print(f"Found {len(policy_files)} policy files")

    env = None
    tb_writer = _create_tb_writer(tensorboard_dir)
    try:
        _tb_add_scalar(tb_writer, "eval_config/num_policies", len(policy_files), 0)
        _tb_add_scalar(tb_writer, "eval_config/num_maps", len(args.maps), 0)
        _tb_add_scalar(tb_writer, "eval_config/num_episodes", args.num_episodes, 0)
        _tb_add_scalar(tb_writer, "eval_config/use_value_guidance", float(args.use_value_guidance), 0)
        _tb_add_scalar(tb_writer, "eval_config/use_final_gate", float(args.use_final_gate), 0)
        _tb_add_scalar(tb_writer, "eval_config/rerank_topk", args.rerank_topk, 0)
        print("Creating evaluation environment...", flush=True)
        env = create_eval_env(
            horizon=int(args.horizon),
            run_args=run_args,
            use_render=args.use_render,
        )

        allowed_maps = [int(m) for m in args.maps]
        install_strict_eval_seed_controller(env, allowed_maps)

        tmp_obs, _ = reset_to_strict_eval_map(
            env,
            seed=int(args.maps[0]),
            allowed_seeds=allowed_maps,
        )
        obs_dim = int(np.asarray(tmp_obs).shape[0])
        dt = infer_dt(env, args.dt)
        print(f"obs_dim={obs_dim}  act_dim={ACT_DIM}  dt={dt:.6f}")

        print("Building DACER agent...", flush=True)
        agent = build_agent(obs_dim, run_args)
        guidance_cfg = {
            "use_value_guidance": bool(args.use_value_guidance),
            "lambda_0": float(args.guidance_lambda0),
            "beta_unc": float(args.guidance_beta_unc),
            "p_decay": float(args.guidance_p_decay),
            "grad_clip": float(args.guidance_grad_clip),
            "guidance_kappa": float(args.guidance_kappa),
            "guidance_injection": str(args.guidance_injection),
            "guidance_injection_id": int(args.guidance_injection_id),
            "guidance_schedule": str(args.guidance_schedule),
            "guidance_schedule_id": int(args.guidance_schedule_id),
            "guidance_mode": str(args.guidance_mode),
            "guidance_mode_id": int(args.guidance_mode_id),
            "guidance_target": str(args.guidance_target),
            "guidance_target_id": int(args.guidance_target_id),
            "guidance_step_interval": int(args.guidance_step_interval),
            "critic_params": str(args.guidance_critic_params),
            "guidance_q_agg": str(args.guidance_q_agg),
            "guidance_q_agg_id": int(args.guidance_q_agg_id),
            "critic_objective": str(args.critic_objective),
            "critic_objective_id": int(args.critic_objective_id),
            "rerank_topk": int(args.rerank_topk),
            "use_final_gate": bool(args.use_final_gate),
            "final_gate_mode": str(args.final_gate_mode),
            "final_gate_proxy_compare": str(args.final_gate_proxy_compare),
            "final_gate_margin": float(args.final_gate_margin),
            "final_gate_min_proxy_gain": float(args.final_gate_min_proxy_gain),
            "final_gate_kappa": float(args.final_gate_kappa),
            "final_gate_beta_prior": float(args.final_gate_beta_prior),
            "final_gate_rho_shift": float(args.final_gate_rho_shift),
            "final_gate_rho_sat": float(args.final_gate_rho_sat),
            "final_gate_max_action_shift": float(args.final_gate_max_action_shift),
            "final_gate_energy_samples": int(args.final_gate_energy_samples),
            "final_gate_energy_alpha": float(args.final_gate_energy_alpha),
            "final_gate_tau_prior": float(args.final_gate_tau_prior),
            "final_gate_tau_u": float(args.final_gate_tau_u),
            "final_gate_project_guidance": bool(args.final_gate_project_guidance),
            "step_diagnostics_stride": int(args.step_diagnostics_stride),
        }

        metric_keys = [
            "reward", "cost", "native_cost", "takeover_cost",
            "length", "takeover_rate", "intervention_count", "interventions_per_km",
            "autonomous_steps", "first_takeover_step", "time_to_first_intervention",
            "success", "crash", "out_of_route", "timeout",
            "route_completion", "closed_loop_distance",
            "lateral_offset_mean", "lateral_offset_std",
            "speed_mean", "speed_std",
            "accel_mean", "accel_std",
            "jerk_mean", "jerk_std",
            "steering_variance", "accel_cmd_variance", "action_saturation_ratio",
            "inference_latency_mean", "inference_latency_p95",
            "final_gate_accept_rate", "final_gate_score_mean",
            "final_gate_proxy_gain_mean", "final_gate_delta_c_mean", "final_gate_delta_e_mean",
            "final_gate_uses_proxy_rate", "final_gate_uses_prior_rate",
            "final_gate_proxy_compare_mode_id_mean",
            "final_gate_proxy_gain_lcb_mean", "final_gate_proxy_gain_robust_mean", "final_gate_prior_improvement_mean", "final_gate_prior_delta_mean",
            "final_gate_proxy_gain_qmean_mean", "final_gate_proxy_gain_mean_penalty_mean",
            "final_gate_action_shift_mean", "final_gate_energy_shift_mean",
            "final_gate_energy_guided_mean", "final_gate_energy_vanilla_mean",
            "final_gate_hard_ok_rate", "final_gate_prior_ok_rate", "final_gate_proxy_ok_rate", "final_gate_bound_ok_rate",
            "final_gate_uncertainty_ok_rate", "final_gate_tau_prior_mean", "final_gate_tau_u_mean", "final_gate_proxy_min_gain_mean",
            "final_gate_reject_low_score_rate", "final_gate_reject_proxy_rate", "final_gate_reject_bound_rate",
            "final_gate_reject_shift_rate", "final_gate_reject_prior_rate", "final_gate_reject_uncertainty_rate", "final_gate_reject_nonfinite_rate",
            "guidance_q_grad_norm_mean", "guidance_q_grad_norm_preclip_mean",
            "guidance_grad_clip_frac_mean", "guidance_grad_nan_frac_mean",
            "guidance_lambda_mean", "guidance_uncertainty_mean",
            "guidance_x0_clip_frac_mean", "guidance_q_mean", "guidance_proxy_score_mean",
            "guidance_action_shift_mean", "guidance_anchor_project_rate_mean", "guidance_anchor_shift_mean",
            "guidance_max_action_shift_mean", "guidance_clean_injection_rate",
        ]

        header = ["policy_name"]
        for m in args.maps:
            for mk in metric_keys:
                header.append(f"map{m}_{mk}")
            if args.num_episodes > 1:
                header.append(f"map{m}_reward_std_across_eps")
                header.append(f"map{m}_success_std_across_eps")
        header.extend([
            "avg_reward", "avg_cost", "avg_native_cost", "avg_takeover_cost",
            "avg_takeover_rate", "avg_intervention_count", "avg_interventions_per_km",
            "avg_autonomous_steps", "avg_first_takeover_step", "avg_time_to_first_intervention",
            "avg_success", "avg_crash", "avg_out_of_route", "avg_timeout",
            "avg_route_completion", "avg_closed_loop_distance",
            "avg_lateral_offset_mean", "avg_speed_mean", "avg_accel_mean", "avg_jerk_mean",
            "avg_steering_variance", "avg_accel_cmd_variance", "avg_action_saturation_ratio",
            "avg_inference_latency_mean", "avg_inference_latency_p95",
            "avg_final_gate_accept_rate", "avg_final_gate_score_mean",
            "avg_final_gate_proxy_gain_mean", "avg_final_gate_delta_c_mean", "avg_final_gate_delta_e_mean",
            "avg_final_gate_uses_proxy_rate", "avg_final_gate_uses_prior_rate",
            "avg_final_gate_proxy_compare_mode_id_mean",
            "avg_final_gate_proxy_gain_lcb_mean", "avg_final_gate_proxy_gain_robust_mean", "avg_final_gate_prior_improvement_mean", "avg_final_gate_prior_delta_mean",
            "avg_final_gate_proxy_gain_qmean_mean", "avg_final_gate_proxy_gain_mean_penalty_mean",
            "avg_final_gate_action_shift_mean", "avg_final_gate_energy_shift_mean",
            "avg_final_gate_energy_guided_mean", "avg_final_gate_energy_vanilla_mean",
            "avg_final_gate_hard_ok_rate", "avg_final_gate_prior_ok_rate", "avg_final_gate_proxy_ok_rate", "avg_final_gate_bound_ok_rate",
            "avg_final_gate_uncertainty_ok_rate", "avg_final_gate_tau_prior_mean", "avg_final_gate_tau_u_mean", "avg_final_gate_proxy_min_gain_mean",
            "avg_final_gate_reject_low_score_rate", "avg_final_gate_reject_proxy_rate", "avg_final_gate_reject_bound_rate",
            "avg_final_gate_reject_shift_rate", "avg_final_gate_reject_prior_rate", "avg_final_gate_reject_uncertainty_rate", "avg_final_gate_reject_nonfinite_rate",
            "avg_guidance_q_grad_norm_mean", "avg_guidance_q_grad_norm_preclip_mean",
            "avg_guidance_grad_clip_frac_mean", "avg_guidance_grad_nan_frac_mean",
            "avg_guidance_lambda_mean", "avg_guidance_uncertainty_mean",
            "avg_guidance_x0_clip_frac_mean", "avg_guidance_q_mean", "avg_guidance_proxy_score_mean",
            "avg_guidance_action_shift_mean", "avg_guidance_anchor_project_rate_mean", "avg_guidance_anchor_shift_mean",
            "avg_guidance_max_action_shift_mean", "avg_guidance_clean_injection_rate",
        ])

        rows: List[List[Any]] = []
        episode_rows: List[Dict[str, Any]] = []
        step_rows: List[Dict[str, Any]] = []
        episode_header: List[str] = ["policy_name", "policy_path", "map_seed", "episode_idx"]
        summaries: List[Tuple[str, float, float]] = []
        total = len(policy_files)

        def _record_episode_rows(policy_name: str, policy_pkl: Path, map_seed: int, episodes: List[Dict[str, Any]]) -> None:
            if not args.episode_output_csv:
                return
            for ep_idx, ep in enumerate(episodes):
                ep_row: Dict[str, Any] = {
                    "policy_name": policy_name,
                    "policy_path": str(policy_pkl),
                    "map_seed": int(map_seed),
                    "episode_idx": int(ep_idx),
                }
                for key, value in ep.items():
                    ep_row[key] = value
                for key in ep_row.keys():
                    if key not in episode_header:
                        episode_header.append(key)
                episode_rows.append(ep_row)

        for idx, policy_pkl in enumerate(policy_files, start=1):
            policy_name = policy_display_name(policy_pkl, policy_root)
            print(f"\n[{idx:3d}/{total}] {policy_name}", flush=True)

            payload = load_policy_payload(policy_pkl)
            if payload is None:
                print("   skip: load failed")
                continue

            adapter = resolve_action_adapter(agent, payload, tmp_obs, guidance_cfg=guidance_cfg)
            if adapter is None:
                print("   ERROR: unable to resolve action interface, skip")
                continue

            if args.guidance_dry_run:
                info = guidance_dry_run(agent, adapter, tmp_obs)
                if info is None:
                    print("   guidance_dry_run: vanilla deterministic adapter, no guidance metrics")
                else:
                    parts = [f"{k}={v:.6g}" for k, v in sorted(info.items())]
                    print("   guidance_dry_run: " + " | ".join(parts))
                continue

            row: List[Any] = [policy_name]
            per_map_agg: List[Dict[str, float]] = []

            for map_seed in args.maps:
                try:
                    t0 = time.time()
                    episodes = eval_map_multi(
                        agent, adapter, env, int(map_seed), dt,
                        args.num_episodes, allowed_maps=allowed_maps, verbose=args.verbose,
                        policy_pkl=policy_pkl,
                        record_video=args.record_video,
                        video_episodes=args.video_episodes,
                        video_fps=args.video_fps,
                        video_size=(int(args.video_size[0]), int(args.video_size[1])),
                        video_scaling=args.video_scaling,
                        video_every_n_steps=args.video_every_n_steps,
                        step_records=step_rows if args.step_output_csv else None,
                        policy_name=policy_name,
                        reliability_horizon=int(args.reliability_horizon),
                    )
                    elapsed = time.time() - t0
                    agg = _aggregate_episodes(episodes)
                    per_map_agg.append(agg)
                    _record_episode_rows(policy_name, policy_pkl, int(map_seed), episodes)

                    for mk in metric_keys:
                        row.append(agg.get(mk, float("nan")))

                    if args.num_episodes > 1:
                        rew_vals = [ep["reward"] for ep in episodes if np.isfinite(ep["reward"])]
                        suc_vals = [ep["success"] for ep in episodes if np.isfinite(ep["success"])]
                        row.append(float(np.std(rew_vals)) if rew_vals else float("nan"))
                        row.append(float(np.std(suc_vals)) if suc_vals else float("nan"))

                    n_succ = sum(1 for ep in episodes if ep["success"] > 0.5)
                    n_crash = sum(1 for ep in episodes if ep["crash"] > 0.5)
                    n_oor = sum(1 for ep in episodes if ep["out_of_route"] > 0.5)
                    n_tout = sum(1 for ep in episodes if ep["timeout"] > 0.5)
                    tb_policy = _tb_safe_name(policy_name)
                    for mk in metric_keys:
                        metric_value = agg.get(mk, float("nan"))
                        _tb_add_scalar(tb_writer, f"eval/{tb_policy}/map_{int(map_seed)}/{mk}", metric_value, idx)
                        _tb_add_scalar(tb_writer, f"eval_by_metric/{mk}/{tb_policy}/map_{int(map_seed)}", metric_value, idx)

                    reasons = [ep.get("reason", "?") for ep in episodes]
                    reason_counts = {}
                    for r in reasons:
                        reason_counts[r] = reason_counts.get(r, 0) + 1
                    reason_str = " ".join(f"{r}x{c}" for r, c in reason_counts.items())

                    print(
                        f"   map={map_seed} ({args.num_episodes}ep)"
                        f" | rew={agg.get('reward', float('nan')):8.2f}"
                        f" | len={int(agg.get('length', 0)):4d}"
                        f" | tkv={agg.get('takeover_rate', float('nan')):.3f}"
                        f" | succ={n_succ}/{args.num_episodes}"
                        f" | crash={n_crash} oor={n_oor} tout={n_tout}"
                        f" | [{reason_str}]"
                        f" | spd={agg.get('speed_mean', float('nan')):.2f}m/s"
                        f" | {elapsed:.1f}s",
                        flush=True,
                    )
                except Exception as e:
                    print(f"   ERROR: map={map_seed} evaluation failed: {e}")
                    traceback.print_exc()
                    nan_count = len(metric_keys) + (2 if args.num_episodes > 1 else 0)
                    row.extend([float('nan')] * nan_count)

            def _avg(key: str) -> float:
                vals = [m[key] for m in per_map_agg if key in m and np.isfinite(m[key])]
                return float(np.mean(vals)) if vals else float("nan")

            avg_reward = _avg("reward")
            avg_success = _avg("success")
            row.extend([
                avg_reward, _avg("cost"), _avg("native_cost"), _avg("takeover_cost"),
                _avg("takeover_rate"), _avg("intervention_count"), _avg("interventions_per_km"),
                _avg("autonomous_steps"), _avg("first_takeover_step"), _avg("time_to_first_intervention"),
                avg_success,
                _avg("crash"), _avg("out_of_route"), _avg("timeout"),
                _avg("route_completion"), _avg("closed_loop_distance"),
                _avg("lateral_offset_mean"), _avg("speed_mean"),
                _avg("accel_mean"), _avg("jerk_mean"),
                _avg("steering_variance"), _avg("accel_cmd_variance"), _avg("action_saturation_ratio"),
                _avg("inference_latency_mean"), _avg("inference_latency_p95"),
                _avg("final_gate_accept_rate"), _avg("final_gate_score_mean"),
                _avg("final_gate_proxy_gain_mean"), _avg("final_gate_delta_c_mean"), _avg("final_gate_delta_e_mean"),
                _avg("final_gate_uses_proxy_rate"), _avg("final_gate_uses_prior_rate"),
                _avg("final_gate_proxy_compare_mode_id_mean"),
                _avg("final_gate_proxy_gain_lcb_mean"), _avg("final_gate_proxy_gain_robust_mean"), _avg("final_gate_prior_improvement_mean"), _avg("final_gate_prior_delta_mean"),
                _avg("final_gate_proxy_gain_qmean_mean"), _avg("final_gate_proxy_gain_mean_penalty_mean"),
                _avg("final_gate_action_shift_mean"), _avg("final_gate_energy_shift_mean"),
                _avg("final_gate_energy_guided_mean"), _avg("final_gate_energy_vanilla_mean"),
                _avg("final_gate_hard_ok_rate"), _avg("final_gate_prior_ok_rate"), _avg("final_gate_proxy_ok_rate"), _avg("final_gate_bound_ok_rate"),
                _avg("final_gate_uncertainty_ok_rate"), _avg("final_gate_tau_prior_mean"), _avg("final_gate_tau_u_mean"), _avg("final_gate_proxy_min_gain_mean"),
                _avg("final_gate_reject_low_score_rate"), _avg("final_gate_reject_proxy_rate"), _avg("final_gate_reject_bound_rate"),
                _avg("final_gate_reject_shift_rate"), _avg("final_gate_reject_prior_rate"), _avg("final_gate_reject_uncertainty_rate"), _avg("final_gate_reject_nonfinite_rate"),
                _avg("guidance_q_grad_norm_mean"), _avg("guidance_q_grad_norm_preclip_mean"),
                _avg("guidance_grad_clip_frac_mean"), _avg("guidance_grad_nan_frac_mean"),
                _avg("guidance_lambda_mean"), _avg("guidance_uncertainty_mean"),
                _avg("guidance_x0_clip_frac_mean"), _avg("guidance_q_mean"), _avg("guidance_proxy_score_mean"),
                _avg("guidance_action_shift_mean"), _avg("guidance_anchor_project_rate_mean"), _avg("guidance_anchor_shift_mean"),
                _avg("guidance_max_action_shift_mean"), _avg("guidance_clean_injection_rate"),
            ])
            rows.append(row)
            summaries.append((policy_name, avg_success, avg_reward))
            tb_policy = _tb_safe_name(policy_name)
            avg_metric_keys = [name[4:] for name in header if name.startswith("avg_")]
            avg_metric_values = row[-len(avg_metric_keys):] if avg_metric_keys else []
            for mk, value in zip(avg_metric_keys, avg_metric_values):
                _tb_add_scalar(tb_writer, f"eval/{tb_policy}/avg/{mk}", value, idx)
                _tb_add_scalar(tb_writer, f"eval_by_metric/avg_{mk}/{tb_policy}", value, idx)
            if tb_writer is not None:
                try:
                    tb_writer.flush()
                except Exception:
                    pass

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

        if args.episode_output_csv:
            episode_output_csv = Path(args.episode_output_csv)
            episode_output_csv.parent.mkdir(parents=True, exist_ok=True)
            with episode_output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=episode_header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(episode_rows)
            print(f"Episode CSV saved to: {episode_output_csv}")

        if args.step_output_csv:
            step_output_csv = Path(args.step_output_csv)
            step_output_csv.parent.mkdir(parents=True, exist_ok=True)
            step_header: List[str] = []
            for row_dict in step_rows:
                for key in row_dict.keys():
                    if key not in step_header:
                        step_header.append(key)
            with step_output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=step_header, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(step_rows)
            print(f"Step reliability CSV saved to: {step_output_csv} ({len(step_rows)} rows)")

        summaries = sorted(
            summaries,
            key=lambda x: (
                float("inf") if not np.isfinite(x[1]) else -x[1],
                float("inf") if not np.isfinite(x[2]) else -x[2],
                x[0],
            ),
        )

        print(f"\n{'=' * 72}")
        print(
            f"Evaluation done: {len(rows)} policies x {len(args.maps)} maps x "
            f"{args.num_episodes} episodes"
        )
        print(f"CSV saved to: {output_csv}")
        print(f"TensorBoard saved to: {tensorboard_dir}")
        if summaries:
            print("Top policies (by avg_success -> avg_reward):")
            for rank, (name, succ, rew) in enumerate(summaries[:10], start=1):
                print(f"  {rank:2d}. succ={succ:.3f} | rew={rew:8.2f} | {name}")
        print(f"{'=' * 72}")

    finally:
        if tb_writer is not None:
            try:
                tb_writer.flush()
                tb_writer.close()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
