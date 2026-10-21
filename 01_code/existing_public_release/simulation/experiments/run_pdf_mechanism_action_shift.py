from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pdf_experiment_utils import DEFAULT_LOG_DIR, DEFAULT_POLICY
from scripts import eval_pvp_policies_fixed_v3 as ev


def _raw_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "raw_data" in payload and "observation" not in payload:
        return payload["raw_data"]
    if isinstance(payload, dict):
        return payload
    return {}


def _pick(data: Dict[str, Any], *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    return None


def load_demo_samples(demo_root: Path, max_states: int, only_interventions: bool) -> List[Tuple[np.ndarray, np.ndarray, str, int]]:
    rows: List[Tuple[np.ndarray, np.ndarray, str, int]] = []
    for path in sorted(demo_root.rglob("*.pkl")):
        try:
            with path.open("rb") as f:
                data = _raw_dict(pickle.load(f))
        except Exception:
            continue
        obs = _pick(data, "observation", "obs")
        human = _pick(data, "action_human", "human_action", "action_behavior", "action")
        inter = _pick(data, "intervention", "interventions")
        if obs is None or human is None:
            continue
        obs_arr = np.asarray(obs, dtype=np.float32)
        human_arr = np.asarray(human, dtype=np.float32)
        if inter is None:
            inter_arr = np.ones(len(obs_arr), dtype=np.float32)
        else:
            inter_arr = np.asarray(inter, dtype=np.float32).reshape(-1)
        n = min(len(obs_arr), len(human_arr), len(inter_arr))
        for idx in range(n):
            if only_interventions and inter_arr[idx] <= 0.5:
                continue
            rows.append((obs_arr[idx], human_arr[idx], str(path), idx))
            if len(rows) >= max_states:
                return rows
    return rows


def q_score(agent, q_params, obs: np.ndarray, act: np.ndarray, q_agg: str) -> float:
    import jax
    import jax.numpy as jnp

    obs_b = jnp.asarray(obs[None, :], dtype=jnp.float32)
    act_b = jnp.asarray(act[None, :], dtype=jnp.float32)
    q1m, _ = agent.q(q_params[0], obs_b, act_b)
    q2m, _ = agent.q(q_params[1], obs_b, act_b)
    if q_agg == "mean":
        score = 0.5 * (q1m + q2m)
    elif q_agg == "single_q1":
        score = q1m
    elif q_agg == "lcb":
        score = jnp.minimum(q1m, q2m)
    elif q_agg == "ucb":
        score = jnp.maximum(q1m, q2m)
    elif q_agg == "conservative":
        score = jnp.maximum(q1m, q2m)
    else:
        score = jnp.minimum(q1m, q2m)
    return float(jax.device_get(jnp.ravel(score)[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.10.1 mechanism check: guided action shift toward human action.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--demo_root", type=Path, default=PROJECT_ROOT / "data104")
    parser.add_argument("--max_states", type=int, default=64)
    parser.add_argument("--all_states", action="store_true", default=False, help="use all demo states instead of only intervention states")
    parser.add_argument("--guidance_lambda0", type=float, default=0.5)
    parser.add_argument("--guidance_beta_unc", type=float, default=1.0)
    parser.add_argument("--guidance_q_agg", choices=["min", "mean", "single_q1", "lcb", "ucb", "conservative"], default="conservative")
    parser.add_argument("--output_csv", type=Path, default=ROOT / "runs" / "pdf_mechanism_action_shift" / "action_shift.csv")
    args = parser.parse_args()
    args.output_csv = args.output_csv.expanduser().resolve()

    samples = load_demo_samples(args.demo_root, max(1, args.max_states), not args.all_states)
    if not samples:
        raise RuntimeError(f"no demo samples found under {args.demo_root}")

    ev.require_jax_gpu_ready()
    run_args = ev.load_run_args(args.log_dir)
    obs_dim = int(samples[0][0].shape[0])
    agent = ev.build_agent(obs_dim, run_args)
    payload = ev.load_policy_payload(args.log_dir / args.policy)
    if payload is None:
        raise RuntimeError(f"failed to load policy {args.log_dir / args.policy}")

    cfg = {
        "use_value_guidance": True,
        "lambda_0": float(args.guidance_lambda0),
        "beta_unc": float(args.guidance_beta_unc),
        "p_decay": 1.0,
        "grad_clip": 1.0,
        "guidance_mode": "proxy_value",
        "guidance_mode_id": ev.GUIDANCE_MODE_IDS["proxy_value"],
        "guidance_target": "x0",
        "guidance_target_id": ev.GUIDANCE_TARGET_IDS["x0"],
        "guidance_step_interval": 1,
        "critic_params": "target",
        "guidance_q_agg": args.guidance_q_agg,
        "guidance_q_agg_id": ev.GUIDANCE_Q_AGG_IDS[args.guidance_q_agg],
        "guidance_kappa": 1.0,
        "guidance_injection": "clean_x0",
        "guidance_injection_id": ev.GUIDANCE_INJECTION_IDS["clean_x0"],
        "guidance_schedule": "noise_level",
        "guidance_schedule_id": ev.GUIDANCE_SCHEDULE_IDS["noise_level"],
        "rerank_topk": 0,
    }
    unguided = ev.resolve_action_adapter(agent, payload, samples[0][0], guidance_cfg={})
    guided = ev.resolve_action_adapter(agent, payload, samples[0][0], guidance_cfg=cfg)
    components = ev._extract_guidance_components_from_payload(payload)
    if unguided is None or guided is None or components is None:
        raise RuntimeError("failed to build unguided/guided adapters with q critics")
    q_params = components["target_q_params"]

    out_rows: List[Dict[str, Any]] = []
    for obs, human, source, index in samples:
        a0 = ev.det_action(agent, unguided, obs)
        ag = ev.det_action(agent, guided, obs)
        out_rows.append({
            "source": source,
            "index": index,
            "dist_unguided_to_human": float(np.linalg.norm(a0 - human)),
            "dist_guided_to_human": float(np.linalg.norm(ag - human)),
            "dist_guided_minus_unguided": float(np.linalg.norm(ag - human) - np.linalg.norm(a0 - human)),
            "action_shift": float(np.linalg.norm(ag - a0)),
            "q_unguided": q_score(agent, q_params, obs, a0, args.guidance_q_agg),
            "q_guided": q_score(agent, q_params, obs, ag, args.guidance_q_agg),
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"wrote {args.output_csv}")
    print(f"mean action_shift={np.mean([r['action_shift'] for r in out_rows]):.6f}")
    print(f"mean q_delta={np.mean([r['q_guided'] - r['q_unguided'] for r in out_rows]):.6f}")


if __name__ == "__main__":
    main()
