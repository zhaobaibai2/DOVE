from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pdf_experiment_utils import DEFAULT_LOG_DIR, DEFAULT_POLICY
from scripts import eval_pvp_policies_fixed_v3 as ev


PairSample = Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]


def _raw_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "raw_data" in payload and "observation" not in payload:
        raw = payload.get("raw_data", {})
        return raw if isinstance(raw, dict) else {}
    return payload if isinstance(payload, dict) else {}


def _pick(data: Dict[str, Any], *keys: str):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _as_2d(arr: Any, *, name: str) -> Optional[np.ndarray]:
    if arr is None:
        return None
    out = np.asarray(arr, dtype=np.float32)
    if out.ndim == 1:
        out = out.reshape(1, -1)
    if out.ndim != 2:
        print(f"WARNING: skip {name}: expected 2D array, got shape={out.shape}")
        return None
    return out


def load_intervention_pairs(demo_root: Path, max_pairs: int, min_action_gap: float) -> List[PairSample]:
    """Load D_I pairs (s, a+, a-) without pre-takeover/demo filtering.

    The paper definition indexes EnergyRank diagnostics by I only.  Therefore this
    loader keeps samples with intervention==1 and a valid rejected action a^-; it
    intentionally does not read is_pre_takeover or is_demo.
    """
    rows: List[PairSample] = []
    for path in sorted(demo_root.rglob("*.pkl")):
        try:
            with path.open("rb") as f:
                data = _raw_dict(pickle.load(f))
        except Exception:
            continue

        obs = _as_2d(_pick(data, "observation", "obs"), name=f"{path}:obs")
        a_pos = _as_2d(
            _pick(data, "action_behavior", "behavior_action", "actions_behavior", "action_human", "human_action", "actions_human", "raw_action", "action"),
            name=f"{path}:a_pos",
        )
        a_neg = _as_2d(
            _pick(data, "actions_novice", "action_novice", "novice_action", "action_agent", "agent_action", "raw_novice_action", "raw_agent_action"),
            name=f"{path}:a_neg",
        )
        inter = _pick(data, "intervention", "interventions", "I")
        pair_ok = _pick(data, "pair_ok", "valid_pair", "has_rejected_action")
        if obs is None or a_pos is None or a_neg is None:
            continue
        if inter is None:
            inter_arr = np.ones((len(obs),), dtype=np.float32)
        else:
            inter_arr = np.asarray(inter, dtype=np.float32).reshape(-1)
        if pair_ok is None:
            pair_ok_arr = np.ones((len(obs),), dtype=np.float32)
        else:
            pair_ok_arr = np.asarray(pair_ok, dtype=np.float32).reshape(-1)

        n = min(len(obs), len(a_pos), len(a_neg), len(inter_arr), len(pair_ok_arr))
        for idx in range(n):
            if inter_arr[idx] <= 0.5 or pair_ok_arr[idx] <= 0.5:
                continue
            if float(np.linalg.norm(a_pos[idx] - a_neg[idx])) < float(min_action_gap):
                continue
            rows.append((obs[idx], a_pos[idx], a_neg[idx], str(path), idx))
            if len(rows) >= max_pairs:
                return rows
    return rows


def prior_energy(agent, policy_obj: Any, obs: np.ndarray, act: np.ndarray, samples: int, alpha: float) -> float:
    import jax.numpy as jnp

    obs_b = jnp.asarray(obs[None, :], dtype=jnp.float32)
    act_b = jnp.asarray(act[None, :], dtype=jnp.float32)
    return ev._diffusion_prior_energy(agent, policy_obj, obs_b, act_b, samples=samples, alpha=alpha)


def sample_policy_actions(agent, policy_obj: Any, obs: np.ndarray, num_samples: int) -> np.ndarray:
    import jax
    import jax.numpy as jnp

    obs_b = jnp.asarray(obs[None, :], dtype=jnp.float32)
    base_key = ev.random_key_from_data(obs_b)
    actions: List[np.ndarray] = []
    for k in range(max(1, int(num_samples))):
        key = jax.random.fold_in(base_key, int(k) + 12345)
        act = agent.get_action(key, policy_obj, obs_b)
        act_np = np.asarray(jax.device_get(act), dtype=np.float32)
        if act_np.ndim == 2 and act_np.shape[0] == 1:
            act_np = act_np[0]
        actions.append(np.clip(act_np, -1.0, 1.0))
    return np.asarray(actions, dtype=np.float32)


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mechanism diagnostics for EnergyRank prior aliasing: PairSat_E, V_E, IAR, and RAGR."
    )
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--demo_root", type=Path, default=PROJECT_ROOT / "data104")
    parser.add_argument("--max_pairs", type=int, default=256)
    parser.add_argument("--energy_samples", type=int, default=4)
    parser.add_argument("--energy_alpha", type=float, default=1.0)
    parser.add_argument("--er_margin", type=float, default=0.05)
    parser.add_argument("--min_action_gap", type=float, default=1e-6)
    parser.add_argument("--ragr_delta", type=float, default=0.05)
    parser.add_argument("--ragr_k", type=int, default=16)
    parser.add_argument("--allow_cpu", action="store_true", help="allow CPU fallback when JAX CUDA is not visible")
    parser.add_argument("--output_csv", type=Path, default=ROOT / "runs" / "pdf_mechanism_prior_aliasing" / "prior_aliasing_pairs.csv")
    parser.add_argument("--summary_csv", type=Path, default=ROOT / "runs" / "pdf_mechanism_prior_aliasing" / "prior_aliasing_summary.csv")
    args = parser.parse_args()

    args.output_csv = args.output_csv.expanduser().resolve()
    args.summary_csv = args.summary_csv.expanduser().resolve()

    pairs = load_intervention_pairs(args.demo_root.expanduser(), max(1, args.max_pairs), args.min_action_gap)
    if not pairs:
        raise RuntimeError(
            f"no valid intervention pairs with rejected action found under {args.demo_root}; "
            "check that the data contains actions_novice/action_novice and intervention fields"
        )

    if args.allow_cpu:
        print("WARNING: running prior-aliasing diagnostics with CPU fallback; JAX CUDA was not required.")
    else:
        ev.require_jax_gpu_ready()
    run_args = ev.load_run_args(args.log_dir)
    agent = ev.build_agent(int(pairs[0][0].shape[0]), run_args)
    payload = ev.load_policy_payload(args.log_dir / args.policy)
    components = ev._extract_guidance_components_from_payload(payload)
    if payload is None or components is None:
        raise RuntimeError("policy payload must include policy and log_alpha for diffusion-prior diagnostics")
    policy_obj = components["policy_obj"]

    rows: List[Dict[str, Any]] = []
    for obs, a_pos, a_neg, source, index in pairs:
        e_pos = prior_energy(agent, policy_obj, obs, a_pos, args.energy_samples, args.energy_alpha)
        e_neg = prior_energy(agent, policy_obj, obs, a_neg, args.energy_samples, args.energy_alpha)
        violation = max(0.0, float(args.er_margin) + e_pos - e_neg)
        samples = sample_policy_actions(agent, policy_obj, obs, args.ragr_k)
        dists = np.linalg.norm(samples - a_neg[None, :], axis=-1)
        min_dist = float(np.min(dists))
        rows.append({
            "source": source,
            "index": index,
            "energy_pos": e_pos,
            "energy_neg": e_neg,
            "energy_gap_neg_minus_pos": e_neg - e_pos,
            "pairsat_e": float(e_pos + args.er_margin <= e_neg),
            "v_e": violation,
            "iar": float(e_neg <= e_pos),
            "ragr": float(min_dist <= args.ragr_delta),
            "min_dist_to_rejected": min_dist,
            "pair_gap": float(np.linalg.norm(a_pos - a_neg)),
            "er_margin": float(args.er_margin),
            "ragr_delta": float(args.ragr_delta),
            "ragr_k": int(args.ragr_k),
        })

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "variant": "prior_aliasing",
        "source_csv": str(args.output_csv),
        "n_pairs": len(rows),
        "pairsat_e": _mean(r["pairsat_e"] for r in rows),
        "v_e": _mean(r["v_e"] for r in rows),
        "iar": _mean(r["iar"] for r in rows),
        "ragr": _mean(r["ragr"] for r in rows),
        "energy_gap_neg_minus_pos": _mean(r["energy_gap_neg_minus_pos"] for r in rows),
        "min_dist_to_rejected_mean": _mean(r["min_dist_to_rejected"] for r in rows),
        "er_margin": float(args.er_margin),
        "ragr_delta": float(args.ragr_delta),
        "ragr_k": int(args.ragr_k),
    }
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(f"wrote {args.output_csv}")
    print(f"wrote {args.summary_csv}")
    print(
        "PairSat_E={pairsat_e:.4f} V_E={v_e:.6f} IAR={iar:.4f} RAGR={ragr:.4f}".format(**summary)
    )


if __name__ == "__main__":
    main()
