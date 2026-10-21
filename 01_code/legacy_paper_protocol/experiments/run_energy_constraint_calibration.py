from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.eval_pvp_policies_fixed_v3 import (  # noqa: E402
    build_agent,
    load_policy_payload,
    load_run_args,
    _extract_guidance_components_from_payload,
)


def _iter_pkl_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
        return
    for p in sorted(root.rglob("*.pkl")):
        if p.name.endswith(".meta.pkl"):
            continue
        yield p


def _array(data: Dict[str, Any], *names: str):
    for n in names:
        if n in data and data[n] is not None:
            return np.asarray(data[n], dtype=np.float32)
    return None


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    order = np.argsort(values)
    values = values[order]
    weights = np.maximum(weights[order], 0.0)
    total = float(np.sum(weights))
    if total <= 0:
        return float(np.quantile(values, quantile))
    cdf = np.cumsum(weights) / total
    return float(values[np.searchsorted(cdf, quantile, side="left").clip(0, len(values) - 1)])


def load_intervention_pairs(
    data_root: Path,
    positive_action: str = "behavior",
    max_pairs: int = 4096,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_list: List[np.ndarray] = []
    pos_list: List[np.ndarray] = []
    neg_list: List[np.ndarray] = []
    weight_list: List[float] = []
    for p in _iter_pkl_files(data_root):
        try:
            with p.open("rb") as f:
                data = pickle.load(f)
        except Exception:
            continue
        if isinstance(data, dict) and "raw_data" in data and "observation" not in data:
            data = data["raw_data"]
        if not isinstance(data, dict):
            continue
        obs = _array(data, "observation", "obs")
        a_b = _array(data, "action_behavior", "action")
        a_h = _array(data, "action_human", "human_action")
        a_n = _array(data, "action_novice", "action_agent", "agent_action")
        inter = _array(data, "intervention", "interventions")
        demo = _array(data, "is_demo")
        pt = _array(data, "is_pre_takeover")
        if obs is None or a_b is None or a_n is None:
            continue
        if a_h is None:
            a_h = a_b
        n = min(len(obs), len(a_b), len(a_n), len(a_h))
        if inter is None:
            inter = np.ones(n, dtype=np.float32)
        if demo is None:
            demo = np.zeros(n, dtype=np.float32)
        if pt is None:
            pt = np.zeros(n, dtype=np.float32)
        inter = np.asarray(inter[:n]).reshape(-1)
        demo = np.asarray(demo[:n]).reshape(-1)
        pt = np.asarray(pt[:n]).reshape(-1)
        # Main-paper EnergyRank calibration: only true intervention-time
        # counterfactual pairs are valid.  Pre-takeover context and pure demos
        # are excluded because they do not provide an unambiguous rejected a^-.
        valid = (inter > 0.5) & (demo <= 0.5) & (pt <= 0.5)
        pos = a_h[:n] if positive_action == "human" else a_b[:n]
        for i in np.where(valid)[0]:
            obs_list.append(np.asarray(obs[i], dtype=np.float32))
            pos_list.append(np.asarray(pos[i], dtype=np.float32))
            neg_list.append(np.asarray(a_n[i], dtype=np.float32))
            weight_list.append(1.0)
            if len(obs_list) >= max_pairs:
                return np.stack(obs_list), np.stack(pos_list), np.stack(neg_list), np.asarray(weight_list, dtype=np.float32)
    if not obs_list:
        raise RuntimeError(f"No valid intervention pairs found under {data_root}")
    return np.stack(obs_list), np.stack(pos_list), np.stack(neg_list), np.asarray(weight_list, dtype=np.float32)


def denoising_energy_pair(
    agent,
    policy_obj,
    obs: np.ndarray,
    act_pos: np.ndarray,
    act_neg: np.ndarray,
    key,
    samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate E(s,a+) and E(s,a-) with shared t and epsilon.

    This mirrors the TeX requirement for EnergyRank: both actions in the
    intervention pair must be compared at the same state, timestep, and noise.
    """
    obs_b = jnp.asarray(obs, dtype=jnp.float32)
    pos_b = jnp.asarray(act_pos, dtype=jnp.float32)
    neg_b = jnp.asarray(act_neg, dtype=jnp.float32)
    n = obs_b.shape[0]
    timesteps = jax.random.randint(key, (samples, n), minval=0, maxval=agent.num_timesteps)
    noise = jax.random.normal(jax.random.fold_in(key, 17), (samples, n, pos_b.shape[-1]))

    def one_sample(t_row, eps_row):
        x_pos = jax.vmap(agent.diffusion.q_sample)(t_row, pos_b, eps_row)
        x_neg = jax.vmap(agent.diffusion.q_sample)(t_row, neg_b, eps_row)
        pred_pos = jax.vmap(lambda o, ti, xi: agent.predict_noise(policy_obj, o, ti, xi))(obs_b, t_row, x_pos)
        pred_neg = jax.vmap(lambda o, ti, xi: agent.predict_noise(policy_obj, o, ti, xi))(obs_b, t_row, x_neg)
        e_pos = jnp.mean((pred_pos - eps_row) ** 2, axis=-1)
        e_neg = jnp.mean((pred_neg - eps_row) ** 2, axis=-1)
        return e_pos, e_neg

    pos_losses, neg_losses = jax.vmap(one_sample)(timesteps, noise)
    return (
        np.asarray(jax.device_get(jnp.mean(pos_losses, axis=0)), dtype=np.float32),
        np.asarray(jax.device_get(jnp.mean(neg_losses, axis=0)), dtype=np.float32),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Calibrate DOVE-ER intervention energy constraint on held-out HITL pairs.")
    ap.add_argument("--log_dir", type=Path, required=True, help="training run directory containing args.json")
    ap.add_argument("--policy_file", type=Path, required=True)
    ap.add_argument("--data_root", type=Path, required=True, help="demo/replay pkl file or directory")
    ap.add_argument("--output_csv", type=Path, required=True)
    ap.add_argument("--max_pairs", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--energy_samples", type=int, default=4)
    ap.add_argument("--er_margin", type=float, default=0.05)
    ap.add_argument("--positive_action", choices=["behavior", "human"], default="behavior")
    ap.add_argument("--quantiles", default="0.5,0.9,0.95,0.99")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    obs, a_pos, a_neg, pair_weights = load_intervention_pairs(
        args.data_root,
        args.positive_action,
        args.max_pairs,
    )
    run_args = load_run_args(args.log_dir)
    agent = build_agent(obs.shape[-1], run_args)
    payload = load_policy_payload(args.policy_file)
    comp = _extract_guidance_components_from_payload(payload)
    if comp is None:
        raise RuntimeError("policy file does not contain a diffusion policy/log_alpha payload")
    policy_obj = comp["policy_obj"]

    e_pos_all: List[np.ndarray] = []
    e_neg_all: List[np.ndarray] = []
    key = jax.random.key(args.seed)
    for start in range(0, len(obs), args.batch_size):
        end = min(start + args.batch_size, len(obs))
        key, k_pair = jax.random.split(key)
        e_pos, e_neg = denoising_energy_pair(
            agent,
            policy_obj,
            obs[start:end],
            a_pos[start:end],
            a_neg[start:end],
            k_pair,
            args.energy_samples,
        )
        e_pos_all.append(e_pos)
        e_neg_all.append(e_neg)
    e_pos = np.concatenate(e_pos_all)
    e_neg = np.concatenate(e_neg_all)
    nonconformity = e_pos - e_neg
    raw = args.er_margin + nonconformity
    gap_neg_minus_pos = e_neg - e_pos

    qs = [float(x) for x in args.quantiles.split(",") if x.strip()]
    weights = np.asarray(pair_weights, dtype=np.float32)
    weight_sum = float(np.sum(weights)) + 1e-8

    def wmean(x: np.ndarray) -> float:
        return float(np.sum(np.asarray(x, dtype=np.float32) * weights) / weight_sum)

    row: Dict[str, Any] = {
        "n_pairs": int(len(e_pos)),
        "weight_sum": float(np.sum(weights)),
        "pair_semantics": "true_intervention_only",
        "er_margin": float(args.er_margin),
        "energy_pos_mean": wmean(e_pos),
        "energy_neg_mean": wmean(e_neg),
        "gap_neg_minus_pos_mean": wmean(gap_neg_minus_pos),
        "nonconformity_mean": wmean(nonconformity),
        "constraint_violation_mean": wmean(np.maximum(raw, 0.0)),
        "violation_rate": wmean((raw > 0.0).astype(np.float32)),
        "margin_satisfied_rate": wmean((raw <= 0.0).astype(np.float32)),
        "gap_minus_margin_mean": wmean(gap_neg_minus_pos - args.er_margin),
    }
    for q in qs:
        row[f"nonconformity_q{int(q*100)}"] = float(np.quantile(nonconformity, q))
        row[f"nonconformity_weighted_q{int(q*100)}"] = weighted_quantile(nonconformity, weights, q)
        row[f"raw_violation_q{int(q*100)}"] = float(np.quantile(raw, q))
        row[f"raw_violation_weighted_q{int(q*100)}"] = weighted_quantile(raw, weights, q)
        row[f"gap_neg_minus_pos_q{int(q*100)}"] = float(np.quantile(gap_neg_minus_pos, q))
        row[f"gap_neg_minus_pos_weighted_q{int(q*100)}"] = weighted_quantile(gap_neg_minus_pos, weights, q)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"wrote {args.output_csv}")
    print(row)


if __name__ == "__main__":
    main()
