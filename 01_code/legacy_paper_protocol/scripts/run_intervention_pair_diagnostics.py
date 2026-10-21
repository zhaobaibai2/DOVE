#!/usr/bin/env python3
"""Intervention-pair diagnostics for DOVEER mechanism tables.

This script computes pair-level diagnostics that cannot be obtained from normal
closed-loop autonomous eval CSVs:

  PairSat_E  = mean[ E(s,a+) + margin <= E(s,a-) ]
  V_E        = mean[ max(0, margin + E(s,a+) - E(s,a-)) ]
  IAR        = mean[ E(s,a-) <= E(s,a+) ]

It also reports critic-side pair satisfaction and Delta C diagnostics when the
policy checkpoint contains critics.  Valid pairs are true intervention-time
pairs only: intervention==1, not pure demo, not pre-takeover context, pair_ok==1,
and with a real rejected/novice action a-.
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts import eval_pvp_policies_fixed_v3 as ev
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Failed to import scripts/eval_pvp_policies_fixed_v3.py. "
        "Run this script from the project root after activating dacer_pvp."
    ) from exc


@dataclass
class PairBatch:
    obs: np.ndarray
    a_pos: np.ndarray
    a_neg: np.ndarray
    sources: List[str]
    indices: np.ndarray


def _iter_pkl_files(root: Path) -> Iterator[Path]:
    root = root.expanduser()
    if root.is_file() and root.suffix == ".pkl":
        yield root
        return
    if not root.exists():
        return
    for p in sorted(root.rglob("*.pkl")):
        # Skip obvious policy checkpoints and metadata when a broad log root is passed.
        name = p.name
        if name.endswith(".meta.pkl"):
            continue
        if re.match(r"^(policy-|policy_|final_policy_|stage1b_offline_bc_policy)", name):
            continue
        yield p


def _load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def _raw_dict(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict) and "raw_data" in payload and "observation" not in payload and "obs" not in payload:
        raw = payload.get("raw_data", {})
        return raw if isinstance(raw, dict) else {}
    return payload if isinstance(payload, dict) else {}


def _pick(data: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _array(data: Dict[str, Any], *keys: str, dtype=np.float32) -> Optional[np.ndarray]:
    value = _pick(data, *keys)
    if value is None:
        return None
    try:
        return np.asarray(value, dtype=dtype)
    except Exception:
        return None


def _as_2d(x: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        return None
    return arr


def _as_1d(x: Optional[np.ndarray], n: int, default: float) -> np.ndarray:
    if x is None:
        return np.full((n,), float(default), dtype=np.float32)
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(arr) < n:
        pad = np.full((n - len(arr),), float(default), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:n]


def _pairs_from_arrays(
    *,
    obs: Optional[np.ndarray],
    a_behavior: Optional[np.ndarray],
    a_human: Optional[np.ndarray],
    a_novice: Optional[np.ndarray],
    intervention: Optional[np.ndarray],
    is_demo: Optional[np.ndarray],
    is_pre_takeover: Optional[np.ndarray],
    pair_ok: Optional[np.ndarray],
    positive_action: str,
    min_action_gap: float,
    source: str,
    max_keep: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]]:
    obs2 = _as_2d(obs)
    beh2 = _as_2d(a_behavior)
    hum2 = _as_2d(a_human)
    neg2 = _as_2d(a_novice)
    if obs2 is None or neg2 is None:
        return []
    if beh2 is None and hum2 is None:
        return []
    if beh2 is None:
        beh2 = hum2
    if hum2 is None:
        hum2 = beh2
    assert beh2 is not None and hum2 is not None

    n = min(len(obs2), len(beh2), len(hum2), len(neg2))
    if n <= 0:
        return []
    inter = _as_1d(intervention, n, 1.0)
    demo = _as_1d(is_demo, n, 0.0)
    pre = _as_1d(is_pre_takeover, n, 0.0)
    ok = _as_1d(pair_ok, n, 1.0)
    pos2 = hum2 if positive_action == "human" else beh2

    rows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]] = []
    for i in range(n):
        if inter[i] <= 0.5:
            continue
        if demo[i] > 0.5 or pre[i] > 0.5 or ok[i] <= 0.5:
            continue
        gap = float(np.linalg.norm(pos2[i] - neg2[i]))
        if gap < float(min_action_gap):
            continue
        if not (np.all(np.isfinite(obs2[i])) and np.all(np.isfinite(pos2[i])) and np.all(np.isfinite(neg2[i]))):
            continue
        rows.append((obs2[i].astype(np.float32), pos2[i].astype(np.float32), neg2[i].astype(np.float32), source, i))
        if len(rows) >= max_keep:
            break
    return rows


def _pairs_from_dict(
    data: Dict[str, Any],
    *,
    source: str,
    positive_action: str,
    min_action_gap: float,
    max_keep: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]]:
    obs = _array(data, "observation", "obs")
    a_behavior = _array(data, "action_behavior", "actions_behavior", "behavior_action", "action")
    a_human = _array(data, "action_human", "actions_human", "human_action", "raw_action")
    a_novice = _array(data, "action_novice", "actions_novice", "action_agent", "agent_action", "raw_agent_action", "raw_novice_action", "novice_action")
    intervention = _array(data, "intervention", "interventions", "takeover", "I")
    is_demo = _array(data, "is_demo", "demo")
    is_pre = _array(data, "is_pre_takeover", "pre_takeover")
    pair_ok = _array(data, "pair_ok", "valid_pair", "has_rejected_action")
    return _pairs_from_arrays(
        obs=obs,
        a_behavior=a_behavior,
        a_human=a_human,
        a_novice=a_novice,
        intervention=intervention,
        is_demo=is_demo,
        is_pre_takeover=is_pre,
        pair_ok=pair_ok,
        positive_action=positive_action,
        min_action_gap=min_action_gap,
        source=source,
        max_keep=max_keep,
    )


def _buffer_to_pairs(
    obj: Any,
    *,
    source: str,
    positive_action: str,
    min_action_gap: float,
    max_keep: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]]:
    # Saved PVPBalancedDualBuffer: use human_buffer because it contains the
    # intervention-side action triples and pair_ok flags.
    buf = getattr(obj, "human_buffer", obj)
    if not all(hasattr(buf, name) for name in ("obs", "actions_behavior", "actions_novice", "actions_human", "interventions")):
        return []
    try:
        size = int(len(buf))
    except Exception:
        size = int(getattr(buf, "_size", 0))
    if size <= 0:
        return []
    return _pairs_from_arrays(
        obs=np.asarray(buf.obs[:size], dtype=np.float32),
        a_behavior=np.asarray(buf.actions_behavior[:size], dtype=np.float32),
        a_human=np.asarray(buf.actions_human[:size], dtype=np.float32),
        a_novice=np.asarray(buf.actions_novice[:size], dtype=np.float32),
        intervention=np.asarray(buf.interventions[:size], dtype=np.float32).reshape(-1),
        is_demo=np.asarray(getattr(buf, "is_demo", np.zeros((size, 1)))[:size], dtype=np.float32).reshape(-1),
        is_pre_takeover=np.asarray(getattr(buf, "is_pre_takeover", np.zeros((size, 1)))[:size], dtype=np.float32).reshape(-1),
        pair_ok=np.asarray(getattr(buf, "pair_ok", np.ones((size, 1)))[:size], dtype=np.float32).reshape(-1),
        positive_action=positive_action,
        min_action_gap=min_action_gap,
        source=source,
        max_keep=max_keep,
    )


def load_pairs(data_root: Path, *, positive_action: str, min_action_gap: float, max_pairs: int) -> PairBatch:
    pairs: List[Tuple[np.ndarray, np.ndarray, np.ndarray, str, int]] = []
    for p in _iter_pkl_files(data_root):
        remain = int(max_pairs) - len(pairs)
        if remain <= 0:
            break
        try:
            payload = _load_pickle(p)
        except Exception as exc:
            print(f"[warn] skip unreadable pkl: {p} ({exc})", flush=True)
            continue
        rows = _buffer_to_pairs(
            payload,
            source=str(p),
            positive_action=positive_action,
            min_action_gap=min_action_gap,
            max_keep=remain,
        )
        if not rows:
            rows = _pairs_from_dict(
                _raw_dict(payload),
                source=str(p),
                positive_action=positive_action,
                min_action_gap=min_action_gap,
                max_keep=remain,
            )
        pairs.extend(rows)
    if not pairs:
        raise RuntimeError(
            f"No valid intervention pairs found under {data_root}. "
            "Need pkl data with observation/obs, action_novice or agent_action, intervention==1, and pair_ok!=0."
        )
    obs = np.stack([r[0] for r in pairs]).astype(np.float32)
    a_pos = np.stack([r[1] for r in pairs]).astype(np.float32)
    a_neg = np.stack([r[2] for r in pairs]).astype(np.float32)
    sources = [r[3] for r in pairs]
    indices = np.asarray([r[4] for r in pairs], dtype=np.int32)
    return PairBatch(obs=obs, a_pos=a_pos, a_neg=a_neg, sources=sources, indices=indices)


def _mean(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def _std(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0 if arr.size == 1 else float("nan")


def _stderr(x: Sequence[float]) -> float:
    arr = np.asarray(x, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    return float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0 if arr.size == 1 else float("nan")


def _fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    if not math.isfinite(mean):
        return ""
    if not math.isfinite(std):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _energy_pair_batch(agent, policy_obj: Any, obs: np.ndarray, a_pos: np.ndarray, a_neg: np.ndarray, *, samples: int, alpha: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    import jax
    import jax.numpy as jnp

    obs_b = jnp.asarray(obs, dtype=jnp.float32)
    pos_b = jnp.asarray(a_pos, dtype=jnp.float32)
    neg_b = jnp.asarray(a_neg, dtype=jnp.float32)
    n = int(obs_b.shape[0])
    key = jax.random.key(int(seed))
    timesteps = jax.random.randint(key, (int(samples), n), minval=0, maxval=int(agent.num_timesteps))
    noise = jax.random.normal(jax.random.fold_in(key, 7919), (int(samples), n, pos_b.shape[-1]))

    def one_sample(t_row, eps_row):
        x_pos = jax.vmap(agent.diffusion.q_sample)(t_row, pos_b, eps_row)
        x_neg = jax.vmap(agent.diffusion.q_sample)(t_row, neg_b, eps_row)
        pred_pos = jax.vmap(lambda o, ti, xi: agent.predict_noise(policy_obj, o, ti, xi))(obs_b, t_row, x_pos)
        pred_neg = jax.vmap(lambda o, ti, xi: agent.predict_noise(policy_obj, o, ti, xi))(obs_b, t_row, x_neg)
        e_pos = jnp.mean((pred_pos - eps_row) ** 2, axis=-1)
        e_neg = jnp.mean((pred_neg - eps_row) ** 2, axis=-1)
        return e_pos, e_neg

    e_pos_s, e_neg_s = jax.vmap(one_sample)(timesteps, noise)
    e_pos = float(alpha) * jnp.mean(e_pos_s, axis=0)
    e_neg = float(alpha) * jnp.mean(e_neg_s, axis=0)
    return (
        np.asarray(jax.device_get(e_pos), dtype=np.float32),
        np.asarray(jax.device_get(e_neg), dtype=np.float32),
    )


def _critic_pair_batch(
    agent,
    q_params: Any,
    obs: np.ndarray,
    a_pos: np.ndarray,
    a_neg: np.ndarray,
    *,
    lower_is_better: bool,
    critic_margin: float,
) -> Dict[str, np.ndarray]:
    import jax
    import jax.numpy as jnp

    obs_b = jnp.asarray(obs, dtype=jnp.float32)
    pos_b = jnp.asarray(a_pos, dtype=jnp.float32)
    neg_b = jnp.asarray(a_neg, dtype=jnp.float32)
    q1p, _ = agent.q(q_params[0], obs_b, pos_b)
    q2p, _ = agent.q(q_params[1], obs_b, pos_b)
    q1n, _ = agent.q(q_params[0], obs_b, neg_b)
    q2n, _ = agent.q(q_params[1], obs_b, neg_b)
    q1p = jnp.ravel(q1p); q2p = jnp.ravel(q2p); q1n = jnp.ravel(q1n); q2n = jnp.ravel(q2n)
    lcb_pos = jnp.minimum(q1p, q2p)
    ucb_pos = jnp.maximum(q1p, q2p)
    qbar_pos = 0.5 * (q1p + q2p)
    lcb_neg = jnp.minimum(q1n, q2n)
    ucb_neg = jnp.maximum(q1n, q2n)
    qbar_neg = 0.5 * (q1n + q2n)
    if lower_is_better:
        delta_c_mean = qbar_neg - qbar_pos
        delta_c_robust = lcb_neg - ucb_pos
    else:
        delta_c_mean = qbar_pos - qbar_neg
        delta_c_robust = lcb_pos - ucb_neg
    pair_acc_c = delta_c_mean > 0.0
    pair_sat_c = delta_c_robust >= 0.0
    pair_sat_c_margin = delta_c_robust >= float(critic_margin)
    v_c = jnp.maximum(0.0, float(critic_margin) - delta_c_robust)
    out = {
        "qbar_pos": qbar_pos,
        "qbar_neg": qbar_neg,
        "delta_c_mean": delta_c_mean,
        "delta_c_robust": delta_c_robust,
        "pairacc_c": pair_acc_c.astype(jnp.float32),
        "pairsat_c": pair_sat_c.astype(jnp.float32),
        "pairsat_c_margin": pair_sat_c_margin.astype(jnp.float32),
        "v_c": v_c,
        "unc_pos": jnp.abs(q1p - q2p),
        "unc_neg": jnp.abs(q1n - q2n),
    }
    return {k: np.asarray(jax.device_get(v), dtype=np.float32) for k, v in out.items()}


def _resolve_policy_path(policy_root: Path, policy_name: str) -> Path:
    p = Path(policy_name).expanduser()
    if p.is_absolute():
        return p
    cand = policy_root / p
    if cand.exists():
        return cand
    return p


def evaluate_policy(policy_root: Path, policy_path: Path, pairs: PairBatch, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ev.require_jax_gpu_ready()
    policy_path = policy_path.expanduser().resolve()
    run_args = ev.load_run_args(policy_root)
    agent = ev.build_agent(int(pairs.obs.shape[-1]), run_args)
    payload = ev.load_policy_payload(policy_path)
    comp = ev._extract_guidance_components_from_payload(payload)
    if payload is None or comp is None:
        raise RuntimeError(f"policy checkpoint lacks diffusion policy/log_alpha/critic payload: {policy_path}")
    policy_obj = comp["policy_obj"]
    q_params = comp.get("target_q_params") or comp.get("q_params")
    critic_objective_id = int(comp.get("critic_objective_id", -1))
    lower_is_better = True if critic_objective_id < 0 else bool(critic_objective_id == 1)

    e_pos_all: List[np.ndarray] = []
    e_neg_all: List[np.ndarray] = []
    critic_chunks: Dict[str, List[np.ndarray]] = {}
    n = len(pairs.obs)
    for start in range(0, n, int(args.batch_size)):
        end = min(start + int(args.batch_size), n)
        e_pos, e_neg = _energy_pair_batch(
            agent,
            policy_obj,
            pairs.obs[start:end],
            pairs.a_pos[start:end],
            pairs.a_neg[start:end],
            samples=int(args.energy_samples),
            alpha=float(args.energy_alpha),
            seed=int(args.seed) + start,
        )
        e_pos_all.append(e_pos); e_neg_all.append(e_neg)
        if q_params is not None:
            cd = _critic_pair_batch(
                agent,
                q_params,
                pairs.obs[start:end],
                pairs.a_pos[start:end],
                pairs.a_neg[start:end],
                lower_is_better=lower_is_better,
                critic_margin=float(args.critic_margin),
            )
            for k, v in cd.items():
                critic_chunks.setdefault(k, []).append(v)
    e_pos = np.concatenate(e_pos_all, axis=0)
    e_neg = np.concatenate(e_neg_all, axis=0)
    energy_gap = e_neg - e_pos
    violation = np.maximum(float(args.er_margin) + e_pos - e_neg, 0.0)
    pairsat_e = (e_pos + float(args.er_margin) <= e_neg).astype(np.float32)
    iar = (e_neg <= e_pos).astype(np.float32)

    critic: Dict[str, np.ndarray] = {}
    for k, chunks in critic_chunks.items():
        critic[k] = np.concatenate(chunks, axis=0)

    rows: List[Dict[str, Any]] = []
    for i in range(n):
        row = {
            "policy_name": policy_path.name,
            "policy_path": str(policy_path),
            "pair_index": int(i),
            "source": pairs.sources[i],
            "source_index": int(pairs.indices[i]),
            "pair_gap": float(np.linalg.norm(pairs.a_pos[i] - pairs.a_neg[i])),
            "energy_pos": float(e_pos[i]),
            "energy_neg": float(e_neg[i]),
            "energy_gap_neg_minus_pos": float(energy_gap[i]),
            "pairsat_e": float(pairsat_e[i]),
            "v_e": float(violation[i]),
            "iar": float(iar[i]),
            "er_margin": float(args.er_margin),
        }
        for ck, cv in critic.items():
            row[ck] = float(cv[i])
        rows.append(row)

    def stat(prefix: str, values: Sequence[float], summary: Dict[str, Any]) -> None:
        summary[f"{prefix}_mean"] = _mean(values)
        summary[f"{prefix}_std"] = _std(values)
        summary[f"{prefix}_se"] = _stderr(values)

    summary: Dict[str, Any] = {
        "policy_name": policy_path.name,
        "policy_path": str(policy_path),
        "n_pairs": int(n),
        "pair_semantics": "true_intervention_only",
        "positive_action": str(args.positive_action),
        "er_margin": float(args.er_margin),
        "critic_margin": float(args.critic_margin),
        "energy_samples": int(args.energy_samples),
        "energy_alpha": float(args.energy_alpha),
        "critic_objective": "cost/lower_is_better" if lower_is_better else "value/higher_is_better",
    }
    stat("PairSat_E", pairsat_e, summary)
    stat("V_E", violation, summary)
    stat("IAR", iar, summary)
    stat("energy_gap_neg_minus_pos", energy_gap, summary)
    stat("energy_pos", e_pos, summary)
    stat("energy_neg", e_neg, summary)
    stat("pair_gap", [np.linalg.norm(pairs.a_pos[i] - pairs.a_neg[i]) for i in range(n)], summary)
    for ck, cv in critic.items():
        stat(ck, cv, summary)

    summary["tex_PairSat"] = _fmt_pm(summary["PairSat_E_mean"], summary["PairSat_E_std"], 3)
    summary["tex_V_E"] = _fmt_pm(summary["V_E_mean"], summary["V_E_std"], 4)
    summary["tex_IAR"] = _fmt_pm(summary["IAR_mean"], summary["IAR_std"], 3)
    if "delta_c_robust_mean" in summary:
        summary["tex_DeltaC_pair"] = _fmt_pm(summary["delta_c_robust_mean"], summary["delta_c_robust_std"], 4)
    else:
        summary["tex_DeltaC_pair"] = ""
    return rows, summary


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_markdown(path: Path, summaries: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Intervention-pair diagnostics\n\n")
        f.write("These metrics are computed from true intervention pairs, not from closed-loop autonomous episodes.\n\n")
        f.write("| policy | n | PairSat | V_E | IAR | DeltaC_pair | energy gap |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for s in summaries:
            f.write(
                f"| `{s.get('policy_name','')}` | {int(s.get('n_pairs', 0))} | "
                f"{s.get('tex_PairSat','')} | {s.get('tex_V_E','')} | {s.get('tex_IAR','')} | "
                f"{s.get('tex_DeltaC_pair','')} | "
                f"{_fmt_pm(float(s.get('energy_gap_neg_minus_pos_mean', float('nan'))), float(s.get('energy_gap_neg_minus_pos_std', float('nan'))), 4)} |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute DOVEER intervention-pair PairSat/V_E/IAR diagnostics.")
    ap.add_argument("--log_dir", type=Path, required=True, help="Training run directory or single policy .pkl.")
    ap.add_argument("--policy", "--policy_file", dest="policy", default=None, help="Optional policy path/name; if omitted, all policies under log_dir are evaluated.")
    ap.add_argument("--data_root", type=Path, required=True, help="Demo/replay pkl file or directory containing true intervention pairs.")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--max_pairs", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--energy_samples", type=int, default=4)
    ap.add_argument("--energy_alpha", type=float, default=1.0)
    ap.add_argument("--er_margin", type=float, default=0.05)
    ap.add_argument("--critic_margin", type=float, default=0.1, help="Diagnostic margin for conservative critic pair satisfaction.")
    ap.add_argument("--min_action_gap", type=float, default=1e-6)
    ap.add_argument("--positive_action", choices=["behavior", "human"], default="behavior")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.output_dir = args.output_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.log_dir = args.log_dir.expanduser()

    print(f"[data_root] {args.data_root}")
    pairs = load_pairs(
        args.data_root,
        positive_action=str(args.positive_action),
        min_action_gap=float(args.min_action_gap),
        max_pairs=int(args.max_pairs),
    )
    print(f"[pairs] loaded {len(pairs.obs)} true intervention pairs")

    policy_root, policy_files, _single = ev.resolve_policy_selection(args.log_dir, args.policy)
    if not policy_files:
        raise RuntimeError(f"No policy checkpoints found from log_dir={args.log_dir} policy={args.policy}")
    print(f"[policies] {len(policy_files)}")

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for idx, policy in enumerate(policy_files, start=1):
        print(f"[{idx}/{len(policy_files)}] {policy}", flush=True)
        rows, summary = evaluate_policy(policy_root, policy, pairs, args)
        all_rows.extend(rows)
        summaries.append(summary)
        print(
            f"    PairSat_E={summary['PairSat_E_mean']:.4f} "
            f"V_E={summary['V_E_mean']:.6f} IAR={summary['IAR_mean']:.4f}",
            flush=True,
        )

    pair_csv = args.output_dir / "intervention_pair_rows.csv"
    summary_csv = args.output_dir / "intervention_pair_summary.csv"
    summary_md = args.output_dir / "intervention_pair_summary.md"
    write_csv(pair_csv, all_rows)
    write_csv(summary_csv, summaries)
    write_markdown(summary_md, summaries)
    print(f"[wrote] {pair_csv}")
    print(f"[wrote] {summary_csv}")
    print(f"[wrote] {summary_md}")


if __name__ == "__main__":
    main()
