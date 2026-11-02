#!/usr/bin/env python3
"""Schedule-resolved audit of the EnergyRank-to-DDPM-VLB proposition.

This is an offline diagnostic.  It never changes a policy checkpoint.  For every
intervention pair it evaluates all 20 diffusion steps with a fixed bank of 64
noise draws shared by the human/rejected actions and by all policy checkpoints.

Two residual conventions are emitted from the same forward passes:
  * l2: literal manuscript definition ||epsilon-epsilon_theta||_2^2;
  * mse: implementation definition mean_j (epsilon_j-epsilon_theta,j)^2.
For the MSE convention the DDPM weights are multiplied by action dimension so
that Delta_VLB is identical to the literal L2 calculation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pdf_mechanism_prior_aliasing import load_intervention_pairs
from scripts import eval_pvp_policies_fixed_v3 as ev


GROUPS = {
    "positive_only": {
        "prefix": "run11_noer",
        "log_dir": "logs/11_train_pvp_dbp_critic_seed104_50000_20260521_204732/pvp_dacer_20260521_204744_s0",
    },
    "er_without_cpcal": {
        "prefix": "run13_eronly",
        "log_dir": "logs/13_train_doveer_er_no_pv_constraint_seed104_50000_20260522_152214/pvp_dacer_20260522_152226_s0",
    },
    "full_dove": {
        "prefix": "run12_full",
        "log_dir": "logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0",
    },
}


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(fields or rows[0].keys())
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def policy_sort_key(path: Path) -> Tuple[int, int, str]:
    m = re.fullmatch(r"policy-(\d+)-(\d+)\.pkl", path.name)
    if m is None:
        return (10**18, 10**18, path.name)
    return (int(m.group(1)), int(m.group(2)), path.name)


def discover_checkpoints(old_audit_dir: Path, group_name: str) -> List[Path]:
    cfg = GROUPS[group_name]
    prefix = str(cfg["prefix"])
    log_dir = ROOT / str(cfg["log_dir"])
    pair_files = sorted(old_audit_dir.glob(f"{prefix}_policy-*_pairs.csv"))
    policies: List[Path] = []
    for pair_file in pair_files:
        stem = pair_file.name[len(prefix) + 1 : -len("_pairs.csv")]
        policy = log_dir / f"{stem}.pkl"
        if not policy.is_file():
            raise FileNotFoundError(f"checkpoint named by old audit is missing: {policy}")
        policies.append(policy)
    policies = sorted(set(policies), key=policy_sort_key)
    expected = {"positive_only": 50, "er_without_cpcal": 50, "full_dove": 46}[group_name]
    if len(policies) != expected:
        raise RuntimeError(f"{group_name}: expected {expected} valid old-audit checkpoints, got {len(policies)}")
    return policies


def read_pair_ids(path: Path) -> List[Tuple[str, int]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return [(row["source"], int(row["index"])) for row in csv.DictReader(f)]


def build_schedule(num_timesteps: int, act_dim: int, decoder_variance: str) -> Dict[str, Any]:
    from relax.utils.diffusion import BetaScheduleCoefficients

    betas = np.asarray(BetaScheduleCoefficients.vp_beta_schedule(num_timesteps), dtype=np.float64)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    abar_prev = np.r_[1.0, abar[:-1]]
    posterior = betas * (1.0 - abar_prev) / (1.0 - abar)
    if decoder_variance == "beta1_posterior":
        sigma2 = posterior.copy()
        sigma2[0] = betas[0]
    elif decoder_variance == "fixed_beta":
        sigma2 = betas.copy()
    else:
        raise ValueError(decoder_variance)
    if not np.all(np.isfinite(sigma2)) or not np.all(sigma2 > 0):
        raise RuntimeError(f"invalid reverse variances: {sigma2}")
    weights = betas**2 / (2.0 * sigma2 * alphas * (1.0 - abar))
    bar_w = float(np.mean(weights))
    d_w = float(np.sum(np.abs(weights - bar_w)))
    lambda_t = float(abar[-1] / 2.0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": abar,
        "posterior_variance": posterior,
        "sigma2": sigma2,
        "weights": weights,
        "bar_w": bar_w,
        "d_w": d_w,
        "lambda_t": lambda_t,
        "terminal_bound": lambda_t * act_dim,
    }


def build_eval_fn(agent, alphas_cumprod: np.ndarray):
    import jax
    import jax.numpy as jnp

    sqrt_abar = jnp.asarray(np.sqrt(alphas_cumprod), dtype=jnp.float32)
    sqrt_one_minus = jnp.asarray(np.sqrt(1.0 - alphas_cumprod), dtype=jnp.float32)
    timesteps = jnp.arange(len(alphas_cumprod), dtype=jnp.int32)

    @jax.jit
    def evaluate(policy_obj, obs, a_pos, a_neg, noise):
        # obs [B,O], actions [B,D], noise [B,T,M,D]
        b, t_count, draws, act_dim = noise.shape
        obs_rep = jnp.broadcast_to(obs[:, None, None, :], (b, t_count, draws, obs.shape[-1]))
        t_rep = jnp.broadcast_to(timesteps[None, :, None], (b, t_count, draws))
        pos_rep = jnp.broadcast_to(a_pos[:, None, None, :], noise.shape)
        neg_rep = jnp.broadcast_to(a_neg[:, None, None, :], noise.shape)
        c1 = sqrt_abar[None, :, None, None]
        c2 = sqrt_one_minus[None, :, None, None]
        x_pos = c1 * pos_rep + c2 * noise
        x_neg = c1 * neg_rep + c2 * noise
        flat_obs = obs_rep.reshape((-1, obs.shape[-1]))
        flat_t = t_rep.reshape((-1,))
        flat_noise = noise.reshape((-1, act_dim))
        pred_pos = agent.predict_noise(policy_obj, flat_obs, flat_t, x_pos.reshape((-1, act_dim)))
        pred_neg = agent.predict_noise(policy_obj, flat_obs, flat_t, x_neg.reshape((-1, act_dim)))
        sq_pos = (pred_pos - flat_noise) ** 2
        sq_neg = (pred_neg - flat_noise) ** 2
        mse_pos = jnp.mean(sq_pos, axis=-1).reshape((b, t_count, draws)).mean(axis=-1)
        mse_neg = jnp.mean(sq_neg, axis=-1).reshape((b, t_count, draws)).mean(axis=-1)
        return mse_pos, mse_neg

    return evaluate


def compute_pair_metrics(
    g_mse: np.ndarray,
    a_pos: np.ndarray,
    a_neg: np.ndarray,
    schedule: Dict[str, Any],
    margin: float,
    convention: str,
) -> Dict[str, np.ndarray]:
    act_dim = int(a_pos.shape[-1])
    weights = np.asarray(schedule["weights"], dtype=np.float64)
    bar_w = float(schedule["bar_w"])
    d_w = float(schedule["d_w"])
    lambda_t = float(schedule["lambda_t"])
    if convention == "l2":
        g = np.asarray(g_mse, dtype=np.float64) * act_dim
        effective_weights = weights
        effective_bar_w = bar_w
        effective_d_w = d_w
    elif convention == "mse":
        g = np.asarray(g_mse, dtype=np.float64)
        # VLB contains a sum over action dimensions, while training uses mean.
        effective_weights = weights * act_dim
        effective_bar_w = bar_w * act_dim
        effective_d_w = d_w * act_dim
    else:
        raise ValueError(convention)
    bar_g = np.mean(g, axis=1)
    d_g = np.max(np.abs(g - bar_g[:, None]), axis=1)
    delta_mis = effective_d_w * d_g + lambda_t * act_dim
    threshold = len(weights) * effective_bar_w * float(margin)
    terminal_delta = lambda_t * (
        np.sum(np.asarray(a_neg, dtype=np.float64) ** 2, axis=1)
        - np.sum(np.asarray(a_pos, dtype=np.float64) ** 2, axis=1)
    )
    delta_vlb = np.sum(effective_weights[None, :] * g, axis=1) + terminal_delta
    pair_sat = bar_g >= float(margin)
    mismatch_ok = delta_mis < threshold
    cert = pair_sat & mismatch_ok
    return {
        "g": g,
        "bar_g": bar_g,
        "d_g": d_g,
        "delta_mis": delta_mis,
        "threshold": np.full_like(bar_g, threshold),
        "cert_slack": threshold - delta_mis,
        "realized_bound_slack": len(weights) * effective_bar_w * bar_g - delta_mis,
        "terminal_delta": terminal_delta,
        "delta_vlb": delta_vlb,
        "pairsat": pair_sat.astype(np.int8),
        "mismatch_ok": mismatch_ok.astype(np.int8),
        "cert": cert.astype(np.int8),
        "vlb_ok": (delta_vlb > 0.0).astype(np.int8),
    }


def summarize_checkpoint(metrics: Dict[str, np.ndarray]) -> Dict[str, Any]:
    n = len(metrics["pairsat"])
    rank_n = int(np.sum(metrics["pairsat"]))
    cert_n = int(np.sum(metrics["cert"]))
    return {
        "n_pairs": n,
        "rank_n": rank_n,
        "cert_n": cert_n,
        "pairsat_e": rank_n / n,
        "rho_mis_given_rank": cert_n / rank_n if rank_n else float("nan"),
        "rho_cert": cert_n / n,
        "rho_vlb": float(np.mean(metrics["vlb_ok"])),
        "cert_slack_median": float(np.median(metrics["cert_slack"])),
        "cert_slack_q25": float(np.quantile(metrics["cert_slack"], 0.25)),
        "cert_slack_q75": float(np.quantile(metrics["cert_slack"], 0.75)),
        "delta_vlb_median": float(np.median(metrics["delta_vlb"])),
        "delta_vlb_q25": float(np.quantile(metrics["delta_vlb"], 0.25)),
        "delta_vlb_q75": float(np.quantile(metrics["delta_vlb"], 0.75)),
    }


def se(values: Iterable[float]) -> float:
    x = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    return float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0


def aggregate(output_dir: Path, conventions: Sequence[str]) -> None:
    checkpoint_rows: List[Dict[str, Any]] = []
    pair_rows_by_key: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for summary_path in sorted(output_dir.glob("checkpoints/*/*_summary.json")):
        row = read_json(summary_path)
        checkpoint_rows.append(row)
        pair_path = Path(row["pair_csv"])
        with pair_path.open("r", newline="", encoding="utf-8") as f:
            pair_rows_by_key.setdefault((row["group"], row["convention"]), []).extend(csv.DictReader(f))
    if not checkpoint_rows:
        raise RuntimeError("no checkpoint summaries found for aggregation")
    write_csv(output_dir / "checkpoint_metrics.csv", checkpoint_rows)
    aggregate_rows: List[Dict[str, Any]] = []
    for group in GROUPS:
        for convention in conventions:
            rows = [r for r in checkpoint_rows if r["group"] == group and r["convention"] == convention]
            if not rows:
                continue
            pairs = pair_rows_by_key[(group, convention)]
            rank_total = sum(int(float(r["pairsat"])) for r in pairs)
            cert_total = sum(int(float(r["cert"])) for r in pairs)
            slack = np.asarray([float(r["cert_slack"]) for r in pairs], dtype=np.float64)
            delta = np.asarray([float(r["delta_vlb"]) for r in pairs], dtype=np.float64)
            finite_conditional = [float(r["rho_mis_given_rank"]) for r in rows
                                  if np.isfinite(float(r["rho_mis_given_rank"]))]
            out: Dict[str, Any] = {
                "group": group,
                "convention": convention,
                "checkpoints": len(rows),
                "pair_checkpoint_rows": len(pairs),
                "pairsat_e_mean": float(np.mean([r["pairsat_e"] for r in rows])),
                "pairsat_e_se": se(r["pairsat_e"] for r in rows),
                "rho_mis_given_rank_pooled": cert_total / rank_total if rank_total else float("nan"),
                "rho_mis_given_rank_checkpoint_mean": float(np.mean(finite_conditional)) if finite_conditional else float("nan"),
                "rho_mis_given_rank_checkpoint_se": se(r["rho_mis_given_rank"] for r in rows),
                "rho_cert_mean": float(np.mean([r["rho_cert"] for r in rows])),
                "rho_cert_se": se(r["rho_cert"] for r in rows),
                "rho_vlb_mean": float(np.mean([r["rho_vlb"] for r in rows])),
                "rho_vlb_se": se(r["rho_vlb"] for r in rows),
                "cert_slack_median": float(np.median(slack)),
                "cert_slack_q25": float(np.quantile(slack, 0.25)),
                "cert_slack_q75": float(np.quantile(slack, 0.75)),
                "delta_vlb_median": float(np.median(delta)),
                "delta_vlb_q25": float(np.quantile(delta, 0.25)),
                "delta_vlb_q75": float(np.quantile(delta, 0.75)),
                "cert_implies_vlb_violations": sum(
                    int(float(r["cert"]) > 0.5 and float(r["vlb_ok"]) < 0.5) for r in pairs
                ),
            }
            aggregate_rows.append(out)
    write_csv(output_dir / "aggregate_metrics.csv", aggregate_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-audit-dir", type=Path, default=ROOT / "logs/110_actor_prior_audit_gpu6_cvci_safe_20260604_115410")
    ap.add_argument("--demo-root", type=Path, default=ROOT / "logs/01_stage1a_collect_seed104_5pass_20260522_151529/pvp_dacer_20260522_151541_s0/data")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=20260810)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--decoder-variance", choices=["beta1_posterior", "fixed_beta"], default="beta1_posterior")
    ap.add_argument("--groups", nargs="+", choices=list(GROUPS), default=list(GROUPS))
    ap.add_argument("--limit-checkpoints", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if args.draws < 64:
        raise ValueError("the preregistered audit requires at least 64 draws per timestep")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ev.require_jax_gpu_ready()

    pairs = load_intervention_pairs(args.demo_root, 256, 1e-6)
    if len(pairs) != 256:
        raise RuntimeError(f"expected 256 intervention pairs, got {len(pairs)}")
    obs = np.stack([x[0] for x in pairs]).astype(np.float32)
    a_pos = np.stack([x[1] for x in pairs]).astype(np.float32)
    a_neg = np.stack([x[2] for x in pairs]).astype(np.float32)
    # Normalize source names to the relative representation used by the frozen
    # RQ2 CSVs.  The resolved files and row indices are still checked exactly.
    pair_ids = [
        (str(Path(x[3]).resolve().relative_to(ROOT)), int(x[4])) for x in pairs
    ]
    ref_candidates = sorted(args.old_audit_dir.glob("run11_noer_policy-*_pairs.csv"))
    if not ref_candidates:
        raise FileNotFoundError("cannot find old RQ2 pair manifest")
    old_pair_ids = read_pair_ids(ref_candidates[0])
    if pair_ids != old_pair_ids:
        raise RuntimeError("new loader does not reproduce the old 256-pair manifest exactly")

    rng = np.random.RandomState(args.noise_seed)
    noise = rng.standard_normal((len(pairs), 20, args.draws, a_pos.shape[-1])).astype(np.float32)
    noise_hash = sha256_bytes(noise.tobytes(order="C"))
    pair_manifest_hash = sha256_bytes("\n".join(f"{p}:{i}" for p, i in pair_ids).encode("utf-8"))

    first_group = args.groups[0]
    first_args = read_json(ROOT / str(GROUPS[first_group]["log_dir"]) / "args.json")
    num_timesteps = int(first_args.get("diffusion_steps", 20))
    if num_timesteps != 20:
        raise RuntimeError(f"expected 20 diffusion steps, got {num_timesteps}")
    if noise.shape[1] != num_timesteps:
        raise RuntimeError("noise bank timestep dimension mismatch")
    architecture = {k: first_args.get(k) for k in ("hidden_num", "hidden_dim", "diffusion_hidden_dim", "diffusion_steps", "action_noise_coef")}
    for group in args.groups[1:]:
        other = read_json(ROOT / str(GROUPS[group]["log_dir"]) / "args.json")
        other_arch = {k: other.get(k) for k in architecture}
        if other_arch != architecture:
            raise RuntimeError(f"architecture mismatch for {group}: {other_arch} != {architecture}")

    schedule = build_schedule(num_timesteps, a_pos.shape[-1], args.decoder_variance)
    agent = ev.build_agent(obs.shape[-1], first_args)
    eval_fn = build_eval_fn(agent, schedule["alphas_cumprod"])
    metadata = {
        "created_unix": time.time(),
        "root": str(ROOT),
        "groups": args.groups,
        "draws_per_timestep": args.draws,
        "noise_seed": args.noise_seed,
        "noise_sha256": noise_hash,
        "pair_manifest_sha256": pair_manifest_hash,
        "n_pairs": len(pairs),
        "num_timesteps": num_timesteps,
        "action_dim": int(a_pos.shape[-1]),
        "margin": args.margin,
        "decoder_variance": args.decoder_variance,
        "architecture": architecture,
        "schedule": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in schedule.items()},
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }
    write_json(args.output_dir / "audit_metadata.json", metadata)
    write_csv(args.output_dir / "pair_manifest.csv", [
        {"pair_id": i, "source": src, "index": idx, "a_pos_0": a_pos[i, 0], "a_pos_1": a_pos[i, 1], "a_neg_0": a_neg[i, 0], "a_neg_1": a_neg[i, 1]}
        for i, (src, idx) in enumerate(pair_ids)
    ])
    print(f"AUDIT metadata={args.output_dir / 'audit_metadata.json'} noise_sha256={noise_hash}", flush=True)
    print(f"SCHEDULE bar_w={schedule['bar_w']:.12g} D_w={schedule['d_w']:.12g} lambda_T={schedule['lambda_t']:.12g}", flush=True)

    import jax

    for group in args.groups:
        policies = discover_checkpoints(args.old_audit_dir, group)
        if args.limit_checkpoints > 0:
            policies = policies[: args.limit_checkpoints]
        print(f"GROUP {group} checkpoints={len(policies)}", flush=True)
        for cidx, policy_path in enumerate(policies, start=1):
            ckpt_tag = policy_path.stem
            ckpt_dir = args.output_dir / "checkpoints" / group
            done_marker = ckpt_dir / f"{ckpt_tag}_l2_summary.json"
            if done_marker.exists() and not args.overwrite:
                print(f"SKIP {group} {ckpt_tag} existing={done_marker}", flush=True)
                continue
            payload = ev.load_policy_payload(policy_path)
            components = ev._extract_guidance_components_from_payload(payload)
            if payload is None or components is None:
                raise RuntimeError(f"cannot extract policy from {policy_path}")
            policy_obj = components["policy_obj"]
            pos_parts: List[np.ndarray] = []
            neg_parts: List[np.ndarray] = []
            started = time.time()
            for begin in range(0, len(pairs), args.batch_size):
                end = min(begin + args.batch_size, len(pairs))
                ep, en = eval_fn(policy_obj, obs[begin:end], a_pos[begin:end], a_neg[begin:end], noise[begin:end])
                pos_parts.append(np.asarray(jax.device_get(ep), dtype=np.float64))
                neg_parts.append(np.asarray(jax.device_get(en), dtype=np.float64))
            e_pos_mse = np.concatenate(pos_parts, axis=0)
            e_neg_mse = np.concatenate(neg_parts, axis=0)
            if e_pos_mse.shape != (256, 20) or not np.all(np.isfinite(e_pos_mse)) or not np.all(np.isfinite(e_neg_mse)):
                raise RuntimeError(f"invalid energy arrays for {policy_path}: {e_pos_mse.shape}")
            g_mse = e_neg_mse - e_pos_mse
            for convention in ("l2", "mse"):
                metrics = compute_pair_metrics(g_mse, a_pos, a_neg, schedule, args.margin, convention)
                pair_rows: List[Dict[str, Any]] = []
                for i, (source, index) in enumerate(pair_ids):
                    row: Dict[str, Any] = {
                        "pair_id": i, "source": source, "index": index,
                        "bar_g": metrics["bar_g"][i], "d_g": metrics["d_g"][i],
                        "delta_mis": metrics["delta_mis"][i], "threshold": metrics["threshold"][i],
                        "cert_slack": metrics["cert_slack"][i], "realized_bound_slack": metrics["realized_bound_slack"][i],
                        "terminal_delta": metrics["terminal_delta"][i], "delta_vlb": metrics["delta_vlb"][i],
                        "pairsat": int(metrics["pairsat"][i]), "mismatch_ok": int(metrics["mismatch_ok"][i]),
                        "cert": int(metrics["cert"][i]), "vlb_ok": int(metrics["vlb_ok"][i]),
                    }
                    for t_idx in range(num_timesteps):
                        row[f"g_t{t_idx + 1:02d}"] = metrics["g"][i, t_idx]
                    pair_rows.append(row)
                pair_csv = ckpt_dir / f"{ckpt_tag}_{convention}_pairs.csv"
                write_csv(pair_csv, pair_rows)
                summary = {
                    "group": group, "checkpoint": ckpt_tag, "checkpoint_path": str(policy_path),
                    "checkpoint_sha256": sha256_file(policy_path), "convention": convention,
                    "pair_csv": str(pair_csv), "pair_csv_sha256": sha256_file(pair_csv),
                    "noise_sha256": noise_hash, "pair_manifest_sha256": pair_manifest_hash,
                    "draws_per_timestep": args.draws, "num_timesteps": num_timesteps,
                    "decoder_variance": args.decoder_variance, "margin": args.margin,
                    "elapsed_seconds": time.time() - started,
                    **summarize_checkpoint(metrics),
                }
                write_json(ckpt_dir / f"{ckpt_tag}_{convention}_summary.json", summary)
            print(f"DONE {group} {cidx}/{len(policies)} {ckpt_tag} seconds={time.time()-started:.2f}", flush=True)
            aggregate(args.output_dir, ("l2", "mse"))
    aggregate(args.output_dir, ("l2", "mse"))
    print(f"COMPLETE aggregate={args.output_dir / 'aggregate_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
