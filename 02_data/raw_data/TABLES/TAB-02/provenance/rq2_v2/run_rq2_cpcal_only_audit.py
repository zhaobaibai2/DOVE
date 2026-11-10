#!/usr/bin/env python3
"""Recompute the four RQ2 table diagnostics for the CPCal-only run.

Uses the frozen 256-pair RQ2 manifest, the manuscript's 64-draw VLB schedule,
and 64 unguided policy samples per pair/checkpoint.  Results are written per
checkpoint so an interrupted run can be inspected and resumed manually.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_elbo_faithfulness_audit as elbo
from experiments.run_pdf_mechanism_prior_aliasing import load_intervention_pairs
from scripts import eval_pvp_policies_fixed_v3 as ev


DEFAULT_POLICY_DIR = ROOT / (
    "logs2/energy_lambda_sweep_20260611_162012/train_lambda_er_0p0_margin_0p05/"
    "pvp_dacer_20260611_162024_s0"
)
DEFAULT_DEMO_ROOT = ROOT / (
    "logs/01_stage1a_collect_seed104_5pass_20260522_151529/"
    "pvp_dacer_20260522_151541_s0/data"
)
DEFAULT_REFERENCE_AUDIT = ROOT / "logs/121_elbo_faithfulness_audit_20step_64noise_20260810"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR)
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--reference-audit", type=Path, default=DEFAULT_REFERENCE_AUDIT)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sample-batch-size", type=int, default=8)
    ap.add_argument("--draws", type=int, default=64)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--noise-seed", type=int, default=20260810)
    ap.add_argument("--limit-checkpoints", type=int, default=0)
    args = ap.parse_args()

    if args.draws != 64 or args.samples != 64:
        raise ValueError("RQ2 table protocol requires exactly 64 VLB draws and 64 actor samples")
    if args.batch_size < 1 or args.sample_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    args.policy_dir = args.policy_dir.expanduser().resolve()
    args.demo_root = args.demo_root.expanduser().resolve()
    args.reference_audit = args.reference_audit.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ev.require_jax_gpu_ready()

    run_args = ev.load_run_args(args.policy_dir)
    if not (
        run_args.get("use_energy_rank") is False
        and float(run_args.get("lambda_er", 1.0)) == 0.0
        and float(run_args.get("lambda_pv_constraint", 0.0)) == 1.0
    ):
        raise RuntimeError("selected checkpoint run is not CPCal-only (ER off, PV/CPCal constraint on)")

    pairs = load_intervention_pairs(args.demo_root, 256, 1e-6)
    if len(pairs) != 256:
        raise RuntimeError(f"expected the RQ2 set of 256 intervention pairs, got {len(pairs)}")
    obs = np.stack([p[0] for p in pairs]).astype(np.float32)
    a_pos = np.stack([p[1] for p in pairs]).astype(np.float32)
    a_neg = np.stack([p[2] for p in pairs]).astype(np.float32)
    pair_ids = [
        (str(Path(p[3]).resolve().relative_to(ROOT)), int(p[4])) for p in pairs
    ]
    reference_pair_manifest = args.reference_audit / "pair_manifest.csv"
    if not reference_pair_manifest.is_file():
        raise FileNotFoundError(f"no frozen RQ2 pair manifest under {args.reference_audit}")
    if pair_ids != elbo.read_pair_ids(reference_pair_manifest):
        raise RuntimeError("loaded intervention pairs do not match the frozen RQ2 256-pair manifest")
    pair_hash = elbo.sha256_bytes(
        "\n".join(f"{source}:{index}" for source, index in pair_ids).encode("utf-8")
    )
    with (args.reference_audit / "audit_metadata.json").open(encoding="utf-8") as f:
        reference_meta = json.load(f)
    if pair_hash != reference_meta["pair_manifest_sha256"]:
        raise RuntimeError("pair manifest hash differs from the published RQ2 audit")

    policies = sorted(args.policy_dir.glob("policy-*.pkl"), key=elbo.policy_sort_key)
    if args.limit_checkpoints > 0:
        policies = policies[: args.limit_checkpoints]
    if not policies:
        raise FileNotFoundError(f"no policy-*.pkl checkpoints in {args.policy_dir}")

    # The original manuscript audits 46 checkpoints for Full DOVE; this run
    # has the same 1k--46k checkpoint horizon because its training stopped early.
    if args.limit_checkpoints == 0 and len(policies) != 46:
        raise RuntimeError(f"expected 46 saved checkpoints (1k--46k), found {len(policies)}")

    architecture = {
        k: run_args.get(k)
        for k in ("hidden_num", "hidden_dim", "diffusion_hidden_dim", "diffusion_steps", "action_noise_coef")
    }
    full_args_path = ROOT / "logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0/args.json"
    with full_args_path.open(encoding="utf-8") as f:
        full_args = json.load(f)
    full_arch = {k: full_args.get(k) for k in architecture}
    if architecture != full_arch:
        raise RuntimeError(f"CPCal-only architecture differs from Full DOVE: {architecture} != {full_arch}")

    rng = np.random.RandomState(args.noise_seed)
    noise = rng.standard_normal((256, 20, 64, a_pos.shape[-1])).astype(np.float32)
    noise_hash = elbo.sha256_bytes(noise.tobytes(order="C"))
    schedule = elbo.build_schedule(20, a_pos.shape[-1], "beta1_posterior")
    agent = ev.build_agent(obs.shape[-1], run_args)
    vlb_eval = elbo.build_eval_fn(agent, schedule["alphas_cumprod"])

    import jax
    import jax.numpy as jnp

    @jax.jit
    def critic_pair_acc(q_params, obs_b, pos_b, neg_b):
        q1p, _ = agent.q(q_params[0], obs_b, pos_b)
        q2p, _ = agent.q(q_params[1], obs_b, pos_b)
        q1n, _ = agent.q(q_params[0], obs_b, neg_b)
        q2n, _ = agent.q(q_params[1], obs_b, neg_b)
        qbar_p = 0.5 * (jnp.ravel(q1p) + jnp.ravel(q2p))
        qbar_n = 0.5 * (jnp.ravel(q1n) + jnp.ravel(q2n))
        # This run's critics model cost, so lower cost for accepted actions is correct.
        return (qbar_n - qbar_p > 0.0).astype(jnp.float32)

    @jax.jit
    def actor_samples(policy_obj, obs_b, keys_b):
        # obs_b is [B, 1, O] to match the established sample_policy_actions helper.
        return jax.vmap(
            lambda obs_one, keys_one: jax.vmap(
                lambda key: agent.get_action(key, policy_obj, obs_one)
            )(keys_one)
        )(obs_b, keys_b)

    # Match run_pdf_mechanism_prior_aliasing.sample_policy_actions: derive a
    # deterministic key from each intervention observation, then fold in k+12345.
    obs_for_sampling = jnp.asarray(obs[:, None, :], dtype=jnp.float32)
    base_keys = jax.vmap(ev.random_key_from_data)(obs_for_sampling)
    sample_ids = jnp.arange(64, dtype=jnp.uint32) + jnp.uint32(12345)
    key_grid = jax.vmap(
        lambda base: jax.vmap(lambda sid: jax.random.fold_in(base, sid))(sample_ids)
    )(base_keys)

    metadata = {
        "created_unix": time.time(),
        "variant": "CPCal-only (EnergyRank disabled)",
        "policy_dir": str(args.policy_dir),
        "policy_config": {k: run_args.get(k) for k in ("use_energy_rank", "lambda_er", "lambda_pv_constraint", "critic_objective")},
        "architecture": architecture,
        "checkpoint_count": len(policies),
        "checkpoint_names": [p.name for p in policies],
        "n_pairs": len(pairs),
        "samples_per_pair_checkpoint": 64,
        "vlb_noise_draws_per_timestep": 64,
        "diffusion_steps": 20,
        "decoder_variance": "beta1_posterior",
        "noise_seed": args.noise_seed,
        "noise_sha256": noise_hash,
        "pair_manifest_sha256": pair_hash,
        "pair_manifest_reference": str(reference_pair_manifest),
        "reference_vlb_audit": str(args.reference_audit),
        "actor_sampling_key": "random_key_from_data(obs), fold_in(k+12345), matching sample_policy_actions",
        "aggregation": "equal-weight mean over checkpoints; each checkpoint uses 256 pairs and 64 actor samples per pair",
    }
    write_json(args.output_dir / "audit_metadata.json", metadata)
    write_csv(
        args.output_dir / "pair_manifest.csv",
        [
            {"pair_id": i, "source": src, "index": idx, "a_pos_0": a_pos[i, 0], "a_pos_1": a_pos[i, 1], "a_neg_0": a_neg[i, 0], "a_neg_1": a_neg[i, 1]}
            for i, (src, idx) in enumerate(pair_ids)
        ],
    )
    print(
        f"START CPCal-only checkpoints={len(policies)} pairs={len(pairs)} "
        f"pair_sha256={pair_hash} noise_sha256={noise_hash}",
        flush=True,
    )

    checkpoint_rows: List[Dict[str, Any]] = []
    for ckpt_idx, policy_path in enumerate(policies, start=1):
        started = time.time()
        payload = ev.load_policy_payload(policy_path)
        comp = ev._extract_guidance_components_from_payload(payload)
        if payload is None or comp is None:
            raise RuntimeError(f"cannot load actor/critic parameters from {policy_path}")
        policy_obj = comp["policy_obj"]
        q_params = comp.get("target_q_params") or comp.get("q_params")
        if q_params is None:
            raise RuntimeError(f"checkpoint has no twin critics: {policy_path}")

        pos_parts: List[np.ndarray] = []
        neg_parts: List[np.ndarray] = []
        for begin in range(0, len(pairs), args.batch_size):
            end = min(begin + args.batch_size, len(pairs))
            ep, en = vlb_eval(
                policy_obj,
                obs[begin:end],
                a_pos[begin:end],
                a_neg[begin:end],
                noise[begin:end],
            )
            pos_parts.append(np.asarray(jax.device_get(ep), dtype=np.float64))
            neg_parts.append(np.asarray(jax.device_get(en), dtype=np.float64))
        e_pos = np.concatenate(pos_parts, axis=0)
        e_neg = np.concatenate(neg_parts, axis=0)
        vlb = elbo.compute_pair_metrics(
            e_neg - e_pos, a_pos, a_neg, schedule, 0.05, "l2"
        )["vlb_ok"].astype(np.int8)

        pairacc_chunks: List[np.ndarray] = []
        for begin in range(0, len(pairs), 64):
            end = min(begin + 64, len(pairs))
            acc = critic_pair_acc(
                q_params,
                jnp.asarray(obs[begin:end]),
                jnp.asarray(a_pos[begin:end]),
                jnp.asarray(a_neg[begin:end]),
            )
            pairacc_chunks.append(np.asarray(jax.device_get(acc), dtype=np.int8))
        pairacc = np.concatenate(pairacc_chunks)

        hsr_counts = np.zeros((len(pairs),), dtype=np.int32)
        rmr_counts = np.zeros((len(pairs),), dtype=np.int32)
        gap_vec = a_pos - a_neg
        gap_norm_sq = np.sum(gap_vec * gap_vec, axis=1)
        gap_norm = np.sqrt(gap_norm_sq)
        midpoint = 0.5 * (a_pos + a_neg)
        for begin in range(0, len(pairs), args.sample_batch_size):
            end = min(begin + args.sample_batch_size, len(pairs))
            actions = actor_samples(
                policy_obj,
                obs_for_sampling[begin:end],
                key_grid[begin:end],
            )
            actions = np.asarray(jax.device_get(actions), dtype=np.float32)[:, :, 0, :]
            actions = np.clip(actions, -1.0, 1.0)
            projection = np.sum(
                (actions - midpoint[begin:end, None, :]) * gap_vec[begin:end, None, :],
                axis=-1,
            ) / gap_norm_sq[begin:end, None]
            hsr_counts[begin:end] = np.sum(projection > 0.0, axis=1)
            # Rejected-mass rate is mass near the rejected/novice action a^-.
            dist_rejected = np.linalg.norm(actions - a_neg[begin:end, None, :], axis=-1)
            rmr_counts[begin:end] = np.sum(
                dist_rejected < 0.5 * gap_norm[begin:end, None], axis=1
            )

        pair_rows = []
        for i, (source, index) in enumerate(pair_ids):
            pair_rows.append(
                {
                    "pair_id": i,
                    "source": source,
                    "index": index,
                    "rho_vlb_indicator": int(vlb[i]),
                    "hsr_count_of_64": int(hsr_counts[i]),
                    "hsr_rate": float(hsr_counts[i] / 64.0),
                    "rmr_0p5_count_of_64": int(rmr_counts[i]),
                    "rmr_0p5_rate": float(rmr_counts[i] / 64.0),
                    "pairacc_c_indicator": int(pairacc[i]),
                }
            )
        pair_csv = args.output_dir / "checkpoints" / f"{policy_path.stem}_pairs.csv"
        write_csv(pair_csv, pair_rows)
        summary = {
            "variant": metadata["variant"],
            "checkpoint": policy_path.name,
            "checkpoint_path": str(policy_path),
            "checkpoint_sha256": sha256_file(policy_path),
            "pair_csv": str(pair_csv),
            "n_pairs": len(pairs),
            "actor_samples": len(pairs) * 64,
            "rho_vlb": float(np.mean(vlb)),
            "hsr": float(np.sum(hsr_counts) / (len(pairs) * 64)),
            "rmr_0p5": float(np.sum(rmr_counts) / (len(pairs) * 64)),
            "pairacc_c_mean": float(np.mean(pairacc)),
            "elapsed_seconds": time.time() - started,
        }
        write_json(args.output_dir / "checkpoints" / f"{policy_path.stem}_summary.json", summary)
        checkpoint_rows.append(summary)
        write_csv(args.output_dir / "checkpoint_metrics.csv", checkpoint_rows)
        print(
            f"DONE {ckpt_idx}/{len(policies)} {policy_path.stem} "
            f"rho_VLB={summary['rho_vlb']:.6f} HSR={summary['hsr']:.6f} "
            f"RMR0.5={summary['rmr_0p5']:.6f} PairAccC={summary['pairacc_c_mean']:.6f} "
            f"seconds={summary['elapsed_seconds']:.1f}",
            flush=True,
        )

    columns = ("rho_vlb", "hsr", "rmr_0p5", "pairacc_c_mean")
    metric_labels = {
        "rho_vlb": ("rho_VLB", 256, "pair-checkpoint fraction"),
        "hsr": ("HSR", 256 * 64, "unguided-sample fraction"),
        "rmr_0p5": ("RMR_0.5", 256 * 64, "unguided-sample fraction"),
        "pairacc_c_mean": ("PairAcc_C_mean", 256, "pair-checkpoint fraction"),
    }
    long_rows: List[Dict[str, Any]] = []
    all_pair_rows: List[Dict[str, Any]] = []
    for row in checkpoint_rows:
        for col in columns:
            label, denominator, unit = metric_labels[col]
            long_rows.append(
                {
                    "checkpoint": row["checkpoint"],
                    "metric": label,
                    "value": row[col],
                    "denominator_per_checkpoint": denominator,
                    "unit": unit,
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
            )
        pair_path = Path(row["pair_csv"])
        with pair_path.open("r", newline="", encoding="utf-8") as f:
            for pair_row in csv.DictReader(f):
                all_pair_rows.append({"checkpoint": row["checkpoint"], **pair_row})
    write_csv(args.output_dir / "checkpoint_metrics_long.csv", long_rows)
    write_csv(args.output_dir / "checkpoint_pair_metrics.csv", all_pair_rows)

    aggregate = {
        "variant": metadata["variant"],
        "checkpoints": len(checkpoint_rows),
        "pair_checkpoint_rows": len(checkpoint_rows) * len(pairs),
        "actor_samples": len(checkpoint_rows) * len(pairs) * 64,
    }
    for col in columns:
        values = np.asarray([row[col] for row in checkpoint_rows], dtype=np.float64)
        aggregate[col] = float(np.mean(values))
        aggregate[f"{col}_se_over_checkpoints"] = float(
            np.std(values, ddof=1) / np.sqrt(len(values))
        ) if len(values) > 1 else float("nan")
    write_csv(args.output_dir / "aggregate_metrics.csv", [aggregate])
    tex = (
        "Full \\DOVE{} w/o EnergyRank & "
        f"{aggregate['rho_vlb']:.3f} & {aggregate['hsr']:.3f} & "
        f"{aggregate['rmr_0p5']:.3f} & {aggregate['pairacc_c_mean']:.3f}\\\\\n"
    )
    tmp_tex = args.output_dir / "table_row.tex.tmp"
    tmp_tex.write_text(tex, encoding="utf-8")
    os.replace(tmp_tex, args.output_dir / "table_row.tex")
    write_json(args.output_dir / "audit_summary.json", aggregate)
    print(f"COMPLETE {args.output_dir / 'aggregate_metrics.csv'}", flush=True)


if __name__ == "__main__":
    main()
