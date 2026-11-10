#!/usr/bin/env python3
"""Regenerate missing HSR/RMR counts by frozen-policy inference (no training)."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

from common import HERE, digest, read_csv, verify_inputs, write_csv, write_json


RUNS = {
    "positive_only": "logs/11_train_pvp_dbp_critic_seed104_50000_20260521_204732/pvp_dacer_20260521_204744_s0",
    "er_without_cpcal": "logs/13_train_doveer_er_no_pv_constraint_seed104_50000_20260522_152214/pvp_dacer_20260522_152226_s0",
    "full_dove": "logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0",
}


def sample_counts(actions, positive, negative):
    """Paper HSR and RMR_0.5; strict inequalities and normalized action space."""
    gap = positive - negative
    norm_sq = np.sum(gap * gap, axis=-1)
    if np.any(norm_sq <= 0):
        raise ValueError("Zero action gap cannot define halfspace diagnostics")
    midpoint = (positive + negative) * .5
    projection = np.sum((actions - midpoint[:, None, :]) * gap[:, None, :], axis=-1) / norm_sq[:, None]
    distances = np.linalg.norm(actions - negative[:, None, :], axis=-1)
    return (projection > 0).sum(axis=1), (distances < .5 * np.sqrt(norm_sq)[:, None]).sum(axis=1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="original project containing saved checkpoints and demo PKLs")
    parser.add_argument("--simulation-root", type=Path, default=HERE.parent)
    parser.add_argument("--output-dir", type=Path, default=HERE / "outputs/resampled_actor")
    parser.add_argument("--groups", nargs="+", choices=list(RUNS), default=list(RUNS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit-checkpoints", type=int, default=0, help="smoke test only; cannot validate a full paper row")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit_checkpoints < 0:
        parser.error("invalid batch size or checkpoint limit")
    verify_inputs()
    root = args.source_root.expanduser().resolve()
    sim = args.simulation_root.expanduser().resolve()
    sys.path.insert(0, str(sim))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    from scripts import eval_pvp_policies_fixed_v3 as ev
    import jax
    import jax.numpy as jnp

    backend, devices = ev.require_jax_gpu_ready()
    manifest = read_csv(HERE / "inputs/table_03/pair_manifest.csv")
    observations, positives, negatives, observation_sources = [], [], [], {}
    for row in manifest:
        path = (root / row["source"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Demo source is outside --source-root")
        if path not in observation_sources:
            with path.open("rb") as stream:
                payload = pickle.load(stream)
            if "raw_data" in payload and "observation" not in payload:
                payload = payload["raw_data"]
            sequence = payload.get("observation", payload.get("obs"))
            if sequence is None:
                raise ValueError(f"No observations: {path}")
            observation_sources[path] = sequence
        observations.append(observation_sources[path][int(row["index"])])
        positives.append([float(row["a_pos_0"]), float(row["a_pos_1"])])
        negatives.append([float(row["a_neg_0"]), float(row["a_neg_1"])])
    obs = np.asarray(observations, dtype=np.float32)
    pos = np.asarray(positives, dtype=np.float32)
    neg = np.asarray(negatives, dtype=np.float32)
    if obs.shape[0] != 256:
        raise ValueError("The frozen manifest must contain 256 pairs")
    obs_b = jnp.asarray(obs[:, None, :])
    base_keys = jax.vmap(ev.random_key_from_data)(obs_b)
    ids = jnp.arange(64, dtype=jnp.uint32) + jnp.uint32(12345)
    key_grid = jax.vmap(lambda base: jax.vmap(lambda i: jax.random.fold_in(base, i))(ids))(base_keys)
    # Verify checkpoint identities against the frozen VLB audit before inference.
    old_metrics = read_csv(root / "logs/121_elbo_faithfulness_audit_20step_64noise_20260810/checkpoint_metrics.csv")
    expected_hashes = {(r["group"], r["checkpoint"]): r["checkpoint_sha256"] for r in old_metrics}
    for group in args.groups:
        run = root / RUNS[group]
        run_args = ev.load_run_args(run)
        agent = ev.build_agent(obs.shape[-1], run_args)

        @jax.jit
        def draw(policy_obj, ob, keys):
            return jax.vmap(lambda o, ks: jax.vmap(lambda key: agent.get_action(key, policy_obj, o))(ks))(ob, keys)

        files = sorted((HERE / "inputs/table_03/vlb" / group).glob("*_mse_pairs.csv"))
        if args.limit_checkpoints:
            files = files[:args.limit_checkpoints]
        meta = dict(group=group, samples_per_pair=64, n_pairs=256, checkpoint_count=len(files),
                    full_protocol=not bool(args.limit_checkpoints), backend=backend, devices=[str(d) for d in devices],
                    pair_manifest_sha256=digest(HERE / "inputs/table_03/pair_manifest.csv"),
                    sampling_key="random_key_from_data(obs[None,:]); fold_in(sample_index+12345)",
                    demo_source_sha256={str(p.relative_to(root)): digest(p) for p in observation_sources})
        write_json(args.output_dir / group / "metadata.json", meta)
        for i, file in enumerate(files, 1):
            stem = file.name.removesuffix("_mse_pairs.csv")
            checkpoint = run / (stem + ".pkl")
            model_hash = digest(checkpoint)
            if expected_hashes.get((group, stem)) != model_hash:
                raise ValueError(f"Checkpoint differs from frozen VLB audit: {checkpoint}")
            payload = ev.load_policy_payload(checkpoint)
            components = ev._extract_guidance_components_from_payload(payload)
            if components is None:
                raise ValueError(f"Checkpoint cannot be loaded: {checkpoint}")
            hsr, rmr = [], []
            for start in range(0, len(obs), args.batch_size):
                end = min(start + args.batch_size, len(obs))
                actions = np.asarray(jax.device_get(draw(components["policy_obj"], obs_b[start:end], key_grid[start:end])), dtype=np.float32)[:, :, 0, :]
                h, r = sample_counts(np.clip(actions, -1., 1.), pos[start:end], neg[start:end])
                hsr.extend(h.tolist())
                rmr.extend(r.tolist())
            rows = [dict(pair_id=j, source=row["source"], index=row["index"], hsr_count_of_64=hsr[j],
                         rmr_0p5_count_of_64=rmr[j], checkpoint_sha256=model_hash) for j, row in enumerate(manifest)]
            write_csv(args.output_dir / group / (stem + "_samples.csv"), rows)
            print(f"{group} {i}/{len(files)} {stem}: HSR={sum(hsr)/(256*64):.6f} RMR={sum(rmr)/(256*64):.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
