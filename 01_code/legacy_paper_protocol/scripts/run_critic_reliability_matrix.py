#!/usr/bin/env python3
"""Run closed-loop critic reliability evaluation over all saved policy checkpoints."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple


VARIANTS: Tuple[Tuple[str, str], ...] = (
    ("dbp_no_er", "logs/11_train_pvp_dbp_critic_seed104_50000_20260521_204732/pvp_dacer_20260521_204744_s0"),
    ("full", "logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0"),
    ("no_cpc", "logs/13_train_doveer_er_no_pv_constraint_seed104_50000_20260522_152214/pvp_dacer_20260522_152226_s0"),
)


def _policy_step(path: Path) -> int:
    match = re.search(r"policy-(\d+)-", path.name)
    return int(match.group(1)) if match else 10**18


def list_tasks(repo_root: Path) -> List[Dict[str, str]]:
    tasks: List[Dict[str, str]] = []
    for variant, rel_dir in VARIANTS:
        policy_dir = repo_root / rel_dir
        policies = sorted(policy_dir.glob("policy-*.pkl"), key=_policy_step)
        for policy in policies:
            tasks.append(
                {
                    "variant": variant,
                    "policy_dir": str(policy_dir),
                    "policy_name": policy.name,
                    "policy_stem": policy.stem,
                }
            )
    return tasks


def write_manifest(tasks: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "policy_dir", "policy_name", "policy_stem"])
        writer.writeheader()
        writer.writerows(tasks)


def run_task(
    repo_root: Path,
    task: Dict[str, str],
    out_root: Path,
    worker_id: int,
    args: argparse.Namespace,
) -> Dict[str, str]:
    variant = task["variant"]
    policy_stem = task["policy_stem"]
    out_dir = out_root / variant / policy_stem
    eval_csv = out_dir / "eval_upvgds_final_gate_full.csv"
    episode_csv = out_dir / "episodes_upvgds_final_gate_full.csv"
    step_csv = out_dir / "steps_upvgds_final_gate_full.csv"
    console_log = out_dir / "console.log"
    status_path = out_dir / "status.txt"

    if step_csv.exists() and not args.overwrite:
        return {
            **task,
            "status": "skipped_existing",
            "returncode": "0",
            "step_csv": str(step_csv),
            "console_log": str(console_log),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    status_path.write_text(f"running worker={worker_id} start={time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    env["JAX_PLATFORMS"] = "cuda"
    env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.xla_mem_fraction)

    cmd = [
        sys.executable,
        "scripts/eval_pvp_policies_fixed_v3.py",
        "--log_dir", task["policy_dir"],
        "--policy", task["policy_name"],
        "--output_csv", str(eval_csv),
        "--episode_output_csv", str(episode_csv),
        "--step_output_csv", str(step_csv),
        "--reliability_horizon", str(args.reliability_horizon),
        "--maps", *[str(m) for m in args.maps],
        "--num_episodes", str(args.num_episodes),
        "--step_diagnostics_stride", str(args.step_diagnostics_stride),
        "--no_use_render",
        "--use_value_guidance",
        "--guidance_lambda0", "0.5",
        "--guidance_beta_unc", "1.0",
        "--guidance_p_decay", "1.0",
        "--guidance_grad_clip", "1.0",
        "--guidance_mode", "proxy_value",
        "--guidance_q_agg", "conservative",
        "--guidance_injection", "clean_x0",
        "--guidance_schedule", "noise_level",
        "--guidance_target", "x0",
        "--use_final_gate",
        "--final_gate_mode", "proxy_prior",
        "--final_gate_proxy_compare", "robust",
        "--final_gate_min_proxy_gain", "0.0",
        "--final_gate_max_action_shift", "0.2",
        "--final_gate_tau_prior", "0.02",
        "--final_gate_tau_u", "1.0",
        "--final_gate_project_guidance", "1",
        "--final_gate_energy_samples", str(args.final_gate_energy_samples),
    ]

    started = time.time()
    with console_log.open("w", encoding="utf-8") as log_f:
        log_f.write("[cmd] " + " ".join(cmd) + "\n")
        log_f.flush()
        proc = subprocess.run(cmd, cwd=repo_root, env=env, stdout=log_f, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started
    status = "ok" if proc.returncode == 0 and step_csv.exists() else "failed"
    status_path.write_text(
        f"{status} worker={worker_id} returncode={proc.returncode} elapsed_sec={elapsed:.1f} end={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )
    return {
        **task,
        "status": status,
        "returncode": str(proc.returncode),
        "elapsed_sec": f"{elapsed:.1f}",
        "step_csv": str(step_csv),
        "console_log": str(console_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_episodes", type=int, default=30)
    parser.add_argument("--maps", type=int, nargs="+", default=[104])
    parser.add_argument("--reliability_horizon", type=int, default=25)
    parser.add_argument("--step_diagnostics_stride", type=int, default=1)
    parser.add_argument("--final_gate_energy_samples", type=int, default=1)
    parser.add_argument("--cuda_visible_devices", default="0")
    parser.add_argument("--xla_mem_fraction", type=float, default=0.45)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--missing_only", action="store_true",
                        help="only submit tasks whose step-level reliability CSV is missing")
    args = parser.parse_args()

    if args.num_workers <= 0:
        raise ValueError("--num_workers must be positive")

    repo_root = Path(__file__).resolve().parents[1]
    out_root = args.output_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    tasks = list_tasks(repo_root)
    if args.missing_only and not args.overwrite:
        tasks = [
            task for task in tasks
            if not (
                out_root
                / task["variant"]
                / task["policy_stem"]
                / "steps_upvgds_final_gate_full.csv"
            ).exists()
        ]
    write_manifest(tasks, out_root / "task_manifest.csv")

    print(f"[matrix] tasks={len(tasks)} workers={args.num_workers} episodes={args.num_episodes} maps={args.maps}", flush=True)
    print(f"[matrix] output={out_root}", flush=True)

    results: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=int(args.num_workers)) as executor:
        futures = {
            executor.submit(run_task, repo_root, task, out_root, idx % int(args.num_workers), args): task
            for idx, task in enumerate(tasks)
        }
        for done_idx, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(
                f"[{done_idx:03d}/{len(tasks):03d}] {result['status']} "
                f"{result['variant']} {result['policy_name']} rc={result['returncode']}",
                flush=True,
            )
            with (out_root / "results_so_far.csv").open("w", newline="", encoding="utf-8") as f:
                fieldnames = sorted({key for row in results for key in row.keys()})
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

    with (out_root / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({key for row in results for key in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    failed = [row for row in results if row.get("status") == "failed"]
    print(f"[matrix] complete ok={len(results) - len(failed)} failed={len(failed)}", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
