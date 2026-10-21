"""Generate (but never execute) the final paper's five-seed training tasks."""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dove_paper_protocol import PAPER_METHOD_NAMES, PROTOCOL
from experiments.pdf_experiment_utils import DACER_ENV, bash_conda_command


METHODS = tuple(PAPER_METHOD_NAMES)
READY_METHODS = {
    "dove",
    "positive_only_diffusion_reference",
    "er_without_cpcal",
}


def parse_list(raw: str, allowed: Iterable[str]) -> List[str]:
    allowed_tuple = tuple(allowed)
    if raw == "all":
        return list(allowed_tuple)
    items = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(items) - set(allowed_tuple))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; allowed {sorted(allowed_tuple)}")
    return items


def _common_dove_args(method: str, seed: int, demo_root: Path) -> List[str]:
    return [
        "python", "-u", str(ROOT / "scripts" / "train_pvp_dacer_metadrive_off.py"),
        "--demo_root", str(demo_root),
        "--seed", str(seed),
        "--start_seed", str(PROTOCOL.training_maps[0]),
        "--num_scenarios", str(len(PROTOCOL.training_maps)),
        "--demo_passes_per_map", str(PROTOCOL.initial_collection_episodes_per_map),
        "--total_step", str(PROTOCOL.total_interaction_budget),
        "--total_interaction_budget", str(PROTOCOL.total_interaction_budget),
        "--stage1b_updates", str(PROTOCOL.behavior_prior_updates),
        "--stage2_critic_warmup_steps", str(PROTOCOL.critic_warmup_updates),
        "--stage2_rl_warmup_steps", "0",
        "--stage2_rl_ramp_steps", "0",
        "--diffusion_steps", str(PROTOCOL.diffusion_steps),
        "--hidden_num", str(PROTOCOL.critic_hidden_layers),
        "--hidden_dim", str(PROTOCOL.critic_hidden_width),
        "--diffusion_hidden_dim", str(PROTOCOL.diffusion_hidden_width),
        "--batch_size", str(PROTOCOL.batch_size),
        "--lr", str(PROTOCOL.critic_learning_rate),
        "--actor_lr", str(PROTOCOL.actor_learning_rate),
        "--gamma", str(PROTOCOL.discount),
        "--tau", str(PROTOCOL.polyak_tau),
        "--actor_delay", str(PROTOCOL.actor_update_interval),
        "--lambda_bc", str(PROTOCOL.lambda_bc),
        "--lambda_rl", str(PROTOCOL.lambda_rl),
        "--lambda_reg", str(PROTOCOL.lambda_reg),
        "--lambda_pv", str(PROTOCOL.lambda_pv),
        "--B", str(PROTOCOL.proxy_bound),
        "--lambda_qreg", str(PROTOCOL.lambda_q),
        "--er_margin", str(PROTOCOL.energy_rank_margin),
        "--er_budget", str(PROTOCOL.energy_constraint_slack),
        "--er_dual_lr", str(PROTOCOL.energy_dual_learning_rate),
        "--er_eta_init", str(PROTOCOL.energy_dual_initial_value),
        "--er_min_action_gap", str(PROTOCOL.pair_retention_threshold),
        "--pv_constraint_margin", str(PROTOCOL.cpcal_margin),
        "--save_policy_every", "1000",
        "--log_dir", str(ROOT / "logs" / f"paper_{method}_seed_{seed}"),
    ]


def command_for(method: str, seed: int, demo_root: Path) -> Dict[str, Any]:
    status = "ready"
    note = ""
    if method in READY_METHODS:
        argv = _common_dove_args(method, seed, demo_root)
        if method == "dove":
            argv += [
                "--policy_mode", "hybrid_dacer",
                "--use_energy_rank", "--lambda_er", "1.0", "--er_use_primal_dual",
                "--lambda_pv_constraint", str(PROTOCOL.lambda_cal),
            ]
        elif method == "er_without_cpcal":
            argv += [
                "--policy_mode", "hybrid_dacer",
                "--use_energy_rank", "--lambda_er", "1.0", "--er_use_primal_dual",
                "--lambda_pv_constraint", "0.0",
            ]
        else:
            argv += [
                "--policy_mode", "pvp_paper", "--lambda_rl", "0.0",
                "--lambda_pv", "0.0", "--lambda_pv_constraint", "0.0",
            ]
        command = bash_conda_command(
            DACER_ENV,
            argv,
            exports={"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": str(ROOT)},
        )
        note = "Command generated only; training is intentionally not started."
    else:
        status = "adapter_required"
        command = ""
        note = (
            f"{PAPER_METHOD_NAMES[method]} source exists outside this isolated release, "
            "but no verified adapter currently enforces the identical DOVE map/budget interface."
        )

    return {
        "task_id": f"train_{method}_seed_{seed}",
        "paper_method": PAPER_METHOD_NAMES[method],
        "method_key": method,
        "training_seed": seed,
        "training_maps": " ".join(map(str, PROTOCOL.training_maps)),
        "heldout_maps": " ".join(map(str, PROTOCOL.heldout_maps)),
        "total_interaction_budget": PROTOCOL.total_interaction_budget,
        "status": status,
        "demo_root": str(demo_root),
        "command": command,
        "note": note,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-aligned DOVE/ablation/baseline training tasks; does not run them."
    )
    parser.add_argument("--methods", default="all", help=f"Comma list from {','.join(METHODS)}, or all")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PROTOCOL.training_seeds))
    parser.add_argument("--demo_root_template", default="/path/to/stage1a/seed_{seed}")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "paper_training_matrix")
    args = parser.parse_args()

    if tuple(args.seeds) != PROTOCOL.training_seeds:
        raise ValueError(f"paper protocol seeds must be exactly {list(PROTOCOL.training_seeds)}")
    methods = parse_list(args.methods, METHODS)
    rows = [
        command_for(
            method,
            seed,
            Path(args.demo_root_template.format(seed=seed)).expanduser(),
        )
        for seed in args.seeds
        for method in methods
    ]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paper_training_tasks.csv"
    sh_path = output_dir / "paper_training_tasks.sh"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with sh_path.open("w", encoding="utf-8") as stream:
        stream.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for row in rows:
            stream.write(f"# {row['task_id']} | {row['paper_method']} | {row['status']}\n")
            if row["command"]:
                stream.write(row["command"] + "\n\n")
            else:
                stream.write("# " + row["note"] + "\n\n")

    print(f"generated {len(rows)} tasks ({sum(r['status'] == 'ready' for r in rows)} ready)")
    print(f"wrote {csv_path}")
    print(f"wrote {sh_path}")
    print("No training or evaluation was executed.")


if __name__ == "__main__":
    main()
