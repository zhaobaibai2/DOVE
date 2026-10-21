from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pdf_experiment_utils import DACER_ENV, PROJECT_ROOT, bash_conda_command
from experiments.dove_paper_protocol import PROTOCOL


COMMON_HYBRID_COST_ARGS = [
    "--policy_mode", "hybrid_dacer",
    "--critic_objective", "cost",
]

ER_STATIC_ARGS = [
    "--use_energy_rank",
    "--lambda_er", "1.0",
    "--er_margin", "0.05",
]

ER_PRIMAL_DUAL_ARGS = [
    *ER_STATIC_ARGS,
    "--er_use_primal_dual",
    "--er_budget", str(PROTOCOL.energy_constraint_slack),
    "--er_dual_lr", "1e-3",
    "--er_eta_init", "1.0",
]

UPVGDS_NO_UNC_ARGS = [
    "--use_value_guidance",
    "--guidance_mode", "proxy_value",
    "--guidance_beta_unc", "0.0",
    "--guidance_q_agg", "conservative",
    "--guidance_injection", "clean_x0",
    "--guidance_schedule", "noise_level",
]

UPVGDS_UNC_ARGS = [
    "--use_value_guidance",
    "--guidance_mode", "proxy_value",
    "--guidance_beta_unc", "1.0",
    "--guidance_q_agg", "conservative",
    "--guidance_injection", "clean_x0",
    "--guidance_schedule", "noise_level",
]


ABLATIONS = {
    "pvp_dacer_no_guidance": [
        *COMMON_HYBRID_COST_ARGS,
        "--guidance_mode", "none",
    ],
    "pvp_dacer_upvgds_no_unc": [
        *COMMON_HYBRID_COST_ARGS,
        *UPVGDS_NO_UNC_ARGS,
    ],

    # Main TeX-aligned constrained DOVE-ER variants.
    "dove_er_constrained_pd_only": [
        *COMMON_HYBRID_COST_ARGS,
        *ER_PRIMAL_DUAL_ARGS,
        "--guidance_mode", "none",
    ],
    "dove_er_constrained_pd_upvgds_no_unc": [
        *COMMON_HYBRID_COST_ARGS,
        *ER_PRIMAL_DUAL_ARGS,
        *UPVGDS_NO_UNC_ARGS,
    ],
    "dove_er_constrained_pd_upvgds_unc": [
        *COMMON_HYBRID_COST_ARGS,
        *ER_PRIMAL_DUAL_ARGS,
        *UPVGDS_UNC_ARGS,
    ],

    # Static Lagrangian relaxation ablations: useful to show why the primal-dual
    # interpretation is more than just a fixed extra loss weight.
    "dove_er_static_lagrangian_only": [
        *COMMON_HYBRID_COST_ARGS,
        *ER_STATIC_ARGS,
        "--guidance_mode", "none",
    ],
    "dove_er_static_lagrangian_upvgds_unc": [
        *COMMON_HYBRID_COST_ARGS,
        *ER_STATIC_ARGS,
        *UPVGDS_UNC_ARGS,
    ],

    # Mechanism ablations under the constrained DOVE-ER training objective.
    "dove_er_no_pv": [
        *COMMON_HYBRID_COST_ARGS,
        "--lambda_pv", "0.0",
        *ER_PRIMAL_DUAL_ARGS,
        "--guidance_mode", "none",
    ],
    "dove_er_no_shared_pair_filter": [
        *COMMON_HYBRID_COST_ARGS,
        *ER_PRIMAL_DUAL_ARGS,
        "--er_min_action_gap", "0.0",
        "--guidance_mode", "none",
    ],
    "dove_er_positive_human": [
        *COMMON_HYBRID_COST_ARGS,
        *ER_PRIMAL_DUAL_ARGS,
        "--er_positive_action", "human",
        "--guidance_mode", "none",
    ],
}

ABLATION_REASONS = {
    "pvp_dacer_no_guidance": "PVP-DACER baseline with lower-is-better proxy-risk critic and no inference guidance",
    "pvp_dacer_upvgds_no_unc": "PVP-DACER plus TeX-conservative UPV-GDS without uncertainty attenuation",
    "dove_er_constrained_pd_only": "Main TeX-aligned training constraint: EnergyRank primal-dual actor update without inference-time guidance",
    "dove_er_constrained_pd_upvgds_no_unc": "Constrained DOVE-ER plus TeX-conservative UPV-GDS without uncertainty attenuation",
    "dove_er_constrained_pd_upvgds_unc": "Full constrained DOVE-ER training plus uncertainty-aware TeX-conservative UPV-GDS",
    "dove_er_static_lagrangian_only": "Static fixed-weight Lagrangian relaxation ablation for EnergyRank only",
    "dove_er_static_lagrangian_upvgds_unc": "Static fixed-weight EnergyRank with UPV-GDS; compares against primal-dual constrained variant",
    "dove_er_no_pv": "Constrained EnergyRank without PV critic anchors",
    "dove_er_no_shared_pair_filter": "Constrained EnergyRank pair-filter ablation",
    "dove_er_positive_human": "Constrained EnergyRank positive-action ablation using a_h instead of executed behavior",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PVP-DACER/UPV-GDS training ablation commands.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PROTOCOL.training_seeds))
    parser.add_argument("--total_step", type=int, default=PROTOCOL.total_interaction_budget)
    parser.add_argument("--start_seed", type=int, default=PROTOCOL.training_maps[0])
    parser.add_argument("--num_scenarios", type=int, default=len(PROTOCOL.training_maps))
    parser.add_argument("--demo_root_template", default=str(PROJECT_ROOT / "data_seed_{seed}"))
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "policy_ablation_training")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    if tuple(args.seeds) != PROTOCOL.training_seeds:
        raise ValueError(f"paper protocol seeds must be exactly {list(PROTOCOL.training_seeds)}")
    map_ids = tuple(range(args.start_seed, args.start_seed + args.num_scenarios))
    if map_ids != PROTOCOL.training_maps:
        raise ValueError(f"paper protocol training maps must be {list(PROTOCOL.training_maps)}")
    if args.total_step != PROTOCOL.total_interaction_budget:
        raise ValueError(f"paper protocol interaction budget must be {PROTOCOL.total_interaction_budget}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_pair_semantics = [
        "--pre_takeover_window", "0",
        "--pre_takeover_bc_coef", "0.0",
        "--pre_takeover_pv_coef", "0.0",
        "--stage2_batch_pre_takeover_frac", "0.0",
    ]

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
      for name, extra in ABLATIONS.items():
        demo_root = Path(args.demo_root_template.format(seed=seed)).expanduser()
        log_dir = args.output_dir / f"{name}_s{seed}"
        cmd = [
            "python", str(ROOT / "scripts" / "train_pvp_dacer_metadrive_off.py"),
            "--seed", str(seed),
            "--start_seed", str(args.start_seed),
            "--num_scenarios", str(args.num_scenarios),
            "--total_step", str(args.total_step),
            "--total_interaction_budget", str(PROTOCOL.total_interaction_budget),
            "--demo_root", str(demo_root),
            "--log_dir", str(log_dir),
            "--controller", "keyboard",
            *main_pair_semantics,
            *extra,
        ]
        rows.append({
            "ablation": name,
            "seed": seed,
            "training_maps": " ".join(map(str, PROTOCOL.training_maps)),
            "total_step": args.total_step,
            "demo_root": str(demo_root),
            "log_dir": str(log_dir),
            "command": bash_conda_command(DACER_ENV, cmd, exports={"CUDA_VISIBLE_DEVICES": "0"}),
            "reason": ABLATION_REASONS[name],
        })

    out_csv = args.output_dir / "policy_ablation_training_commands.csv"
    out_sh = args.output_dir / "policy_ablation_training_commands.sh"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with out_sh.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n\n")
        for row in rows:
            f.write(f"# {row['ablation']}: {row['reason']}\n{row['command']}\n\n")
    for row in rows:
        print(row["command"])
    print(f"wrote {out_csv}")
    print(f"wrote {out_sh}")


if __name__ == "__main__":
    main()
