from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pdf_experiment_utils import (
    DACER_ENV,
    PROJECT_ROOT,
    bash_conda_command,
    print_written,
    python_cmd,
    write_command_files,
)
from experiments.dove_paper_protocol import PROTOCOL


VARIANTS = [
    ("main_pair_semantics", [], "main: true intervention pairs only for EnergyRank/PV constraints"),
    ("merge_actions", ["--merge_action_semantics"], "collapse novice/human into behavior action"),
    ("pre_takeover_soft_context", ["--pre_takeover_window", "25", "--pre_takeover_bc_coef", "1.5", "--stage2_batch_pre_takeover_frac", "0.10"], "optional ablation: add pre-takeover soft context to BC/batch mix only"),
    ("no_stop_td_mask", ["--disable_stop_td_mask"], "do not mask TD at takeover/demo boundaries"),
    ("pv_on_all_expert_data", ["--pv_on_all_expert_data"], "intentionally over-label expert-like data as PV intervention"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.9.5 data semantics training ablation.")
    parser.add_argument("--demo_root_template", default=str(PROJECT_ROOT / "data_seed_{seed}"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PROTOCOL.training_seeds))
    parser.add_argument("--total_step", type=int, default=PROTOCOL.total_interaction_budget)
    parser.add_argument("--stage1b_updates", type=int, default=PROTOCOL.behavior_prior_updates)
    parser.add_argument("--start_seed", type=int, default=PROTOCOL.training_maps[0])
    parser.add_argument("--num_scenarios", type=int, default=len(PROTOCOL.training_maps))
    parser.add_argument("--traffic_density", type=float, default=0.06)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_data_semantics_ablation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    if tuple(args.seeds) != PROTOCOL.training_seeds:
        raise ValueError(f"paper protocol seeds must be exactly {list(PROTOCOL.training_seeds)}")
    map_ids = tuple(range(args.start_seed, args.start_seed + args.num_scenarios))
    if map_ids != PROTOCOL.training_maps:
        raise ValueError(f"paper protocol training maps must be {list(PROTOCOL.training_maps)}")
    if args.total_step != PROTOCOL.total_interaction_budget:
        raise ValueError(f"paper protocol interaction budget must be {PROTOCOL.total_interaction_budget}")

    rows: List[Dict[str, Any]] = []
    for seed in args.seeds:
      for variant, extra, note in VARIANTS:
        demo_root = Path(args.demo_root_template.format(seed=seed)).expanduser()
        log_dir = args.output_dir / f"{variant}_s{seed}"
        train_args: List[Any] = [
            "--seed", seed,
            "--start_seed", args.start_seed,
            "--num_scenarios", args.num_scenarios,
            "--traffic_density", args.traffic_density,
            "--total_step", args.total_step,
            "--total_interaction_budget", PROTOCOL.total_interaction_budget,
            "--stage1b_updates", args.stage1b_updates,
            "--demo_root", demo_root,
            "--log_dir", log_dir,
            "--controller", "keyboard",
            *extra,
        ]
        rows.append({
            "variant": variant,
            "seed": seed,
            "training_maps": " ".join(map(str, PROTOCOL.training_maps)),
            "demo_root": str(demo_root),
            "log_dir": str(log_dir),
            "note": note,
            "command": bash_conda_command(
                DACER_ENV,
                python_cmd(ROOT / "scripts" / "train_pvp_dacer_metadrive_off.py", train_args),
                exports={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        })
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_data_semantics_training_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
