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
from map_splits import TRAIN_MAP_SEEDS


VARIANTS = [
    ("main_pair_semantics", [], "main: true intervention pairs only for EnergyRank/PV constraints"),
    ("merge_actions", ["--merge_action_semantics"], "collapse novice/human into behavior action"),
    ("pre_takeover_soft_context", ["--pre_takeover_window", "25", "--pre_takeover_bc_coef", "1.5", "--stage2_batch_pre_takeover_frac", "0.10"], "optional ablation: add pre-takeover soft context to BC/batch mix only"),
    ("no_stop_td_mask", ["--disable_stop_td_mask"], "do not mask TD at takeover/demo boundaries"),
    ("pv_on_all_expert_data", ["--pv_on_all_expert_data"], "intentionally over-label expert-like data as PV intervention"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.9.5 data semantics training ablation.")
    parser.add_argument("--demo_root", type=Path, default=PROJECT_ROOT / "data104")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--total_step", type=int, default=100000)
    parser.add_argument("--stage1b_updates", type=int, default=25000)
    parser.add_argument("--start_seed", type=int, default=TRAIN_MAP_SEEDS[0])
    parser.add_argument("--num_scenarios", type=int, default=len(TRAIN_MAP_SEEDS))
    parser.add_argument("--traffic_density", type=float, default=0.06)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_data_semantics_ablation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    rows: List[Dict[str, Any]] = []
    for variant, extra, note in VARIANTS:
        log_dir = args.output_dir / variant
        train_args: List[Any] = [
            "--seed", args.seed,
            "--start_seed", args.start_seed,
            "--num_scenarios", args.num_scenarios,
            "--traffic_density", args.traffic_density,
            "--total_step", args.total_step,
            "--stage1b_updates", args.stage1b_updates,
            "--demo_root", args.demo_root,
            "--log_dir", log_dir,
            "--controller", "keyboard",
            *extra,
        ]
        rows.append({
            "variant": variant,
            "demo_root": str(args.demo_root),
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
