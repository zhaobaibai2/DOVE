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
    DEFAULT_LOG_DIR,
    DEFAULT_POLICY,
    bash_conda_command,
    parse_int_csv,
    print_written,
    python_cmd,
    write_command_files,
)
from map_splits import TEST_MAP_SEEDS


VARIANTS = [
    ("target_conservative_double_q", "target", "conservative", "TeX-aligned objective-aware conservative target: UCB for risk critics, LCB for value critics"),
    ("target_ucb_double_q", "target", "ucb", "lower-is-better exact UCB proxy-risk ablation: max over twin critics"),
    ("target_lcb_double_q", "target", "lcb", "optimistic proxy-risk aggregation ablation"),
    ("target_min_double_q", "target", "min", "conservative min-Q target critics"),
    ("online_min_double_q", "online", "min", "online critics, checks stability against target critics"),
    ("target_single_q1", "target", "single_q1", "single critic, checks overestimation sensitivity"),
    ("target_mean_double_q", "target", "mean", "mean double-Q, less conservative than min"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.9.4 critic choice ablation.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--maps", default=",".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_critic_choice_ablation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    maps = parse_int_csv(args.maps)
    rows: List[Dict[str, Any]] = []
    for variant, critic_params, q_agg, note in VARIANTS:
        output_csv = args.output_dir / f"{variant}.csv"
        eval_args: List[Any] = [
            "--log_dir", args.log_dir,
            "--policy", args.policy,
            "--maps", *maps,
            "--num_episodes", args.episodes,
            "--horizon", args.horizon,
            "--output_csv", output_csv,
            "--use_value_guidance",
            "--guidance_mode", "proxy_value",
            "--guidance_critic_params", critic_params,
            "--guidance_q_agg", q_agg,
            "--critic_objective", "lower",
            "--guidance_injection", "clean_x0",
            "--guidance_schedule", "noise_level",
        ]
        rows.append({
            "variant": variant,
            "critic_params": critic_params,
            "q_agg": q_agg,
            "note": note,
            "output_csv": str(output_csv),
            "command": bash_conda_command(
                DACER_ENV,
                python_cmd(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py", eval_args),
                exports={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        })
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_critic_choice_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
