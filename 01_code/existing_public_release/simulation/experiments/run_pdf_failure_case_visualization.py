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


CASES = [
    ("no_guidance_failure", ["--guidance_mode", "none"], "no guidance: inspect unsafe drift/crash cases"),
    ("fixed_strong_guidance_failure", ["--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_lambda0", "1.0", "--guidance_beta_unc", "0.0"], "fixed strong guidance: inspect over-strong correction"),
    ("adaptive_guidance_success", ["--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_lambda0", "0.5", "--guidance_beta_unc", "1.0"], "adaptive guidance: inspect early correction"),
    ("high_uncertainty_adaptive", ["--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_lambda0", "0.5", "--guidance_beta_unc", "2.0"], "high uncertainty: inspect reduced guidance strength"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.10.4 failure/success case visualization command matrix.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--maps", default=str(TEST_MAP_SEEDS[0]))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_failure_visualization")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    maps = parse_int_csv(args.maps)
    rows: List[Dict[str, Any]] = []
    for case, extra, note in CASES:
        output_csv = args.output_dir / f"{case}.csv"
        eval_args: List[Any] = [
            "--log_dir", args.log_dir,
            "--policy", args.policy,
            "--maps", *maps,
            "--num_episodes", args.episodes,
            "--horizon", args.horizon,
            "--output_csv", output_csv,
            "--record_video",
            "--video_episodes", args.episodes,
            *extra,
        ]
        rows.append({
            "case": case,
            "note": note,
            "output_csv": str(output_csv),
            "command": bash_conda_command(
                DACER_ENV,
                python_cmd(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py", eval_args),
                exports={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        })
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_failure_visualization_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
