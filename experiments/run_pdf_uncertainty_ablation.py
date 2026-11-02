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
    ("fixed_weak", "0.25", "0.0", "weak fixed guidance, no std gate"),
    ("fixed_strong", "1.0", "0.0", "strong fixed guidance, no std gate"),
    ("adaptive_mild", "0.5", "0.5", "mild uncertainty gate"),
    ("adaptive_default", "0.5", "1.0", "default uncertainty gate"),
    ("adaptive_strong", "0.5", "2.0", "strong uncertainty gate"),
    ("no_std_gate", "0.5", "0.0", "same as beta_unc=0; isolates std gate removal"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Surrogate-guidance uncertainty attenuation/gate ablation.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--maps", default=",".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_uncertainty_ablation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    maps = parse_int_csv(args.maps)
    rows: List[Dict[str, Any]] = []
    for variant, lambda0, beta_unc, note in VARIANTS:
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
            "--guidance_lambda0", lambda0,
            "--guidance_beta_unc", beta_unc,
            "--guidance_q_agg", "conservative",
            "--guidance_injection", "clean_x0",
            "--guidance_schedule", "noise_level",
            "--critic_objective", "lower",
        ]
        rows.append({
            "variant": variant,
            "lambda0": lambda0,
            "beta_unc": beta_unc,
            "note": note,
            "output_csv": str(output_csv),
            "command": bash_conda_command(
                DACER_ENV,
                python_cmd(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py", eval_args),
                exports={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        })
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_uncertainty_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
