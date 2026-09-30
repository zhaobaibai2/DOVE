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
    ("q_on_x0_every_step", "x0", "1", "clean_x0", "noise_level", "default: critic evaluates Tweedie clean-action estimate every denoising step"),
    ("legacy_latent_mean", "x0", "1", "latent_mean", "alpha_cumprod", "legacy implementation: backpropagate to x_t and inject into reverse mean"),
    ("q_on_xt_every_step", "xt", "1", "latent_mean", "noise_level", "OOD target test: critic evaluates noisy latent xt; uses latent injection"),
    ("x0_final_step_only", "x0", "0", "clean_x0", "noise_level", "only apply guidance at final denoising step"),
    ("x0_every_2_steps", "x0", "2", "clean_x0", "noise_level", "compute/effect tradeoff: apply every 2 steps"),
    ("x0_every_4_steps", "x0", "4", "clean_x0", "noise_level", "compute/effect tradeoff: apply every 4 steps"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.9.3 guidance target and schedule ablation.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--maps", default=",".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_guidance_target_ablation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    maps = parse_int_csv(args.maps)
    rows: List[Dict[str, Any]] = []
    for variant, target, step_interval, injection, schedule, note in VARIANTS:
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
            "--guidance_target", target,
            "--guidance_step_interval", step_interval,
            "--guidance_q_agg", "conservative",
            "--guidance_injection", injection,
            "--guidance_schedule", schedule,
            "--critic_objective", "lower",
        ]
        rows.append({
            "variant": variant,
            "guidance_target": target,
            "guidance_step_interval": step_interval,
            "guidance_injection": injection,
            "guidance_schedule": schedule,
            "note": note,
            "output_csv": str(output_csv),
            "command": bash_conda_command(
                DACER_ENV,
                python_cmd(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py", eval_args),
                exports={"CUDA_VISIBLE_DEVICES": "0"},
            ),
        })
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_guidance_target_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
