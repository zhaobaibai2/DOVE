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


def build_eval(
    variant: str,
    log_dir: Path,
    policy: str,
    maps: List[int],
    episodes: int,
    horizon: int,
    output_dir: Path,
    extra: List[Any],
    expected: str,
) -> Dict[str, Any]:
    output_csv = output_dir / f"{variant}.csv"
    args: List[Any] = [
        "--log_dir", log_dir,
        "--policy", policy,
        "--maps", *maps,
        "--num_episodes", episodes,
        "--horizon", horizon,
        "--output_csv", output_csv,
        *extra,
    ]
    return {
        "variant": variant,
        "expected": expected,
        "output_csv": str(output_csv),
        "command": bash_conda_command(
            DACER_ENV,
            python_cmd(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py", args),
            exports={"CUDA_VISIBLE_DEVICES": "0"},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.9.1 guidance source ablation.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--maps", default=",".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_guidance_source_ablation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    maps = parse_int_csv(args.maps)
    rows = [
        build_eval("none", args.log_dir, args.policy, maps, args.episodes, args.horizon, args.output_dir, ["--guidance_mode", "none"], "baseline"),
        build_eval("random_grad", args.log_dir, args.policy, maps, args.episodes, args.horizon, args.output_dir, ["--use_value_guidance", "--guidance_mode", "random_grad"], "should not improve consistently"),
        build_eval("opposite_signed_proxy_value", args.log_dir, args.policy, maps, args.episodes, args.horizon, args.output_dir, ["--use_value_guidance", "--guidance_mode", "reverse_grad"], "opposite-signed diagnostic"),
        build_eval("proxy_value_fixed", args.log_dir, args.policy, maps, args.episodes, args.horizon, args.output_dir, ["--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_beta_unc", "0.0"], "value guidance without uncertainty"),
        build_eval("proxy_value_uncertainty", args.log_dir, args.policy, maps, args.episodes, args.horizon, args.output_dir, ["--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_beta_unc", "1.0"], "full UPV-GDS"),
    ]
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_guidance_source_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
