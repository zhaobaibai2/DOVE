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
    PROJECT_ROOT,
    bash_conda_command,
    print_written,
    python_cmd,
    write_command_files,
)
from map_splits import TEST_MAP_SEEDS


SCRIPTS = [
    ("scene_protocol", "run_pdf_scene_protocol.py"),
    ("closed_loop_main", "run_pdf_closed_loop_main.py"),
    ("guidance_source", "run_pdf_guidance_source_ablation.py"),
    ("uncertainty", "run_pdf_uncertainty_ablation.py"),
    ("final_gate", "run_pdf_final_gate_ablation.py"),
    ("guidance_target", "run_pdf_guidance_target_ablation.py"),
    ("critic_choice", "run_pdf_critic_choice_ablation.py"),
    ("data_semantics", "run_pdf_data_semantics_ablation.py"),
    ("mechanism_action_shift", "run_pdf_mechanism_action_shift.py"),
    ("mechanism_proxy_risk_drop", "run_pdf_mechanism_proxy_risk_drop.py"),
    ("mechanism_uncertainty_gate", "run_pdf_mechanism_uncertainty_gate.py"),
    ("failure_visualization", "run_pdf_failure_case_visualization.py"),
    ("aggregate_tables", "aggregate_pdf_metrics.py"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate absolute commands for all PDF-defined experiments.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--demo_root", type=Path, default=PROJECT_ROOT / "data104")
    parser.add_argument("--maps", default=",".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_full_suite")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    rows: List[Dict[str, Any]] = []
    for label, script_name in SCRIPTS:
        script = ROOT / "experiments" / script_name
        script_args: List[Any] = []
        if label not in {"scene_protocol", "data_semantics", "aggregate_tables"}:
            script_args.extend(["--log_dir", args.log_dir, "--policy", args.policy])
        if label in {"closed_loop_main", "guidance_source", "uncertainty", "final_gate", "guidance_target", "critic_choice", "failure_visualization"}:
            script_args.extend(["--maps", args.maps, "--episodes", args.episodes, "--horizon", args.horizon])
        if label in {"data_semantics"}:
            script_args.extend(["--demo_root", args.demo_root])
        if label in {"mechanism_action_shift", "mechanism_proxy_risk_drop", "mechanism_uncertainty_gate"}:
            script_args.extend(["--demo_root", args.demo_root])
            script_args.extend(["--output_csv", args.output_dir / label / f"{label}.csv"])
        elif label == "aggregate_tables":
            script_args.extend([
                "--input_dirs",
                args.output_dir / "closed_loop_main",
                args.output_dir / "guidance_source",
                args.output_dir / "uncertainty",
                args.output_dir / "final_gate",
                args.output_dir / "guidance_target",
                args.output_dir / "critic_choice",
                "--output_dir", args.output_dir / label,
            ])
        else:
            script_args.extend(["--output_dir", args.output_dir / label])
        exports = {"CUDA_VISIBLE_DEVICES": "0"} if label.startswith("mechanism_") else None
        rows.append({
            "label": label,
            "script": str(script),
            "command": bash_conda_command(DACER_ENV, python_cmd(script, script_args), exports=exports),
        })
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_full_suite_generation_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
