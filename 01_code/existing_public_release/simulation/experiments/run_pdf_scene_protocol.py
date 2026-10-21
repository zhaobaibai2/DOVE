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
    HACO_ENV,
    bash_conda_command,
    print_written,
    python_cmd,
    write_command_files,
)
from map_splits import TEST_MAP_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.8.4 scene protocol smoke/simulation command matrix.")
    parser.add_argument("--scenes", default="traffic_003,traffic_006,traffic_010,traffic_015,seen_maps,unseen_maps")
    parser.add_argument("--seeds", default=" ".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_scene_protocol")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    rows: List[Dict[str, Any]] = []
    common_args: List[Any] = [
        "--scenes", args.scenes,
        "--seeds", *args.seeds.split(),
        "--max_steps", args.max_steps,
    ]
    dacer_out = args.output_dir / "dacer_pvp_methods"
    rows.append({
        "label": "dacer_pvp_methods_scene_protocol",
        "env": DACER_ENV,
        "methods": "dacer,pvp,hdsac",
        "output_dir": str(dacer_out),
        "command": bash_conda_command(
            DACER_ENV,
            python_cmd(ROOT / "experiments" / "run_scene_simulation_matrix.py", [
                "--methods", "dacer,pvp,hdsac",
                *common_args,
                "--output_dir", dacer_out,
            ]),
            exports={"JAX_PLATFORMS": "cpu"},
        ),
    })
    haco_out = args.output_dir / "haco"
    rows.append({
        "label": "haco_scene_protocol",
        "env": HACO_ENV,
        "methods": "haco",
        "output_dir": str(haco_out),
        "command": bash_conda_command(
            HACO_ENV,
            python_cmd(ROOT / "experiments" / "run_scene_simulation_matrix.py", [
                "--methods", "haco",
                *common_args,
                "--output_dir", haco_out,
            ]),
        ),
    })
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_scene_protocol_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
