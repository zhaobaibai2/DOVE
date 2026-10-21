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

from experiments.pdf_experiment_utils import DACER_ENV, bash_conda_command


def build_upvgds_eval(log_dir: Path, policy: str, maps: List[int], episodes: int, guidance_mode: str, output_dir: Path) -> Dict[str, Any]:
    suffix = f"{Path(policy).stem}_{guidance_mode}.csv"
    output_csv = output_dir / suffix
    cmd = [
        "python", str(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py"),
        "--log_dir", str(log_dir),
        "--policy", policy,
        "--maps", *[str(m) for m in maps],
        "--num_episodes", str(episodes),
        "--output_csv", str(output_csv),
    ]
    if guidance_mode != "none":
        cmd.extend(["--use_value_guidance", "--guidance_mode", guidance_mode])
    return {
        "method": "upvgds",
        "policy": policy,
        "guidance_mode": guidance_mode,
        "output_csv": str(output_csv),
        "command": bash_conda_command(DACER_ENV, cmd, exports={"CUDA_VISIBLE_DEVICES": "0"}),
        "note": "requires trained PVP-DACER checkpoint and JAX GPU",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate evaluation commands for trained checkpoints.")
    parser.add_argument("--upvgds_log_dir", type=Path, default=None, help="PVP-DACER/UPV-GDS log dir containing args.json and policy pkl.")
    parser.add_argument("--upvgds_policies", nargs="+", default=["final_policy_20324.pkl"])
    parser.add_argument("--maps", type=int, nargs="+", default=[104])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--guidance_modes", nargs="+", default=["none", "proxy_value", "random_grad", "reverse_grad"])
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "checkpoint_eval_matrix")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    if args.upvgds_log_dir is not None:
        for policy in args.upvgds_policies:
            for mode in args.guidance_modes:
                rows.append(build_upvgds_eval(args.upvgds_log_dir, policy, args.maps, args.episodes, mode, args.output_dir))
    else:
        rows.append({
            "method": "upvgds",
            "policy": "",
            "guidance_mode": "",
            "output_csv": "",
            "command": "",
            "note": "not generated: pass --upvgds_log_dir with trained checkpoint directory",
        })

    out_csv = args.output_dir / "checkpoint_eval_commands.csv"
    out_sh = args.output_dir / "checkpoint_eval_commands.sh"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with out_sh.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n\n")
        for row in rows:
            if row["command"]:
                f.write(f"# {row['method']} {row['policy']} {row['guidance_mode']}\n{row['command']}\n\n")
    for row in rows:
        print(row)
    print(f"wrote {out_csv}")
    print(f"wrote {out_sh}")


if __name__ == "__main__":
    main()
