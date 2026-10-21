from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.pdf_experiment_utils import DACER_ENV, bash_conda_command


def frange_values(raw: Iterable[str]) -> List[str]:
    values: List[str] = []
    for item in raw:
        values.extend(part.strip() for part in str(item).split(",") if part.strip())
    return values


def build_command(log_dir: Path, policy: str, maps: List[int], episodes: int, mode: str, lambda0: str, beta_unc: str, p_decay: str, grad_clip: str, output_dir: Path) -> Dict[str, Any]:
    name = f"{Path(policy).stem}_{mode}_l{lambda0}_b{beta_unc}_p{p_decay}_g{grad_clip}".replace(".", "p")
    output_csv = output_dir / f"{name}.csv"
    cmd = [
        "python", str(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py"),
        "--log_dir", str(log_dir),
        "--policy", policy,
        "--maps", *[str(m) for m in maps],
        "--num_episodes", str(episodes),
        "--output_csv", str(output_csv),
        "--guidance_mode", mode,
        "--guidance_lambda0", lambda0,
        "--guidance_beta_unc", beta_unc,
        "--guidance_p_decay", p_decay,
        "--guidance_grad_clip", grad_clip,
    ]
    if mode != "none":
        cmd.extend([
            "--use_value_guidance",
            "--guidance_q_agg", "conservative",
            "--critic_objective", "lower",
            "--guidance_injection", "clean_x0",
            "--guidance_schedule", "noise_level",
        ])
    return {
        "policy": policy,
        "guidance_mode": mode,
        "lambda0": lambda0,
        "beta_unc": beta_unc,
        "p_decay": p_decay,
        "grad_clip": grad_clip,
        "output_csv": str(output_csv),
        "command": bash_conda_command(DACER_ENV, cmd, exports={"CUDA_VISIBLE_DEVICES": "0"}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate UPV-GDS guidance ablation evaluation commands.")
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--policies", nargs="+", default=["final_policy_20324.pkl"])
    parser.add_argument("--maps", type=int, nargs="+", default=[104])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--guidance_modes", nargs="+", default=["none", "proxy_value", "random_grad", "reverse_grad"])
    parser.add_argument("--lambda0_values", nargs="+", default=["0.25", "0.5", "1.0"])
    parser.add_argument("--beta_unc_values", nargs="+", default=["0.5", "1.0", "2.0"])
    parser.add_argument("--p_decay_values", nargs="+", default=["0.5", "1.0", "2.0"])
    parser.add_argument("--grad_clip_values", nargs="+", default=["0.5", "1.0", "2.0", "5.0"])
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "upvgds_ablation_matrix")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    guidance_modes = frange_values(args.guidance_modes)
    allowed_modes = {"none", "proxy_value", "random_grad", "reverse_grad"}
    unknown_modes = sorted(set(guidance_modes) - allowed_modes)
    if unknown_modes:
        raise ValueError(f"unknown guidance modes: {unknown_modes}")

    for policy in args.policies:
        for mode in guidance_modes:
            if mode == "none":
                rows.append(build_command(args.log_dir, policy, args.maps, args.episodes, mode, "0.0", "1.0", "1.0", "1.0", args.output_dir))
                continue
            for lambda0 in frange_values(args.lambda0_values):
                for beta_unc in frange_values(args.beta_unc_values):
                    for p_decay in frange_values(args.p_decay_values):
                        for grad_clip in frange_values(args.grad_clip_values):
                            rows.append(build_command(args.log_dir, policy, args.maps, args.episodes, mode, lambda0, beta_unc, p_decay, grad_clip, args.output_dir))

    out_csv = args.output_dir / "upvgds_ablation_commands.csv"
    out_sh = args.output_dir / "upvgds_ablation_commands.sh"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with out_sh.open("w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\nset -euo pipefail\ncd " + shlex.quote(str(ROOT)) + "\n\n")
        for row in rows:
            f.write(row["command"] + "\n")
    print(f"generated {len(rows)} commands")
    print(f"wrote {out_csv}")
    print(f"wrote {out_sh}")


if __name__ == "__main__":
    main()
