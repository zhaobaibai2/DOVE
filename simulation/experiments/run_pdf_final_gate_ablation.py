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
    claim: str,
    guidance_beta_unc: float,
) -> Dict[str, Any]:
    output_csv = output_dir / f"{variant}.csv"
    eval_args: List[Any] = [
        "--log_dir", log_dir,
        "--policy", policy,
        "--maps", *maps,
        "--num_episodes", episodes,
        "--horizon", horizon,
        "--output_csv", output_csv,
        "--use_value_guidance",
        "--guidance_mode", "proxy_value",
        "--guidance_beta_unc", guidance_beta_unc,
        "--guidance_q_agg", "conservative",
        "--guidance_injection", "clean_x0",
        "--guidance_schedule", "noise_level",
        "--critic_objective", "lower",
        *extra,
    ]
    return {
        "variant": variant,
        "claim": claim,
        "output_csv": str(output_csv),
        "command": bash_conda_command(
            DACER_ENV,
            python_cmd(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py", eval_args),
            exports={"CUDA_VISIBLE_DEVICES": "0"},
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF empirical Final Gate acceptance-set ablation matrix.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--maps", default=",".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--min_proxy_gain", type=float, default=0.0)
    parser.add_argument("--beta_prior", type=float, default=1.0)
    parser.add_argument("--rho_shift", type=float, default=0.0)
    parser.add_argument("--energy_samples", type=int, default=4)
    parser.add_argument("--guidance_beta_unc", type=float, default=1.0,
                        help="UPV-GDS uncertainty attenuation; default 1.0 matches TeX lambda_t")
    parser.add_argument("--tau_prior", type=float, default=0.02)
    parser.add_argument("--tau_u", type=float, default=1.0, help="Final Gate uncertainty budget U_psi(s,ag)<=tau_U; set <0 to disable")
    parser.add_argument("--max_action_shift", type=float, default=0.2)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_final_gate_ablation")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    common_gate_args: List[Any] = [
        "--final_gate_margin", args.margin,
        "--final_gate_min_proxy_gain", args.min_proxy_gain,
        "--final_gate_beta_prior", args.beta_prior,
        "--final_gate_rho_shift", args.rho_shift,
        "--final_gate_energy_samples", args.energy_samples,
        "--final_gate_tau_prior", args.tau_prior,
        "--final_gate_tau_u", args.tau_u,
        "--final_gate_max_action_shift", args.max_action_shift,
        "--final_gate_project_guidance", 1,
        "--final_gate_proxy_compare", "robust",
    ]
    maps = parse_int_csv(args.maps)
    rows = [
        build_eval(
            "no_final_gate",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            [],
            "UPV-GDS executes the trust-region projected proposal without the empirical accept/reject gate.",
            args.guidance_beta_unc,
        ),
        build_eval(
            "proxy_only_gate",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_final_gate", "--final_gate_mode", "proxy_only", *common_gate_args],
            "Uses TeX-conservative proxy-risk gain only; removes diffusion-prior value term.",
            args.guidance_beta_unc,
        ),
        build_eval(
            "prior_only_gate",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_final_gate", "--final_gate_mode", "prior_only", *common_gate_args],
            "Uses diffusion-prior relative value only; tests whether prior compatibility alone is sufficient.",
            args.guidance_beta_unc,
        ),
        build_eval(
            "proxy_prior_final_gate",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_final_gate", "--final_gate_mode", "proxy_prior", *common_gate_args],
            "Full surrogate acceptance gate with TeX-conservative proxy gain, diffusion-prior relative value, uncertainty budget, and action-shift projection.",
            args.guidance_beta_unc,
        ),
        build_eval(
            "proxy_prior_no_disagreement_gate",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_final_gate", "--final_gate_mode", "proxy_prior", *common_gate_args, "--final_gate_tau_u", -1],
            "Full proxy+prior surrogate gate but disables the hard U_psi disagreement check; isolates the disagreement-gate term.",
            args.guidance_beta_unc,
        ),
    ]
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_final_gate_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
