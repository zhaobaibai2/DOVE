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
    DEFAULT_BC_POLICY,
    DEFAULT_LOG_DIR,
    DEFAULT_POLICY,
    bash_conda_command,
    parse_int_csv,
    print_written,
    python_cmd,
    write_command_files,
)
from map_splits import TEST_MAP_SEEDS


def eval_row(
    variant: str,
    log_dir: Path,
    policy: str,
    maps: List[int],
    episodes: int,
    horizon: int,
    output_dir: Path,
    extra_args: List[Any],
    note: str,
) -> Dict[str, Any]:
    output_csv = output_dir / f"{variant}.csv"
    args: List[Any] = [
        "--log_dir", log_dir,
        "--policy", policy,
        "--maps", *maps,
        "--num_episodes", episodes,
        "--horizon", horizon,
        "--output_csv", output_csv,
        *extra_args,
    ]
    command = bash_conda_command(
        DACER_ENV,
        python_cmd(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py", args),
        exports={"CUDA_VISIBLE_DEVICES": "0"},
    )
    return {
        "variant": variant,
        "policy": policy,
        "maps": " ".join(str(m) for m in maps),
        "episodes": episodes,
        "horizon": horizon,
        "output_csv": str(output_csv),
        "status": "ready",
        "note": note,
        "command": command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF Sec.8 main closed-loop driving evaluation matrix.")
    parser.add_argument("--log_dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--bc_policy", default=DEFAULT_BC_POLICY)
    parser.add_argument("--maps", default=",".join(map(str, TEST_MAP_SEEDS)))
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--tau_prior", default="0.02", help="Final Gate hard prior budget E(ag)-E(a0)<=tau_prior")
    parser.add_argument("--tau_u", default="1.0", help="Final Gate hard uncertainty budget U_psi(s,ag)<=tau_U; set <0 to disable")
    parser.add_argument("--max_action_shift", default="0.2", help="UPV-GDS/final-gate L2 action-shift budget d_max")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "pdf_main_closed_loop")
    args = parser.parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()

    maps = parse_int_csv(args.maps)
    rows = [
        eval_row(
            "bc_only_diffusion_policy",
            args.log_dir,
            args.bc_policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            [],
            "BC-only diffusion policy: evaluates Stage1b offline BC checkpoint without inference guidance.",
        ),
        eval_row(
            "pvp_dacer_without_guidance",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            [],
            "PVP-DACER policy, inference uses original deterministic diffusion actor without test-time guidance.",
        ),
        eval_row(
            "pvp_dacer_random_guidance",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_value_guidance", "--guidance_mode", "random_grad", "--guidance_q_agg", "conservative", "--critic_objective", "lower"],
            "Negative control: same magnitude random gradient instead of proxy-value gradient.",
        ),
        eval_row(
            "pvp_dacer_opposite_signed_proxy_guidance",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_value_guidance", "--guidance_mode", "reverse_grad", "--guidance_q_agg", "conservative", "--critic_objective", "lower"],
            "Opposite-signed proxy-value gradient diagnostic.",
        ),
        eval_row(
            "pvp_dacer_corrected_upvgds_no_unc",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_beta_unc", "0.0",
             "--guidance_q_agg", "conservative", "--guidance_injection", "clean_x0", "--guidance_schedule", "noise_level",
             "--critic_objective", "lower"],
            "Corrected UPV-GDS under lower-is-better proxy-risk semantics; no uncertainty attenuation.",
        ),
        eval_row(
            "pvp_dacer_corrected_upvgds_unc",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_beta_unc", "1.0",
             "--guidance_q_agg", "conservative", "--guidance_injection", "clean_x0", "--guidance_schedule", "noise_level",
             "--critic_objective", "lower"],
            "Uncertainty attenuation ablation for corrected proxy-risk guidance.",
        ),
        eval_row(
            "pvp_dacer_upvgds_proxy_gate_robust",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            [
                "--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_beta_unc", "0.0",
                "--guidance_q_agg", "conservative", "--guidance_injection", "clean_x0", "--guidance_schedule", "noise_level",
                "--critic_objective", "lower",
                "--use_final_gate", "--final_gate_mode", "proxy_only",
                "--final_gate_proxy_compare", "robust",
                "--final_gate_min_proxy_gain", "0.0",
                "--final_gate_margin", "0.0",
                "--final_gate_max_action_shift", args.max_action_shift,
                "--final_gate_tau_u", args.tau_u,
                "--final_gate_project_guidance", "1",
            ],
            "UPV-GDS with TeX-conservative proxy-improvement final gate aligned with the TeX feasibility condition.",
        ),
        eval_row(
            "pvp_dacer_upvgds_proxy_prior_gate_robust",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            [
                "--use_value_guidance", "--guidance_mode", "proxy_value", "--guidance_beta_unc", "1.0",
                "--guidance_q_agg", "conservative", "--guidance_injection", "clean_x0", "--guidance_schedule", "noise_level",
                "--critic_objective", "lower",
                "--use_final_gate", "--final_gate_mode", "proxy_prior",
                "--final_gate_proxy_compare", "robust",
                "--final_gate_min_proxy_gain", "0.0",
                "--final_gate_beta_prior", "1.0",
                "--final_gate_margin", "0.0",
                "--final_gate_max_action_shift", args.max_action_shift,
                "--final_gate_tau_u", args.tau_u,
                "--final_gate_project_guidance", "1",
                "--final_gate_tau_prior", args.tau_prior,
                "--final_gate_energy_samples", "4",
            ],
            "Full DOVEER execution: surrogate trust-region UPV-GDS proposal with TeX-conservative proxy, relative diffusion-prior, and U_psi empirical final gate.",
        ),
        eval_row(
            "topk_q_reranking",
            args.log_dir,
            args.policy,
            maps,
            args.episodes,
            args.horizon,
            args.output_dir,
            ["--rerank_topk", args.topk, "--guidance_q_agg", "conservative", "--critic_objective", "lower"],
            "Post-hoc Top-K proxy-risk reranking baseline under lower-is-better semantics.",
        ),
    ]
    csv_path, sh_path = write_command_files(rows, args.output_dir, "pdf_closed_loop_main_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
