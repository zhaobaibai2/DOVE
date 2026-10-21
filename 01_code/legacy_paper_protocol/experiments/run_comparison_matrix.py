from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dove_paper_protocol import PROTOCOL

MPLCONFIGDIR = "runs/mplconfig"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print top-level commands for the 0508 MetaDrive comparison workspace."
    )
    parser.add_argument("--seed", type=int, default=PROTOCOL.training_maps[0],
                        help="Map seed used only by smoke/diagnostic commands.")
    parser.add_argument("--steps", type=int, default=1, help="Short smoke/simulation horizon.")
    parser.add_argument("--timesteps", type=int, default=PROTOCOL.total_interaction_budget,
                        help="Training command budget; paper protocol requires 50000.")
    parser.add_argument(
        "--scenes",
        default="empty,traffic,dense_traffic,multi_seed,toy_cot",
        help="Comma-separated scene names for generated matrix commands.",
    )
    args = parser.parse_args()

    scenes = args.scenes
    commands: list[tuple[str, str]] = [
        (
            "source/import integrity",
            f"JAX_PLATFORMS=cpu MPLCONFIGDIR={MPLCONFIGDIR} "
            f"python experiments/run_source_integrity_check.py --seed {args.seed} "
            f"--steps {args.steps} --output_dir runs/integrity_manual",
        ),
        (
            "method smoke",
            f"JAX_PLATFORMS=cpu MPLCONFIGDIR={MPLCONFIGDIR} "
            f"python experiments/run_metadrive_smoke.py --methods all --seed {args.seed} "
            f"--steps {args.steps} --output_dir runs/smoke_manual",
        ),
        (
            "named scene simulation",
            f"JAX_PLATFORMS=cpu MPLCONFIGDIR={MPLCONFIGDIR} "
            f"python experiments/run_scene_simulation_matrix.py --methods all --scenes {scenes} "
            f"--seeds {args.seed} --max_steps {args.steps} --output_dir runs/scene_matrix_manual",
        ),
        (
            "paper training command matrix",
            "python experiments/run_training_matrix.py --methods all "
            f"--seeds {' '.join(map(str, PROTOCOL.training_seeds))} "
            "--output_dir runs/training_matrix_manual",
        ),
        (
            "external diffusion baseline dependency check",
            "python experiments/run_external_diffusion_baseline_checks.py "
            "--run_cep_toy_smoke --device cuda:0 --output_dir runs/external_diffusion_manual",
        ),
        (
            "CEP/VGD/DOVE fair comparison command matrix",
            "python experiments/run_external_diffusion_fair_comparison.py "
            "--log_dir logs/pvp_dacer_20260422_203800_s0 --policy final_policy_20324.pkl "
            f"--maps {','.join(map(str, PROTOCOL.heldout_maps))} "
            f"--episodes {PROTOCOL.heldout_episodes_per_map} --horizon 1000 "
            "--include_native --output_dir runs/external_diffusion_fair_manual",
        ),
        (
            "UPV-GDS/PVP-DACER checkpoint evaluation command matrix",
            "python experiments/run_checkpoint_eval_matrix.py --maps 104 --episodes 1 "
            "--output_dir runs/checkpoint_eval_manual",
        ),
        (
            "UPV-GDS inference ablation command matrix",
            "python experiments/run_upvgds_ablation_matrix.py "
            "--log_dir logs/TODO_REPLACE_WITH_TRAINED_UPVGDS --policies final_policy.pkl "
            "--maps 104 --episodes 1 --guidance_modes none,proxy_value,random_grad,reverse_grad "
            "--lambda0_values 0.5 --beta_unc_values 1.0 --p_decay_values 1.0 "
            "--grad_clip_values 1.0 --output_dir runs/upvgds_ablation_manual",
        ),
        (
            "UPV-GDS/PVP-DACER training ablation command matrix",
            "python experiments/run_policy_ablation_training_matrix.py "
            f"--seeds {' '.join(map(str, PROTOCOL.training_seeds))} "
            f"--total_step {args.timesteps} --output_dir runs/policy_ablation_training_manual",
        ),
        (
            "HACO trainer note",
            "# HACO trainer still needs ray/rllib or a separate compatible conda environment; "
            "the current workspace smoke only verifies the HACO MetaDrive env.",
        ),
        (
            "CEP/VGD comparability note",
            "# CEP and VGD are useful related diffusion-guidance baselines, but their upstream code "
            "does not run the same MetaDrive HITL protocol without additional adaptation.",
        ),
    ]
    for label, command in commands:
        print(f"# {label}")
        print(command)
        print()


if __name__ == "__main__":
    main()
