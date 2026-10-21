"""Static consistency check for the final DOVE paper experiment protocol.

This script reads source/configuration files only.  It never imports the
training/evaluation entrypoints and never launches an environment or model.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.dove_paper_protocol import (
    PAPER_METHOD_ALIASES,
    PAPER_METHOD_NAMES,
    PROTOCOL,
    canonical_paper_method_name,
)
from experiments.run_training_matrix import METHODS, command_for


HELDOUT_SHELL = "200 201 202 203 204 205 206 207 208 209"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_python_syntax(relative: str) -> None:
    ast.parse(read(relative), filename=relative)


def main() -> None:
    PROTOCOL.validate()
    require(len(PROTOCOL.training_maps) == 20, "training map count is not 20")
    require(len(PROTOCOL.heldout_maps) == 10, "held-out map count is not 10")
    require(len(PROTOCOL.training_seeds) == 5, "training seed count is not 5")
    require(PROTOCOL.heldout_episodes_per_map == 2, "held-out episodes/map is not 2")
    require(
        tuple(PAPER_METHOD_NAMES.values())
        == ("DOVE", "Positive-only", "ER_without_CPCal", "H-DSAC", "PVP"),
        "paper-facing method names differ from the final TeX/Harness convention",
    )
    require(
        canonical_paper_method_name("Positive-only Diffusion") == "Positive-only",
        "model-manifest Positive-only alias is not normalized",
    )
    require(
        canonical_paper_method_name("ER without CPCal") == "ER_without_CPCal",
        "model-manifest ER-without-CPCal alias is not normalized",
    )
    require(len(PAPER_METHOD_ALIASES) >= len(PAPER_METHOD_NAMES), "method alias map is incomplete")

    training_rows = [
        command_for(method, seed, Path(f"/path/to/stage1a/seed_{seed}"))
        for seed in PROTOCOL.training_seeds
        for method in METHODS
    ]
    require(len(training_rows) == 25, "training matrix must contain 5 methods x 5 seeds")
    require(
        {row["training_seed"] for row in training_rows} == set(PROTOCOL.training_seeds),
        "training matrix does not contain exactly seeds 0--4",
    )
    require(
        all(row["training_maps"] == " ".join(map(str, PROTOCOL.training_maps)) for row in training_rows),
        "training matrix map split differs from the protocol",
    )

    python_protocol_files = [
        "experiments/run_training_matrix.py",
        "experiments/run_policy_ablation_training_matrix.py",
        "experiments/run_pdf_data_semantics_ablation.py",
        "scripts/train_pvp_dacer_metadrive_off.py",
    ]
    for relative in python_protocol_files:
        source = read(relative)
        check_python_syntax(relative)
        require("PROTOCOL" in source or "TRAINING_MAPS" in source, f"{relative}: no protocol import")

    eval_protocol_files = [
        "experiments/run_pdf_closed_loop_main.py",
        "experiments/run_pdf_full_suite.py",
        "experiments/run_pdf_critic_choice_ablation.py",
        "experiments/run_pdf_final_gate_ablation.py",
        "experiments/run_pdf_guidance_source_ablation.py",
        "experiments/run_pdf_guidance_target_ablation.py",
        "experiments/run_pdf_uncertainty_ablation.py",
        "experiments/run_external_diffusion_fair_comparison.py",
        "scripts/eval_pvp_policies_fixed_v3.py",
    ]
    for relative in eval_protocol_files:
        source = read(relative)
        check_python_syntax(relative)
        require("heldout_maps" in source or "HELDOUT_MAPS" in source, f"{relative}: no held-out split")
        require(
            "heldout_episodes_per_map" in source,
            f"{relative}: no paper episodes-per-map default",
        )

    env_source = read("relax_env/human_in_the_loop_env.py")
    require('"num_scenarios": 20' in env_source, "environment default is not 20 maps")
    require('"start_seed": 104' in env_source, "environment map start is not 104")

    diffusion_ql_source = read("scripts/eval_diffusion_ql_checkpoint.py")
    require("default=200" in diffusion_ql_source, "Diffusion-QL held-out map start is not 200")
    require("default=10" in diffusion_ql_source, "Diffusion-QL held-out map count is not 10")
    require("default=20" in diffusion_ql_source, "Diffusion-QL total episode count is not 20")

    training_shells = [
        "runs/01_stage1a_collect_seed104_5pass_only.sh",
        "runs/10_train_dbp_only_seed104.sh",
        "runs/11_train_pvp_dbp_critic_seed104_50k.sh",
        "runs/12_train_doveer_er_seed104_50k.sh",
        "runs/13_train_doveer_er_no_pv_constraint_seed104_50k.sh",
    ]
    for relative in training_shells:
        source = read(relative)
        require(
            "--num_scenarios 20" in source or 'NUM_SCENARIOS="${NUM_SCENARIOS:-20}"' in source,
            f"{relative}: training map count is not 20",
        )
        require("TRAINING_SEED" in source, f"{relative}: independent training seed is not explicit")

    eval_shells = [
        "runs/20_eval_vanilla_policy_seed104.sh",
        "runs/21_eval_upvgds_no_gate_policy_seed104.sh",
        "runs/22_eval_upvgds_final_gate_full_policy_seed104.sh",
        "runs/23_eval_final_gate_no_disagreement_policy_seed104.sh",
        "runs/24_eval_final_gate_proxy_only_policy_seed104.sh",
        "runs/25_eval_final_gate_prior_only_policy_seed104.sh",
        "runs/26_eval_topk_rerank_policy_seed104.sh",
        "runs/30_eval_all_main_for_policy_seed104.sh",
        "runs/31_eval_11_all_policies_vanilla_seed104.sh",
        "runs/40_eval_all_policies_all_modes_seed104.sh",
        "runs/41_eval_training_runs_all_policies_seed104.sh",
        "runs/42_run_all_evals_gpu_jax_seed104.sh",
        "runs/44_run_all_evals_and_diagnostics_gpu_jax_seed104.sh",
        "runs/45_eval_every_eval_script_all_policies_seed104.sh",
        "runs/46_eval_training_runs_every_eval_script_all_policies_seed104.sh",
        "runs/47_run_every_eval_script_and_diagnostics_gpu_jax_seed104.sh",
        "runs/50_run_all_useful_evals_gpu_seed104.sh",
        "runs/50_resume_useful_evals_gpu_seed104.sh",
    ]
    maps_pattern = re.compile(r'MAPS="\$\{MAPS:-' + re.escape(HELDOUT_SHELL) + r'\}"')
    for relative in eval_shells:
        source = read(relative)
        require(maps_pattern.search(source) is not None, f"{relative}: held-out map default differs")
        require(
            'NUM_EPISODES="${NUM_EPISODES:-2}"' in source,
            f"{relative}: episodes/map default is not 2",
        )

    print("PAPER_PROTOCOL_CONSISTENCY=PASS")
    print(f"training_maps={list(PROTOCOL.training_maps)}")
    print(f"heldout_maps={list(PROTOCOL.heldout_maps)}")
    print(f"training_seeds={list(PROTOCOL.training_seeds)}")
    print(f"paper_method_names={list(PAPER_METHOD_NAMES.values())}")
    print(f"heldout_episodes_per_map={PROTOCOL.heldout_episodes_per_map}")
    print(f"generated_training_tasks={len(training_rows)}")
    print("experiments_started=0")


if __name__ == "__main__":
    main()
