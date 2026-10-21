from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

_THIS_ROOT = Path(__file__).resolve().parents[1]
if str(_THIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_THIS_ROOT))

from experiments.pdf_experiment_utils import DACER_ENV, ROOT, bash_conda_command, parse_csv, write_command_files, print_written
from experiments.dove_paper_protocol import PROTOCOL


def _eval_cmd(
    label: str,
    log_dir: Path,
    policy: str,
    maps: Sequence[str],
    episodes: int,
    horizon: int,
    output_dir: Path,
    extra_args: Sequence[Any],
    note: str,
) -> Dict[str, Any]:
    out_csv = output_dir / f"{label}.csv"
    cmd = [
        "python",
        str(ROOT / "scripts" / "eval_pvp_policies_fixed_v3.py"),
        "--log_dir",
        str(log_dir),
        "--policy",
        policy,
        "--maps",
        *[str(m) for m in maps],
        "--num_episodes",
        str(episodes),
        "--horizon",
        str(horizon),
        "--output_csv",
        str(out_csv),
        *[str(x) for x in extra_args],
    ]
    return {
        "family": "metadrive_same_checkpoint",
        "label": label,
        "policy": policy,
        "maps": ",".join(str(m) for m in maps),
        "episodes": episodes,
        "horizon": horizon,
        "output_csv": str(out_csv),
        "command": bash_conda_command(DACER_ENV, cmd, exports={"CUDA_VISIBLE_DEVICES": "0"}),
        "note": note,
    }


def build_same_checkpoint_rows(
    log_dir: Path,
    policy: str,
    maps: Sequence[str],
    episodes: int,
    horizon: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    rows = [
        _eval_cmd(
            "00_vanilla_dacer",
            log_dir,
            policy,
            maps,
            episodes,
            horizon,
            output_dir,
            ["--guidance_mode", "none"],
            "Same checkpoint without deployment-time guidance.",
        ),
        _eval_cmd(
            "10_vgd_style_value_guided_denoising",
            log_dir,
            policy,
            maps,
            episodes,
            horizon,
            output_dir,
            [
                "--use_value_guidance",
                "--guidance_mode",
                "proxy_value",
                "--guidance_beta_unc",
                "0.0",
                "--guidance_q_agg",
                "mean",
                "--guidance_target",
                "x0",
                "--guidance_injection",
                "clean_x0",
            ],
            "VGD-style same-task proxy: value-gradient denoising, no uncertainty attenuation and no final gate.",
        ),
        _eval_cmd(
            "20_cep_style_latent_energy_proxy",
            log_dir,
            policy,
            maps,
            episodes,
            horizon,
            output_dir,
            [
                "--use_value_guidance",
                "--guidance_mode",
                "proxy_value",
                "--guidance_beta_unc",
                "0.0",
                "--guidance_q_agg",
                "mean",
                "--guidance_target",
                "xt",
                "--guidance_injection",
                "latent_mean",
            ],
            "CEP-style same-task proxy: noisy-latent energy guidance. This is a proxy, not the original contrastive Qt model.",
        ),
        _eval_cmd(
            "30_dove_upv_gds_no_gate",
            log_dir,
            policy,
            maps,
            episodes,
            horizon,
            output_dir,
            [
                "--use_value_guidance",
                "--guidance_mode",
                "proxy_value",
                "--guidance_beta_unc",
                "1.0",
                "--guidance_q_agg",
                "lcb",
                "--critic_objective",
                "lower",
                "--guidance_target",
                "x0",
                "--guidance_injection",
                "clean_x0",
            ],
            "DOVE without final gate: uncertainty-attenuated proxy-value guidance.",
        ),
        _eval_cmd(
            "40_dove_full_final_gate",
            log_dir,
            policy,
            maps,
            episodes,
            horizon,
            output_dir,
            [
                "--use_value_guidance",
                "--guidance_mode",
                "proxy_value",
                "--guidance_beta_unc",
                "1.0",
                "--guidance_q_agg",
                "lcb",
                "--critic_objective",
                "lower",
                "--guidance_target",
                "x0",
                "--guidance_injection",
                "clean_x0",
                "--use_final_gate",
                "--final_gate_mode",
                "proxy_prior",
                "--final_gate_proxy_compare",
                "tex_lcb",
                "--final_gate_tau_prior",
                "0.02",
                "--final_gate_max_action_shift",
                "0.2",
                "--final_gate_tau_u",
                "1.0",
                "--final_gate_project_guidance",
                "1",
                "--final_gate_energy_samples",
                "4",
            ],
            "Full DOVE: UPV-GDS proposal plus empirical Final Gate acceptance set.",
        ),
        _eval_cmd(
            "50_random_guidance_control",
            log_dir,
            policy,
            maps,
            episodes,
            horizon,
            output_dir,
            ["--use_value_guidance", "--guidance_mode", "random_grad"],
            "Negative control: same magnitude random guidance direction.",
        ),
    ]
    return rows


def build_native_rows(output_dir: Path, seed: int) -> List[Dict[str, Any]]:
    cep_cmd = [
        "python",
        "bandit_toy.py",
        "--env",
        "rings",
        "--seed",
        str(seed),
        "--expid",
        f"cep_gpu_native_s{seed}",
        "--device",
        "cuda:0",
        "--n_epochs",
        "10",
        "--toy_datanum",
        "100000",
        "--toy_batch_size",
        "2048",
        "--diffusion_steps",
        "15",
        "--alpha",
        "3",
        "--method",
        "CEP",
    ]
    vgd_cmd = [
        "python",
        "train_vgd.py",
        "--config-path=cfg/gym",
        "--config-name=vgd_hopper.yaml",
        f"seed={seed}",
        "use_wandb=False",
        f"log_dir={output_dir / 'vgd_native_gpu'}",
        "device=cuda:0",
    ]
    return [
        {
            "family": "native_external_protocol",
            "label": "cep_native_gpu_toy2d",
            "policy": "",
            "maps": "",
            "episodes": "",
            "horizon": "",
            "output_csv": "",
            "command": bash_conda_command(
                DACER_ENV,
                cep_cmd,
                cwd=ROOT / "baselines" / "cep",
                exports={"CUDA_VISIBLE_DEVICES": "0"},
            ),
            "note": "Native CEP toy protocol on GPU; not directly comparable to MetaDrive closed-loop metrics.",
        },
        {
            "family": "native_external_protocol",
            "label": "vgd_native_gpu_gym",
            "policy": "",
            "maps": "",
            "episodes": "",
            "horizon": "",
            "output_csv": "",
            "command": bash_conda_command(
                DACER_ENV,
                vgd_cmd,
                cwd=ROOT / "baselines" / "vgd",
                exports={"CUDA_VISIBLE_DEVICES": "0"},
            ),
            "note": "Native VGD Gym protocol on GPU; blocked until hydra/omegaconf and dppo/SB3 submodules are installed.",
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fair CEP/VGD/DOVE comparison commands.")
    parser.add_argument("--log_dir", type=Path, required=True)
    parser.add_argument("--policy", default="final_policy_20324.pkl")
    parser.add_argument("--maps", default=",".join(map(str, PROTOCOL.heldout_maps)))
    parser.add_argument("--episodes", type=int, default=PROTOCOL.heldout_episodes_per_map)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--include_native", action="store_true")
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "external_diffusion_fair_comparison")
    args = parser.parse_args()

    args.output_dir = args.output_dir.expanduser().resolve()
    maps = parse_csv(args.maps)
    rows = build_same_checkpoint_rows(args.log_dir.expanduser().resolve(), args.policy, maps, args.episodes, args.horizon, args.output_dir)
    if args.include_native:
        rows.extend(build_native_rows(args.output_dir, args.seed))
    csv_path, sh_path = write_command_files(rows, args.output_dir, "external_diffusion_fair_comparison_commands")
    print_written(rows, csv_path, sh_path)


if __name__ == "__main__":
    main()
