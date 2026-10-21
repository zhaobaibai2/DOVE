#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import pickle
import random
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

_RUNTIME_READY = False


def ensure_runtime_imports() -> None:
    global _RUNTIME_READY
    global torch, PVPDACERTorch, TorchDACERConfig, DACERActionConfig, DACERTorchAgent, TorchPVPBuffer
    if _RUNTIME_READY:
        return

    import torch as _torch
    torch = _torch
    try:
        from .torch_algorithm import PVPDACERTorch as _PVPDACERTorch, TorchDACERConfig as _TorchDACERConfig
        from .torch_networks import DACERActionConfig as _DACERActionConfig, DACERTorchAgent as _DACERTorchAgent
        from .torch_replay_buffer import TorchPVPBuffer as _TorchPVPBuffer
    except ImportError:
        package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sys.path.insert(0, package_root)
        from car_dacer_torch.torch_algorithm import PVPDACERTorch as _PVPDACERTorch, TorchDACERConfig as _TorchDACERConfig
        from car_dacer_torch.torch_networks import DACERActionConfig as _DACERActionConfig, DACERTorchAgent as _DACERTorchAgent
        from car_dacer_torch.torch_replay_buffer import TorchPVPBuffer as _TorchPVPBuffer

    PVPDACERTorch = _PVPDACERTorch
    TorchDACERConfig = _TorchDACERConfig
    DACERActionConfig = _DACERActionConfig
    DACERTorchAgent = _DACERTorchAgent
    TorchPVPBuffer = _TorchPVPBuffer
    _RUNTIME_READY = True


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["config_path"] = os.path.abspath(path)
    return cfg


def _buffer_files_in_dir(directory: str) -> List[str]:
    patterns = [
        os.path.join(directory, "buffer_*.pkl"),
        os.path.join(directory, "segment_*.pkl"),
    ]
    paths: List[str] = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    return sorted(set(paths), key=os.path.getmtime)


def auto_find_latest_buffer_dir() -> Optional[str]:
    search_paths = [
        "./outputs/DOVEER_REAL01_R1_HIL_COLLECT*/buffers",
        "./outputs/*R1_HIL_COLLECT*/buffers",
        "./outputs/*R1_FULL_ONLINE_FROM_SCRATCH*/buffers",
        "./buffers",
    ]

    candidates = []
    for pattern in search_paths:
        for path in glob.glob(pattern):
            if not os.path.isdir(path):
                continue
            files = _buffer_files_in_dir(path)
            if files:
                candidates.append((max(os.path.getmtime(f) for f in files), path))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[-1][1]


def resolve_buffer_paths(config: dict, buffer_input: Optional[str]) -> List[str]:
    env_buffer = os.environ.get("DACER_BUFFER_PATH") or os.environ.get("DACER_BUFFER_DIR")
    if buffer_input is None and env_buffer:
        buffer_input = env_buffer

    if buffer_input and buffer_input != "auto":
        if os.path.isdir(buffer_input):
            return _buffer_files_in_dir(buffer_input)
        if os.path.isfile(buffer_input):
            return [buffer_input]
        raise FileNotFoundError(f"buffer input not found: {buffer_input}")
    if buffer_input == "auto":
        latest_dir = auto_find_latest_buffer_dir()
        if latest_dir is None:
            raise FileNotFoundError("no latest Route1 buffer dir found")
        print(f"[buffer] auto selected latest Route1 buffer dir: {latest_dir}")
        return resolve_buffer_paths(config, latest_dir)

    train_cfg = config.get("training", {}) or {}
    if train_cfg.get("demo_data_path"):
        return [str(train_cfg["demo_data_path"])]
    if train_cfg.get("demo_data_dir") and str(train_cfg.get("demo_data_dir")).lower() != "auto":
        return resolve_buffer_paths(config, str(train_cfg["demo_data_dir"]))
    if str(train_cfg.get("demo_data_dir", "")).lower() == "auto":
        latest_dir = auto_find_latest_buffer_dir()
        if latest_dir is None:
            raise FileNotFoundError("no latest Route1 buffer dir found")
        print(f"[buffer] auto selected latest Route1 buffer dir: {latest_dir}")
        return resolve_buffer_paths(config, latest_dir)

    raise FileNotFoundError(
        "no Route1 buffer input configured; set --buffer, DACER_BUFFER_DIR, DACER_BUFFER_PATH, "
        "training.demo_data_dir, training.demo_data_path, or demo_data_dir: auto"
    )


def make_buffer(config: dict) -> TorchPVPBuffer:
    ensure_runtime_imports()
    train_cfg = config.get("training", {}) or {}
    buffer_max_size = int(train_cfg.get("buffer_max_size", int(train_cfg.get("replay_batch_size", 64)) * 100))
    human_max_size = int(train_cfg.get("human_buffer_max_size", max(1, buffer_max_size // 2)))
    return TorchPVPBuffer(
        max_size=buffer_max_size,
        human_max_size=human_max_size,
        obs_dim=int(config["env"]["state_dim"]),
        act_dim=int(config["env"]["action_dim"]),
    )


def merge_buffer_file(dst: TorchPVPBuffer, path: str) -> Tuple[int, int]:
    ensure_runtime_imports()
    base = os.path.basename(path)
    if base.startswith("segment_"):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        human = list(payload.get("human", []))
        pvp = list(payload.get("pvp", []))
    else:
        tmp = TorchPVPBuffer(
            max_size=dst.max_size,
            human_max_size=dst.human_max_size,
            obs_dim=dst.obs_dim,
            act_dim=dst.act_dim,
        )
        tmp.load(path)
        human = list(tmp.human_buffer)
        pvp = list(tmp.pvp_buffer)

    for exp in human:
        dst.add_human(exp)
    for exp in pvp:
        dst.add_pvp(exp)
    return len(human), len(pvp)


def load_buffers(config: dict, paths: Iterable[str]) -> TorchPVPBuffer:
    buffer = make_buffer(config)
    paths = list(paths)
    if not paths:
        raise FileNotFoundError("no Route1 buffer files found")

    total_human = 0
    total_pvp = 0
    for path in paths:
        h, p = merge_buffer_file(buffer, path)
        total_human += h
        total_pvp += p
        print(f"[buffer] {os.path.basename(path)} human={h} pvp={p}")

    stats = buffer.get_statistics()
    print(
        "[buffer] merged files={} human_loaded={} pvp_loaded={} human_kept={} pvp_kept={}".format(
            len(paths), total_human, total_pvp, stats["human_size"], stats["pvp_size"]
        )
    )
    return buffer


def setup_algorithm(config: dict) -> Tuple[PVPDACERTorch, torch.device]:
    ensure_runtime_imports()
    hw = config.get("hardware", {}) or {}
    use_gpu = bool(hw.get("use_gpu", True))
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    net_cfg = config.get("network", {}) or {}
    alg_cfg = config.get("algorithm", {}) or {}

    action_cfg = DACERActionConfig(
        init_alpha=float(alg_cfg.get("init_alpha", 0.1)),
        action_noise_scale=float(alg_cfg.get("action_noise_scale", 0.05)),
        use_ddim=bool(alg_cfg.get("use_ddim", True)),
        ddim_steps=int(alg_cfg.get("ddim_steps", max(1, min(int(alg_cfg.get("num_timesteps", 20)), 5)))),
        ddim_eta=float(alg_cfg.get("ddim_eta", 0.0)),
        use_guidance=bool(alg_cfg.get("use_guidance", True)),
        guidance_scale=float(alg_cfg.get("guidance_scale", 0.04)),
        guidance_uncertainty_kappa=float(alg_cfg.get("guidance_uncertainty_kappa", 1.0)),
        guidance_min_gate=float(alg_cfg.get("guidance_min_gate", 0.05)),
        guidance_sigma_ref=float(alg_cfg.get("guidance_sigma_ref", 1.0)),
        guidance_grad_clip=float(alg_cfg.get("guidance_grad_clip", 1.0)),
        final_gate_enabled=bool(alg_cfg.get("final_gate_enabled", True)),
        final_gate_max_shift=float(alg_cfg.get("final_gate_max_shift", 0.45)),
        final_gate_min_q_improve=float(alg_cfg.get("final_gate_min_q_improve", 0.0)),
        final_gate_max_uncertainty=float(alg_cfg.get("final_gate_max_uncertainty", 0.50)),
        final_gate_use_uncertainty=bool(alg_cfg.get("final_gate_use_uncertainty", True)),
    )

    agent = DACERTorchAgent(
        obs_dim=int(config["env"]["state_dim"]),
        act_dim=int(config["env"]["action_dim"]),
        hidden_dims=tuple(net_cfg.get("hidden_dims", [256, 256, 256])),
        diffusion_hidden_dims=tuple(net_cfg.get("diffusion_hidden_dims", [256, 256, 256])),
        num_timesteps=int(alg_cfg.get("num_timesteps", 20)),
        target_entropy=float(alg_cfg.get("target_entropy", -2.0)),
        time_dim=int(net_cfg.get("time_dim", 16)),
        activation=str(net_cfg.get("activation", "relu")),
        use_layer_norm=bool(net_cfg.get("use_layer_norm", True)),
        action_cfg=action_cfg,
        device=device,
    ).to(device)

    cfg = TorchDACERConfig(
        gamma=float(alg_cfg.get("gamma", 0.99)),
        tau=float(alg_cfg.get("tau", 0.005)),
        lr=float(alg_cfg.get("lr", 1e-4)),
        alpha_lr=float(alg_cfg.get("alpha_lr", 3e-2)),
        delay_update=int(alg_cfg.get("delay_update", 1)),
        delay_alpha_update=int(alg_cfg.get("delay_alpha_update", 1000)),
        reward_scale=float(alg_cfg.get("reward_scale", 1.0)),
        lambda_pv=float(alg_cfg.get("lambda_pv", 1.0)),
        B=float(alg_cfg.get("B", 1.0)),
        lambda_bc=float(alg_cfg.get("lambda_bc", 5.0)),
        reward_free=bool(alg_cfg.get("reward_free", True)),
        phase3_use_bc_boost=bool(alg_cfg.get("phase3_use_bc_boost", True)),
        fix_alpha=bool(alg_cfg.get("fix_alpha", True)),
        target_entropy=float(alg_cfg.get("target_entropy", -2.0)),
        lambda_energy=float(alg_cfg.get("lambda_energy", 0.5)),
        energy_margin=float(alg_cfg.get("energy_margin", 0.5)),
        energy_min_action_gap=float(alg_cfg.get("energy_min_action_gap", 0.03)),
        rejected_action_radius=float(alg_cfg.get("rejected_action_radius", 0.20)),
        critic_objective=str(alg_cfg.get("critic_objective", "cost")),
    )
    return PVPDACERTorch(agent, cfg, device=device), device


def human_bc_batch(buffer: TorchPVPBuffer, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    ensure_runtime_imports()
    exps = buffer.sample_human(batch_size)
    if not exps:
        raise RuntimeError("human buffer is empty")
    obs = torch.as_tensor(np.stack([e.obs for e in exps], axis=0), device=device, dtype=torch.float32)
    act = torch.as_tensor(np.stack([e.actions_human for e in exps], axis=0), device=device, dtype=torch.float32)
    return obs, act


def make_output_dir(config: dict, override: Optional[str]) -> str:
    if override:
        out = override
    else:
        logging_cfg = config.get("logging", {}) or {}
        root = logging_cfg.get("result_root", "./outputs")
        exp = logging_cfg.get("experiment_name", "DOVEER_OFFLINE")
        out = os.path.join(root, f"{exp}_" + datetime.datetime.now().strftime("%y%m%d-%H%M%S"))
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    return out


def write_metrics(writer: csv.DictWriter, step: int, stage: str, metrics: Dict[str, float]) -> None:
    row = {"step": step, "stage": stage}
    row.update(metrics)
    writer.writerow(row)


def run(config: dict, buffer_input: Optional[str], mode: Optional[str], output_dir: Optional[str]) -> str:
    ensure_runtime_imports()
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_cfg = config.get("training", {}) or {}
    mode = str(mode or train_cfg.get("offline_mode", "full")).lower()
    if mode not in ("full", "positive_only", "positive-only", "bc_only"):
        raise ValueError(f"unsupported offline mode: {mode}")

    paths = resolve_buffer_paths(config, buffer_input)
    buffer = load_buffers(config, paths)
    if buffer.human_size < int(train_cfg.get("buffer_warm_size", 1)):
        raise RuntimeError(f"not enough human samples: {buffer.human_size}")

    algorithm, device = setup_algorithm(config)
    init_model = train_cfg.get("init_model_path")
    if init_model:
        algorithm.load(str(init_model))
        print(f"[model] loaded init checkpoint: {init_model}")

    out_dir = make_output_dir(config, output_dir)
    with open(os.path.join(out_dir, "config_offline.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    batch_size = int(train_cfg.get("replay_batch_size", 64))
    bc_updates = int(train_cfg.get("offline_bc_updates", train_cfg.get("phase2_updates", 10000)))
    pvp_updates = int(train_cfg.get("offline_pvp_updates", train_cfg.get("phase3_updates", 10000)))
    human_ratio = float(train_cfg.get("phase3_human_mix_ratio", 0.5))
    log_interval = int(train_cfg.get("log_save_interval", 500))
    save_interval = int(train_cfg.get("apprfunc_save_interval", 5000))
    run_pvp = mode == "full" and pvp_updates > 0

    metrics_path = os.path.join(out_dir, "logs", "offline_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "step", "stage", "bc_loss", "q1_loss", "q2_loss", "policy_loss",
            "energy_loss", "energy_violation", "pair_sat_e", "iar",
            "action_gap_l2", "ragr", "p_delta_plus", "p_delta_minus",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for step in range(1, bc_updates + 1):
            obs, act = human_bc_batch(buffer, batch_size, device)
            metrics = algorithm.train_offline_bc(obs, act)
            if step % log_interval == 0 or step == 1 or step == bc_updates:
                write_metrics(writer, step, "bc", metrics)
                f.flush()
                print(f"[bc] {step}/{bc_updates} loss={metrics.get('bc_loss', 0.0):.6f}")
            if step % save_interval == 0:
                ckpt = os.path.join(out_dir, "models", f"offline_bc_{step:08d}.pt")
                algorithm.save(ckpt)

        if run_pvp:
            for step in range(1, pvp_updates + 1):
                exps = buffer.sample_pvp_mixed(batch_size, human_ratio=human_ratio)
                batch = buffer.to_pvp_batch(exps, device=device)
                if batch is None:
                    raise RuntimeError("cannot build PVP batch")
                metrics = algorithm.train_pvp(batch)
                if step % log_interval == 0 or step == 1 or step == pvp_updates:
                    write_metrics(writer, step, "pvp", metrics)
                    f.flush()
                    print(
                        "[pvp] {}/{} policy={:.6f} energy={:.6f} pair_sat={:.4f}".format(
                            step,
                            pvp_updates,
                            metrics.get("policy_loss", 0.0),
                            metrics.get("energy_loss", 0.0),
                            metrics.get("pair_sat_e", 0.0),
                        )
                    )
                if step % save_interval == 0:
                    ckpt = os.path.join(out_dir, "models", f"offline_pvp_{step:08d}.pt")
                    algorithm.save(ckpt)

    final_path = os.path.join(out_dir, "models", f"offline_{mode.replace('-', '_')}_final.pt")
    algorithm.save(final_path)
    print(f"[done] final checkpoint: {final_path}")
    print(f"[done] metrics: {metrics_path}")
    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Route1 training for DOVE-ER real-car experiments.")
    parser.add_argument("--config", default=os.environ.get("DACER_CONFIG_PATH", "config.yaml"))
    parser.add_argument("--buffer", default=None, help="Route1 buffer file/dir, or auto. Defaults to config paths.")
    parser.add_argument("--mode", default=None, help="full or positive_only. Defaults to training.offline_mode.")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run(config, args.buffer, args.mode, args.output_dir)


if __name__ == "__main__":
    main()
