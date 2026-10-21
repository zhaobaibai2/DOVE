from __future__ import annotations

import argparse
import csv
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common.metadrive_compat import (
    SCENE_PRESETS,
    reset_env,
    sample_action,
    scene_config,
    scene_names,
    step_env,
)
from experiments.run_metadrive_smoke import _hdsac_args, parse_methods


def _add_path(path: Path) -> None:
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


def _empty_row(method: str, scene: str, seed: int, status: str, note: str = "") -> Dict[str, Any]:
    return {
        "method": method,
        "scene": scene,
        "seed": seed,
        "status": status,
        "policy_source": "",
        "steps": 0,
        "total_reward": 0.0,
        "total_cost": 0.0,
        "success": False,
        "crash": False,
        "out_of_road": False,
        "final_reason": "",
        "note": note,
    }


def _accumulate(row: Dict[str, Any], reward: float, info: Dict[str, Any], terminated: bool, truncated: bool) -> None:
    row["steps"] += 1
    row["total_reward"] += float(reward)
    row["total_cost"] += float(info.get("cost", info.get("native_cost", 0.0)) or 0.0)
    row["success"] = bool(row["success"] or info.get("success", False) or info.get("arrive_dest", False))
    row["crash"] = bool(row["crash"] or info.get("crash", False) or info.get("crash_vehicle", False))
    row["out_of_road"] = bool(row["out_of_road"] or info.get("out_of_road", False))
    if terminated or truncated:
        if row["success"]:
            row["final_reason"] = "success"
        elif row["crash"]:
            row["final_reason"] = "crash"
        elif row["out_of_road"]:
            row["final_reason"] = "out_of_road"
        else:
            row["final_reason"] = "terminated" if terminated else "truncated"


def run_dacer(scene: str, seed: int, max_steps: int) -> Dict[str, Any]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    from relax.network.dacer import create_dacer_net
    from relax_env.human_in_the_loop_env import HumanInTheLoopEnv

    env = HumanInTheLoopEnv(scene_config(scene, seed))
    row = _empty_row("dacer", scene, seed, "ok")
    row["policy_source"] = "fresh_dacer_init"
    try:
        obs, _ = reset_env(env)
        net, params = create_dacer_net(jax.random.key(seed), obs.shape[0], env.action_space.shape[0], [32, 32], [32, 32], num_timesteps=2)
        for _ in range(max_steps):
            action = np.asarray(net.get_deterministic_action((params.policy, params.log_alpha), jnp.asarray(obs[None, :]))[0], dtype=np.float32)
            obs, reward, terminated, truncated, info, _ret_len = step_env(env, np.clip(action, -1.0, 1.0))
            _accumulate(row, reward, info, terminated, truncated)
            if terminated or truncated:
                break
        return row
    finally:
        env.close()


def run_pvp(scene: str, seed: int, max_steps: int) -> Dict[str, Any]:
    _add_path(ROOT / "baselines" / "pvp")
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    from pvp.pvp_td3 import PVPTD3
    from pvp.sb3.haco.haco_buffer import HACOReplayBuffer
    from pvp.sb3.td3.policies import TD3Policy

    env = HumanInTheLoopEnv(scene_config(scene, seed))
    row = _empty_row("pvp", scene, seed, "ok")
    row["policy_source"] = "fresh_pvptd3_init"
    try:
        obs, _ = reset_env(env)
        model = PVPTD3(
            policy=TD3Policy,
            env=env,
            learning_rate=1e-4,
            buffer_size=100,
            learning_starts=10,
            batch_size=8,
            train_freq=(1, "step"),
            gradient_steps=0,
            replay_buffer_class=HACOReplayBuffer,
            replay_buffer_kwargs={"discard_reward": True},
            policy_kwargs={"net_arch": [32, 32]},
            verbose=0,
            seed=seed,
            device="cpu",
        )
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info, _ret_len = step_env(env, np.clip(action, -1.0, 1.0))
            _accumulate(row, reward, info, terminated, truncated)
            if terminated or truncated:
                break
        return row
    finally:
        env.close()


def run_hdsac(scene: str, seed: int, max_steps: int, output_dir: Path) -> Dict[str, Any]:
    import torch

    hdsac_root = ROOT / "baselines" / "hdsac"
    _add_path(hdsac_root)
    _add_path(hdsac_root / "utils")
    from utils.initialization import create_alg, create_env
    from utils.init_args import init_args

    cfg = scene_config(scene, seed)
    args = _hdsac_args(seed, output_dir)
    args.update({k: v for k, v in cfg.items() if k in {"num_scenarios", "traffic_density", "start_seed", "horizon", "map"}})
    env = create_env(**args)
    row = _empty_row("hdsac", scene, seed, "ok")
    row["policy_source"] = "fresh_dsac_v2_pvp_init"
    try:
        obs, _ = reset_env(env)
        args = init_args(env, **args)
        alg = create_alg(**args)
        for _ in range(max_steps):
            obs_t = torch.from_numpy(obs[None, :].astype("float32"))
            with torch.no_grad():
                logits = alg.networks.policy(obs_t)
                action = alg.networks.create_action_distributions(logits).mode().detach()[0].cpu().numpy()
            obs, reward, terminated, truncated, info, _ret_len = step_env(env, np.clip(action, -1.0, 1.0))
            _accumulate(row, reward, info, terminated, truncated)
            if terminated or truncated:
                break
        return row
    finally:
        env.close()


def run_haco(scene: str, seed: int, max_steps: int) -> Dict[str, Any]:
    _add_path(ROOT / "baselines" / "haco")
    from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv

    cfg = scene_config(scene, seed)
    cfg["environment_num"] = cfg.get("num_scenarios", 1)
    env = HumanInTheLoopEnv(cfg)
    row = _empty_row("haco", scene, seed, "env_ok", "uses random action until ray/rllib trainer dependency is installed")
    row["policy_source"] = "random_action_env_only"
    try:
        obs, _ = reset_env(env)
        for _ in range(max_steps):
            obs, reward, terminated, truncated, info, _ret_len = step_env(env, sample_action(env))
            _accumulate(row, reward, info, terminated, truncated)
            if terminated or truncated:
                break
        return row
    finally:
        env.close()


def run_one(method: str, scene: str, seed: int, max_steps: int, output_dir: Path) -> Dict[str, Any]:
    runners = {
        "dacer": lambda: run_dacer(scene, seed, max_steps),
        "pvp": lambda: run_pvp(scene, seed, max_steps),
        "hdsac": lambda: run_hdsac(scene, seed, max_steps, output_dir),
        "haco": lambda: run_haco(scene, seed, max_steps),
    }
    try:
        return runners[method]()
    except Exception as exc:
        traceback.print_exc()
        return _empty_row(method, scene, seed, "failed", f"{type(exc).__name__}: {exc}")


def run_matrix(methods: Iterable[str], scenes: Iterable[str], seeds: Iterable[int], max_steps: int, output_dir: Path) -> List[Dict[str, Any]]:
    rows = []
    for scene in scenes:
        for seed in seeds:
            for method in methods:
                rows.append(run_one(method, scene, int(seed), max_steps, output_dir))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run headless MetaDrive simulation scenes for trimmed baselines.")
    parser.add_argument("--methods", default="all", help="Comma list from dacer,pvp,hdsac,haco, or all.")
    parser.add_argument("--scenes", default="all", help=f"Comma list from {','.join(SCENE_PRESETS)}, or all.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[104])
    parser.add_argument("--max_steps", type=int, default=20)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "scene_matrix")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_matrix(parse_methods(args.methods), scene_names(args.scenes), args.seeds, max(1, args.max_steps), args.output_dir)
    out_csv = args.output_dir / "scene_simulation_matrix.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

