from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.common.metadrive_compat import metadrive_headless_config, reset_env, sample_action, step_env


ALL_METHODS = ("dacer", "pvp", "hdsac", "haco")


def _add_path(path: Path) -> None:
    path_s = str(path)
    if path_s not in sys.path:
        sys.path.insert(0, path_s)


def _row(method: str, status: str, **extra: Any) -> Dict[str, Any]:
    row = {
        "method": method,
        "status": status,
        "obs_dim": "",
        "act_dim": "",
        "step_return_len": "",
        "reward": "",
        "note": "",
    }
    row.update(extra)
    return row


def smoke_dacer(seed: int, steps: int) -> Dict[str, Any]:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    from relax.network.dacer import create_dacer_net
    from relax_env.human_in_the_loop_env import HumanInTheLoopEnv

    env = HumanInTheLoopEnv(metadrive_headless_config(seed))
    try:
        obs, _ = reset_env(env)
        net, params = create_dacer_net(
            jax.random.key(seed),
            obs.shape[0],
            env.action_space.shape[0],
            [32, 32],
            [32, 32],
            num_timesteps=2,
        )
        action = np.asarray(
            net.get_deterministic_action((params.policy, params.log_alpha), jnp.asarray(obs[None, :]))[0],
            dtype=np.float32,
        )
        reward = 0.0
        ret_len = ""
        for _ in range(steps):
            obs, reward, terminated, truncated, _info, ret_len = step_env(env, np.clip(action, -1.0, 1.0))
            if terminated or truncated:
                break
        return _row("dacer", "ok", obs_dim=obs.shape[0], act_dim=env.action_space.shape[0], step_return_len=ret_len, reward=reward)
    finally:
        env.close()


def smoke_pvp(seed: int, steps: int) -> Dict[str, Any]:
    _add_path(ROOT / "baselines" / "pvp")
    from pvp.experiments.metadrive.human_in_the_loop_env import HumanInTheLoopEnv
    from pvp.pvp_td3 import PVPTD3
    from pvp.sb3.haco.haco_buffer import HACOReplayBuffer
    from pvp.sb3.td3.policies import TD3Policy

    env = HumanInTheLoopEnv(metadrive_headless_config(seed))
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
        action, _ = model.predict(obs, deterministic=True)
        reward = 0.0
        ret_len = ""
        for _ in range(steps):
            obs, reward, terminated, truncated, _info, ret_len = step_env(env, np.clip(action, -1.0, 1.0))
            if terminated or truncated:
                break
        return _row("pvp", "ok", obs_dim=obs.shape[0], act_dim=env.action_space.shape[0], step_return_len=ret_len, reward=reward)
    finally:
        env.close()


def _hdsac_args(seed: int, output_dir: Path) -> Dict[str, Any]:
    return {
        "env_id": "gym_metadrivepvp",
        "algorithm": "DSAC_V2_PVP",
        "enable_cuda": False,
        "seed": seed,
        "reward_scale": 1,
        "action_type": "continu",
        "is_render": False,
        "manual_control": False,
        "record_video": False,
        "num_scenarios": 1,
        "start_seed": seed,
        "traffic_density": 0.0,
        "value_func_name": "ActionValueDistri",
        "value_func_type": "MLP",
        "value_hidden_sizes": [32, 32],
        "value_hidden_activation": "gelu",
        "value_output_activation": "linear",
        "value_min_log_std": -8,
        "value_max_log_std": 8,
        "policy_func_name": "StochaPolicy",
        "policy_func_type": "MLP",
        "policy_act_distribution": "TanhGaussDistribution",
        "policy_hidden_sizes": [32, 32],
        "policy_hidden_activation": "gelu",
        "policy_output_activation": "tanh",
        "policy_min_log_std": -20,
        "policy_max_log_std": 3,
        "value_learning_rate": 5e-4,
        "policy_learning_rate": 5e-4,
        "alpha_learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "auto_alpha": True,
        "alpha": 0.2,
        "delay_update": 2,
        "q_bound": 5,
        "cql_coefficient": 1.0,
        "sample_batch_size": 1,
        "trainer": "pvp_trainer",
        "save_folder": str(output_dir / "hdsac_smoke"),
    }


def smoke_hdsac(seed: int, steps: int, output_dir: Path) -> Dict[str, Any]:
    hdsac_root = ROOT / "baselines" / "hdsac"
    _add_path(hdsac_root)
    _add_path(hdsac_root / "utils")
    from utils.initialization import create_alg, create_env
    from utils.init_args import init_args

    args = _hdsac_args(seed, output_dir)
    env = create_env(**args)
    try:
        obs, _ = reset_env(env)
        args = init_args(env, **args)
        alg = create_alg(**args)
        reward = 0.0
        ret_len = ""
        for _ in range(steps):
            obs, reward, terminated, truncated, _info, ret_len = step_env(env, sample_action(env))
            if terminated or truncated:
                break
        return _row(
            "hdsac",
            "ok",
            obs_dim=obs.shape[0],
            act_dim=env.action_space.shape[0],
            step_return_len=ret_len,
            reward=reward,
            note=alg.__class__.__name__,
        )
    finally:
        env.close()


def smoke_haco(seed: int, steps: int) -> Dict[str, Any]:
    _add_path(ROOT / "baselines" / "haco")
    from haco.utils.human_in_the_loop_env import HumanInTheLoopEnv

    config = metadrive_headless_config(seed)
    config["environment_num"] = 1
    env = HumanInTheLoopEnv(config)
    try:
        obs, _ = reset_env(env)
        reward = 0.0
        ret_len = ""
        for _ in range(steps):
            obs, reward, terminated, truncated, _info, ret_len = step_env(env, sample_action(env))
            if terminated or truncated:
                break
        note = "ray missing; HACO trainer source kept but not importable" if importlib.util.find_spec("ray") is None else "ray available"
        return _row("haco", "env_ok", obs_dim=obs.shape[0], act_dim=env.action_space.shape[0], step_return_len=ret_len, reward=reward, note=note)
    finally:
        env.close()


def run_methods(methods: Iterable[str], seed: int, steps: int, output_dir: Path) -> List[Dict[str, Any]]:
    runners = {
        "dacer": lambda: smoke_dacer(seed, steps),
        "pvp": lambda: smoke_pvp(seed, steps),
        "hdsac": lambda: smoke_hdsac(seed, steps, output_dir),
        "haco": lambda: smoke_haco(seed, steps),
    }
    rows = []
    for method in methods:
        try:
            rows.append(runners[method]())
        except Exception as exc:
            rows.append(_row(method, "failed", note=f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
    return rows


def parse_methods(raw: str) -> List[str]:
    if raw == "all":
        return list(ALL_METHODS)
    methods = [item.strip().lower() for item in raw.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(ALL_METHODS))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}; expected {ALL_METHODS}")
    return methods


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test trimmed baselines on headless MetaDrive.")
    parser.add_argument("--methods", default="all", help="Comma list from dacer,pvp,hdsac,haco, or all.")
    parser.add_argument("--seed", type=int, default=104)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--output_dir", type=Path, default=ROOT / "runs" / "smoke")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = run_methods(parse_methods(args.methods), args.seed, max(1, args.steps), args.output_dir)
    out_csv = args.output_dir / "metadrive_smoke.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()

