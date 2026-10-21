#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv


ROOT = Path(__file__).resolve().parents[1]
DQL_ROOT = ROOT / "baselines" / "diffusion_ql_source" / "Diffusion-Policies-for-Offline-RL"
sys.path.insert(0, str(DQL_ROOT))

from agents.ql_diffusion import Diffusion_QL


@dataclass
class EpisodeStats:
    reward: float = 0.0
    cost: float = 0.0
    length: int = 0
    success: float = 0.0


def make_env(start_seed: int, num_scenarios: int, traffic_density: float, horizon: int) -> SafeMetaDriveEnv:
    return SafeMetaDriveEnv({
        "use_render": False,
        "manual_control": False,
        "num_scenarios": int(num_scenarios),
        "start_seed": int(start_seed),
        "traffic_density": float(traffic_density),
        "horizon": int(horizon),
        "out_of_route_done": True,
        "crash_vehicle_done": True,
        "crash_object_done": True,
        "crash_human_done": True,
        "cost_to_reward": False,
    })


def reset_env(env) -> np.ndarray:
    ret = env.reset()
    obs = ret[0] if isinstance(ret, tuple) else ret
    return np.asarray(obs, dtype=np.float32)


def step_env(env, action) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    ret = env.step(np.asarray(action, dtype=np.float32))
    if len(ret) == 5:
        obs, reward, terminated, truncated, info = ret
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = ret
        done = bool(done)
    return np.asarray(obs, dtype=np.float32), float(reward), done, dict(info or {})


def info_cost(info: dict[str, Any]) -> float:
    return float(info.get("cost", info.get("native_cost", 0.0)) or 0.0)


def info_success(info: dict[str, Any]) -> float:
    for key in ("arrive_dest", "success", "is_success"):
        if key in info:
            return float(bool(info[key]))
    return 0.0


def info_seed(info: dict[str, Any]) -> Any:
    for key in ("scenario_index", "seed", "current_seed"):
        if key in info:
            return info[key]
    return ""


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = np.asarray([r["reward"] for r in rows], dtype=np.float64)
    costs = np.asarray([r["cost"] for r in rows], dtype=np.float64)
    succ = np.asarray([r["success"] for r in rows], dtype=np.float64)
    route = np.asarray([r["route_completion"] for r in rows], dtype=np.float64)
    return {
        "episodes": len(rows),
        "return_mean": float(rewards.mean()) if len(rewards) else float("nan"),
        "return_std": float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0,
        "test_cost_mean": float(costs.mean()) if len(costs) else float("nan"),
        "test_cost_std": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "success_mean": float(succ.mean()) if len(succ) else float("nan"),
        "success_std": float(succ.std(ddof=1)) if len(succ) > 1 else 0.0,
        "route_completion_mean": float(route.mean()) if len(route) else float("nan"),
        "route_completion_std": float(route.std(ddof=1)) if len(route) > 1 else 0.0,
    }


def eval_agent(agent: Diffusion_QL, args) -> list[dict[str, Any]]:
    env = make_env(args.eval_start_seed, args.eval_num_scenarios, args.traffic_density, args.horizon)
    rows = []
    try:
        for ep in range(int(args.episodes)):
            obs = reset_env(env)
            stats = EpisodeStats()
            final_info = {}
            for _ in range(args.horizon):
                action = agent.sample_action(obs)
                action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
                next_obs, reward, done, info = step_env(env, action)
                stats.reward += reward
                stats.cost += info_cost(info)
                stats.length += 1
                final_info = info
                obs = next_obs
                if done:
                    break
            stats.success = info_success(final_info)
            row = {
                "episode": ep,
                "scenario_index": info_seed(final_info),
                "reward": stats.reward,
                "cost": stats.cost,
                "length": stats.length,
                "success": stats.success,
                "route_completion": float(final_info.get("route_completion", final_info.get("progress", np.nan))),
            }
            rows.append(row)
            if (ep + 1) % 10 == 0:
                print("[progress] " + json.dumps(summarize(rows), sort_keys=True), flush=True)
    finally:
        env.close()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--checkpoint_id", type=str, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--eval_start_seed", type=int, default=200)
    parser.add_argument("--eval_num_scenarios", type=int, default=10)
    parser.add_argument("--traffic_density", type=float, default=0.06)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--beta_schedule", type=str, default="vp")
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--grad_norm", type=float, default=5.0)
    parser.add_argument("--max_q_backup", action="store_true")
    parser.add_argument("--lr_decay", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "args.json").open("w") as f:
        json.dump({k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, f, indent=2, sort_keys=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[torch] device={device} cuda={torch.cuda.is_available()}", flush=True)

    env = make_env(args.eval_start_seed, args.eval_num_scenarios, args.traffic_density, args.horizon)
    obs = reset_env(env)
    obs_dim = int(np.asarray(obs).shape[-1])
    act_dim = int(env.action_space.shape[0])
    max_action = float(np.asarray(env.action_space.high).max())
    env.close()
    print(f"[env] obs_dim={obs_dim} act_dim={act_dim} max_action={max_action}", flush=True)
    print(f"[eval] map_start={args.eval_start_seed} maps={args.eval_num_scenarios} episodes={args.episodes}", flush=True)

    agent = Diffusion_QL(
        state_dim=obs_dim,
        action_dim=act_dim,
        max_action=max_action,
        device=device,
        discount=args.discount,
        tau=args.tau,
        max_q_backup=args.max_q_backup,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.T,
        eta=args.eta,
        lr=args.lr,
        lr_decay=args.lr_decay,
        lr_maxt=1,
        grad_norm=args.grad_norm,
    )
    agent.load_model(str(args.checkpoint_dir), args.checkpoint_id)
    rows = eval_agent(agent, args)
    summary = summarize(rows)
    summary["checkpoint_dir"] = str(args.checkpoint_dir)
    summary["checkpoint_id"] = args.checkpoint_id
    summary["eval_start_seed"] = args.eval_start_seed
    summary["eval_num_scenarios"] = args.eval_num_scenarios
    summary["traffic_density"] = args.traffic_density
    write_csv(args.out_dir / "eval_episodes.csv", rows, ["episode", "scenario_index", "reward", "cost", "length", "success", "route_completion"])
    with (args.out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print("[summary] " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
