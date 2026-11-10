"""Evaluate the three recorded DOVE checkpoints and publish complete results."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import statistics
from threading import Lock

from dove_table01 import (DATA, MODELS, ROOT, activate_run, append_terminal_csv,
                          create_terminal_csv, digest, episode_shards,
                          model_manifest, read_csv, write_csv, write_json)


def calculate(terminal_csv=None):
    command = [sys.executable, str(ROOT / "02_data/raw_data/TABLES/TAB-01/calculate.py")]
    if terminal_csv is not None:
        command.extend(["--terminal-csv", str(terminal_csv)])
    subprocess.run(command,
                   cwd=ROOT, check=True)
    if terminal_csv is not None:
        latest = DATA.parent.parent / "test_results.csv"
        temporary = latest.with_suffix(".csv.tmp")
        shutil.copyfile(terminal_csv, temporary)
        temporary.replace(latest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "400")))
    parser.add_argument("--replay", action="store_true", help="Use the three archived CSVs; no simulation")
    parser.add_argument("--envs-per-model", type=int, default=int(os.environ.get("DOVE_ENVS_PER_MODEL", "2")),
                        help="Independent simulator processes per model (default: 2)")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("DOVE_EVAL_WORKERS", "6")),
                        help="Maximum concurrent simulator processes (default: 6)")
    args = parser.parse_args()
    manifest = model_manifest()
    spec = importlib.util.spec_from_file_location("dove_calculation_settings", DATA.parent.parent / "calculate.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    reward_factor = settings.REPORT_REWARD_FACTOR
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    terminal_csv = DATA.parent.parent / f"test_results_{run_id}.csv"
    if args.replay:
        create_terminal_csv(terminal_csv)
        activate_run(DATA / "historical_three", manifest)
        calculate(terminal_csv)
        return
    if args.num_episodes < 2:
        parser.error("--num-episodes must be at least 2")
    if not 1 <= args.envs_per_model <= args.num_episodes or args.workers < 1:
        parser.error("Invalid number of environment processes")
    # Each model gets a fresh simulator process, as in the original logs50 jobs.
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("JAX_PLATFORMS", "cuda")
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env["DOVE_EPISODE_PROGRESS"] = "1"
    env["DOVE_NO_TENSORBOARD"] = "1"
    env["DOVE_CORE_METRICS_ONLY"] = "1"
    env.pop("DOVE_TRAFFIC_SEED_BASE", None)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = "1"
    if env["JAX_PLATFORMS"] != "cpu":
        subprocess.run(["nvidia-smi"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run_dir = DATA / "evaluations" / run_id
    run_dir.mkdir(parents=True)
    create_terminal_csv(terminal_csv)
    evaluator = ROOT / "scripts/eval_pvp_policies_fixed_v3.py"
    run = {"source": "live_simulation", "protocol": manifest["protocol"],
           "num_episodes": args.num_episodes, "models": [],
           "workers": args.workers,
           "envs_per_model": args.envs_per_model,
           "terminal_results_csv": terminal_csv.name,
           "core_metrics_only": True,
           "reporting_reward_factor": reward_factor,
           "sampling": {"scheme": "replicated_original_config_v1",
                        "note": "Original map/reset/traffic configuration in every environment; deterministic replicas may repeat outcomes."},
           "python": sys.executable, "args_sha256": digest(MODELS / "args.json"),
           "eval_flags": manifest["eval_flags"],
           "packages": {name: importlib.metadata.version(name) for name in
                        ("jax", "jaxlib", "numpy", "metadrive-simulator", "gymnasium")},
           "code_sha256": {str(p.relative_to(ROOT)): digest(p) for directory in
                           (ROOT / "scripts", ROOT / "relax", ROOT / "relax_env")
                           for p in directory.rglob("*.py")},
           "environment": {key: env.get(key) for key in
                           ("CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "XLA_FLAGS")}}
    write_json(run_dir / "pending_run.json", run)
    progress = {index: {"completed": 0, "arrived": 0, "crash": 0, "out_of_route": 0, "timeout": 0, "other": 0}
                for index in (1, 2, 3)}
    progress_lock = Lock()
    jobs = []
    for model_index, item in enumerate(manifest["models"], start=1):
        output = run_dir / Path(item["filename"]).stem
        output.mkdir()
        for shard_index, (start, count) in enumerate(episode_shards(args.num_episodes, args.envs_per_model)):
            jobs.append((model_index, item, shard_index, start, count))

    def evaluate_one(job):
        model_index, item, shard_index, start, count = job
        name = item["filename"]
        output = run_dir / Path(name).stem / "shards" / str(shard_index)
        output.mkdir(parents=True)
        worker_env = env.copy()
        worker_env["DOVE_EPISODE_START"] = str(start)
        command = [sys.executable, "-u", str(evaluator), "--log_dir", str(MODELS),
                   "--policy", name, "--num_episodes", str(count),
                   "--output_csv", str(output / "summary.csv"),
                   "--episode_output_csv", str(output / "episodes.csv"),
                   "--tensorboard_dir", str(output / "tb"), *manifest["eval_flags"]]
        with subprocess.Popen(command, cwd=ROOT, env=worker_env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            for line in process.stdout:
                if line.startswith("DOVE_EPISODE "):
                    episode = json.loads(line.removeprefix("DOVE_EPISODE "))
                    episode["other"] = int(episode["reason"] not in ("success", "crash", "out_of_route", "timeout"))
                    with progress_lock:
                        state = progress[model_index]
                        state["completed"] += 1
                        state["arrived"] += episode["arrive"]
                        for key in ("crash", "out_of_route", "timeout", "other"):
                            state[key] += episode[key]
                        text = (f"第{model_index}组 {state['completed']:03d}/{args.num_episodes} "
                                f"arrive={episode['arrive']} crash={episode['crash']} "
                                f"out_of_route={episode['out_of_route']} timeout={episode['timeout']} other={episode['other']} "
                                f"success={state['arrived'] / state['completed']:.1%}")
                        append_terminal_csv(terminal_csv, {
                            "record_type": "episode", "group": model_index,
                            "episode": state["completed"], "episode_id": episode["episode"],
                            "total_episodes": args.num_episodes, "reward": episode["reward"] * reward_factor,
                            "cost": episode["cost"], "arrive": episode["arrive"],
                            "arrive_total": state["arrived"],
                            **{key: episode[key] for key in ("crash", "out_of_route", "timeout", "other")},
                            **{key + "_total": state[key] for key in ("crash", "out_of_route", "timeout", "other")},
                            "reward_factor": reward_factor,
                            "success": state["arrived"] / state["completed"], "terminal_text": text,
                        })
                        print(text, flush=True)
            if process.wait():
                raise RuntimeError(f"第{model_index}组测试失败，未更新汇总结果。")
        rows = read_csv(output / "episodes.csv")
        if len(rows) != count or [int(row["episode_idx"]) for row in rows] != list(range(start, start + count)):
            raise ValueError(f"第{model_index}组回合数据不完整，未更新汇总结果。")
        return {"filename": name, "start": start, "count": count,
                "episode_csv": str((output / "episodes.csv").relative_to(run_dir)),
                "episodes_sha256": digest(output / "episodes.csv")}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        run["shards"] = list(executor.map(evaluate_one, jobs))
    for item in manifest["models"]:
        name = item["filename"]
        rows = [row for shard in run["shards"] if shard["filename"] == name
                for row in read_csv(run_dir / shard["episode_csv"])]
        rows.sort(key=lambda row: int(row["episode_idx"]))
        output = run_dir / Path(name).stem
        write_csv(output / "episodes.csv", rows)
        write_csv(output / "summary.csv", [{key + "_mean": statistics.fmean(float(row[key]) for row in rows)
                                             for key in ("reward", "cost", "success")}])
        run["models"].append({"filename": name, "checkpoint_sha256": item["sha256"],
                              "episode_csv": str((output / "episodes.csv").relative_to(run_dir)),
                              "episodes_sha256": digest(output / "episodes.csv")})
    write_json(run_dir / "run_manifest.json", run)
    activate_run(run_dir, manifest)
    (run_dir / "pending_run.json").unlink()
    calculate(terminal_csv)


if __name__ == "__main__":
    main()
