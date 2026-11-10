"""Standard-library validation and aggregation for the three DOVE checkpoints."""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "02_data/models/final/simulation/DOVE"
DATA = ROOT / "02_data/raw_data/TABLES/TAB-01/raw_data/DOVE"
METRICS = ("reward", "cost", "success")
TERMINAL_FIELDS = ("record_type", "group", "episode", "episode_id", "total_episodes",
                   "reward", "reward_sd", "cost", "cost_sd", "arrive", "arrive_total",
                   "crash", "crash_total", "out_of_route", "out_of_route_total", "timeout", "timeout_total", "other", "other_total",
                   "reward_factor",
                   "success", "success_sd", "train_cost", "terminal_text")


def create_terminal_csv(path):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=TERMINAL_FIELDS).writeheader()


def append_terminal_csv(path, record):
    with Path(path).open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=TERMINAL_FIELDS).writerow(record)


def episode_shards(total, count):
    if total < 1 or count < 1 or count > total:
        raise ValueError("Invalid number of episodes or environment shards")
    quotient, remainder = divmod(total, count)
    result, start = [], 0
    for index in range(count):
        size = quotient + int(index < remainder)
        result.append((start, size))
        start += size
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def model_manifest():
    result = json.loads((MODELS / "manifest.json").read_text())
    names = [item["filename"] for item in result["models"]]
    if len(names) != 3 or len(set(names)) != 3:
        raise ValueError("Expected exactly three distinct DOVE checkpoint filenames")
    for item in result["models"]:
        if digest(MODELS / item["filename"]) != item["sha256"]:
            raise ValueError(f"Checkpoint SHA-256 mismatch: {item['filename']}")
    if digest(MODELS / "args.json") != result["args_sha256"]:
        raise ValueError("Training args.json SHA-256 mismatch")
    return result


def summarize_run(run_dir, manifest):
    """Average each model's episodes, then mean/sample SD over 3 model means."""
    run_dir = Path(run_dir).resolve()
    run = json.loads((run_dir / "run_manifest.json").read_text())
    expected = {m["filename"]: m for m in manifest["models"]}
    aliases = {name: model["filename"] for model in manifest["models"]
               for name in (model["filename"], model.get("original_filename", model["filename"]))}

    def canonical_name(name):
        basename = Path(name).name
        return aliases.get(basename, basename)

    names = [canonical_name(m["filename"]) for m in run["models"]]
    if len(names) != 3 or set(names) != set(expected):
        raise ValueError("Results must contain exactly the three checkpoints in manifest.json")
    if run["args_sha256"] != manifest["args_sha256"] or run["eval_flags"] != manifest["eval_flags"]:
        raise ValueError("Result configuration differs from the declared evaluation protocol")
    per_model, all_episodes = [], []
    for item in run["models"]:
        name = canonical_name(item["filename"])
        if item["checkpoint_sha256"] != expected[name]["sha256"]:
            raise ValueError(f"Results reference a different checkpoint: {name}")
        path = (run_dir / item["episode_csv"]).resolve()
        if not path.is_relative_to(run_dir):
            raise ValueError("Episode CSV must be inside its result directory")
        if digest(path) != item["episodes_sha256"]:
            raise ValueError(f"Episode CSV SHA-256 mismatch: {path}")
        rows = read_csv(path)
        if len(rows) != int(run["num_episodes"]) or len(rows) < 2:
            raise ValueError(f"Incomplete episode output: {path} ({len(rows)} rows)")
        for index, row in enumerate(rows):
            if (canonical_name(row["policy_name"]) != name or int(row["map_seed"]) != 104
                    or int(row["episode_idx"]) != index):
                raise ValueError(f"Wrong policy/map/episode index: {path}, row {index}")
            if not all(math.isfinite(float(row[key])) for key in METRICS):
                raise ValueError(f"Non-finite episode metric: {path}, row {index}")
            if float(row["success"]) not in (0.0, 1.0) or float(row["cost"]) < 0:
                raise ValueError(f"Invalid success/cost: {path}, row {index}")
            if run.get("sampling", {}).get("scheme") == "independent_episode_traffic_v1":
                if int(row["traffic_seed"]) != int(run["sampling"]["traffic_seed_base"]) + index:
                    raise ValueError(f"Wrong or duplicated traffic seed: {path}, row {index}")
        summary = {"policy_name": name, "n_episodes": len(rows)}
        for key in METRICS:
            values = [float(row[key]) for row in rows]
            summary[f"{key}_mean"] = statistics.fmean(values)
            summary[f"{key}_episode_sd"] = statistics.stdev(values)
        per_model.append(summary)
        all_episodes.extend({"policy_name": name, "map_seed": row["map_seed"],
                             "episode_idx": row["episode_idx"],
                             **{key: row[key] for key in METRICS},
                             **{key: row[key] for key in ("crash", "out_of_route", "timeout", "reason")
                                if key in row}} for row in rows)
    result = {"source": run["source"], "run_dir": str(run_dir),
              "n_checkpoints": 3, "n_episodes": len(all_episodes),
              "reward_transform": "none", "dispersion": manifest["dispersion"],
              "sampling": run.get("sampling", {"scheme": "original_single_environment_sequence"})}
    for key in METRICS:
        means = [item[f"{key}_mean"] for item in per_model]
        values = [float(row[key]) for row in all_episodes]
        result[f"{key}_mean"] = statistics.fmean(means)
        result[f"{key}_sd"] = statistics.stdev(means)
        result[f"{key}_se"] = statistics.stdev(means) / math.sqrt(3)
        result[f"{key}_pooled_episode_sd"] = statistics.stdev(values)
        result[f"{key}_pooled_episode_se"] = statistics.stdev(values) / math.sqrt(len(values))
    return result, per_model, all_episodes


def activate_run(run_dir, manifest):
    result, per_model, episodes = summarize_run(run_dir, manifest)
    write_csv(Path(run_dir) / "per_checkpoint.csv", per_model)
    write_json(Path(run_dir) / "statistics.json", result)
    write_csv(DATA / "episodes.csv", episodes)
    run = json.loads((Path(run_dir) / "run_manifest.json").read_text())
    write_json(DATA / "measurement.json", {
        "source": result["source"], "run_id": Path(run_dir).name,
        "num_episodes_per_checkpoint": run["num_episodes"],
        "episodes_sha256": digest(DATA / "episodes.csv"),
        "sampling": run.get("sampling", {"scheme": "original_single_environment_sequence"}),
    })
    write_json(DATA / "active_run.json", {"run_dir": str(Path(run_dir).resolve().relative_to(DATA)),
                                         "source": result["source"]})
    return result
