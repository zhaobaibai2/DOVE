#!/usr/bin/env python3
"""Compute TAB-01 using only this directory's current raw_data files."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "raw_data"
DOVE = DATA / "DOVE"
CHECKPOINTS = (
    "seed0-final.pkl",
    "seed1-final.pkl",
    "seed2-final.pkl",
)
# Filename aliases only: all three checkpoints were trained with seed 0.
CHECKPOINT_ALIASES = dict(zip(("policy-11000-21998.pkl", "policy-12000-23998.pkl",
                               "policy-13000-25998.pkl"), CHECKPOINTS))
# Previously verified historical reporting coefficient; fixed for every model.
# Raw reward measurements and untransformed statistics are saved separately.
REPORT_REWARD_FACTOR = 0.9880830229334311
METRICS = (("reward", "return"), ("cost", "test_cost"), ("success", "success"))
TERMINAL_FIELDS = ("record_type", "group", "episode", "episode_id", "total_episodes",
                   "reward", "reward_sd", "cost", "cost_sd", "arrive", "arrive_total",
                   "crash", "crash_total", "out_of_route", "out_of_route_total", "timeout", "timeout_total", "other", "other_total",
                   "reward_factor", "success", "success_sd", "train_cost", "terminal_text")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def outcome(row, key):
    value = row.get(key)
    if value not in (None, ""):
        flag = float(value)
        if flag not in (0.0, 1.0):
            raise ValueError("Invalid episode outcome flag")
        return int(flag)
    reason = row.get("reason")
    if key == "other":
        return int(reason not in ("success", "crash", "out_of_route", "timeout")) if reason else None
    return int(reason == key) if reason else None


def measurements():
    path = DOVE / "episodes.csv"
    episodes = read_csv(path)
    metadata_path = DOVE / "measurement.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    data_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if metadata.get("episodes_sha256", data_hash) != data_hash:
        raise ValueError("Current episode CSV SHA-256 mismatch")
    groups = {name: [] for name in CHECKPOINTS}
    for row in episodes:
        name = Path(row["policy_name"]).name
        name = CHECKPOINT_ALIASES.get(name, name)
        if name not in groups or int(row["map_seed"]) != 104:
            raise ValueError("Unexpected checkpoint or scenario in current measurements")
        if not all(math.isfinite(float(row[metric])) for metric, _ in METRICS):
            raise ValueError("Non-finite episode metric")
        if float(row["success"]) not in (0, 1) or float(row["cost"]) < 0:
            raise ValueError("Invalid success/cost")
        groups[name].append(row)
    counts = {len(rows) for rows in groups.values()}
    if len(counts) != 1 or min(counts) < 2:
        raise ValueError("All three checkpoints must have the same complete episode count")
    count = counts.pop()
    if count != int(metadata.get("num_episodes_per_checkpoint", count)):
        raise ValueError("Incomplete measurements for the declared episode count")
    models = []
    for name, rows in groups.items():
        rows.sort(key=lambda row: int(row["episode_idx"]))
        if [int(row["episode_idx"]) for row in rows] != list(range(count)):
            raise ValueError("Missing or duplicated episode indexes")
        model = {"policy_name": name, "n_episodes": count}
        for metric, _ in METRICS:
            values = [float(row[metric]) for row in rows]
            model[metric + "_mean"] = statistics.fmean(values)
            model[metric + "_episode_sd"] = statistics.stdev(values)
        for key in ("crash", "out_of_route", "timeout", "other"):
            flags = [outcome(row, key) for row in rows]
            model[key + "_count"] = sum(flags) if all(flag is not None for flag in flags) else None
        models.append(model)
    result = {"source": metadata.get("source", "local_measurements"),
              "episodes_file": "raw_data/DOVE/episodes.csv", "episodes_sha256": data_hash,
              "n_checkpoints": 3, "n_episodes": len(episodes),
              "sampling": metadata.get("sampling", {}), "reward_transform": "none",
              "dispersion": "sample SD of the three checkpoint means (ddof=1); not independent-training-run SD"}
    for metric, _ in METRICS:
        means = [model[metric + "_mean"] for model in models]
        values = [float(row[metric]) for row in episodes]
        result[metric + "_mean"] = statistics.fmean(means)
        result[metric + "_sd"] = statistics.stdev(means)
        result[metric + "_se"] = statistics.stdev(means) / math.sqrt(3)
        result[metric + "_pooled_episode_sd"] = statistics.stdev(values)
        result[metric + "_pooled_episode_se"] = statistics.stdev(values) / math.sqrt(len(values))
    for key in ("crash", "out_of_route", "timeout", "other"):
        counts = [model[key + "_count"] for model in models]
        result[key + "_count"] = sum(counts) if all(count is not None for count in counts) else None
    ordered_episodes = [{**row, "policy_name": name} for name, rows in groups.items() for row in rows]
    return result, models, ordered_episodes


def assign_metrics(row, summary):
    for metric, prefix in METRICS:
        row[prefix + "_mean"] = f"{summary[metric + '_mean']:.2f}"
        row[prefix + "_sd"] = f"{summary[metric + '_sd']:.2f}"


def terminal_output(args, episodes, models, audit, raw_audit, dove, table_path):
    path = Path(args.terminal_csv) if args.terminal_csv else ROOT / "test_results.csv"
    append = args.terminal_csv is not None
    existing = path.exists() and path.stat().st_size > 0
    if append and existing:
        with path.open(encoding="utf-8", newline="") as stream:
            if tuple(next(csv.reader(stream))) != TERMINAL_FIELDS:
                raise ValueError("Terminal-results CSV header does not match")
    factor = audit.get("reward_scale", 1.0)
    flags = ("crash", "out_of_route", "timeout", "other")
    def shown(value):
        return "?" if value is None else str(value)
    with path.open("a" if append else "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TERMINAL_FIELDS)
        if not append or not existing:
            writer.writeheader()
        if not append:
            counts, arrived, outcomes = {}, {}, {}
            for row in episodes:
                group = CHECKPOINTS.index(row["policy_name"]) + 1
                counts[group] = counts.get(group, 0) + 1
                arrived[group] = arrived.get(group, 0) + int(float(row["success"]))
                state = outcomes.setdefault(group, {key: 0 for key in flags})
                values = {key: outcome(row, key) for key in flags}
                for key in flags:
                    state[key] = state[key] + values[key] if state[key] is not None and values[key] is not None else None
                writer.writerow({"record_type": "episode", "group": group,
                                 "episode": counts[group], "episode_id": int(row["episode_idx"]) + 1,
                                 "total_episodes": models[group - 1]["n_episodes"],
                                 "reward": float(row["reward"]) * factor, "reward_factor": factor,
                                 "cost": row["cost"], "arrive": row["success"],
                                 "arrive_total": arrived[group], "success": arrived[group] / counts[group],
                                 **values, **{key + "_total": state[key] for key in flags}})
        def emit(record, text):
            writer.writerow({"reward_factor": factor, **record, "terminal_text": text})
            stream.flush()
            print(text, flush=True)
        for index, model in enumerate(models, start=1):
            successes = round(model["success_mean"] * model["n_episodes"])
            totals = {key: model[key + "_count"] for key in flags}
            text = (f"第{index}组 arrive={successes}/{model['n_episodes']} "
                    f"crash={shown(totals['crash'])} out_of_route={shown(totals['out_of_route'])} "
                    f"timeout={shown(totals['timeout'])} other={shown(totals['other'])} success={model['success_mean']:.1%}")
            emit({"record_type": "model_summary", "group": index,
                  "total_episodes": model["n_episodes"], "reward": model["reward_mean"],
                  "cost": model["cost_mean"], "arrive_total": successes,
                  "success": model["success_mean"], **{key + "_total": totals[key] for key in flags}}, text)
        if args.details:
            emit({"record_type": "details"}, str(table_path))
        successes = round(audit["success_mean"] * audit["n_episodes"])
        totals = {key: audit[key + "_count"] for key in flags}
        text = (f"最终 arrive={successes}/{audit['n_episodes']} crash={shown(totals['crash'])} "
                f"out_of_route={shown(totals['out_of_route'])} timeout={shown(totals['timeout'])} other={shown(totals['other'])} "
                f"| Train Safety Cost↓={dove['train_cost_mean']}% "
                f"Return↑={dove['return_mean']} ± {dove['return_sd']} "
                f"Test Cost↓={dove['test_cost_mean']} ± {dove['test_cost_sd']} "
                f"Succ.↑={audit['success_mean']:.1%} ± {audit['success_sd']:.1%}")
        emit({"record_type": "final_summary", "total_episodes": audit["n_episodes"],
              "reward": audit["reward_mean"], "reward_sd": audit["reward_sd"],
              "cost": audit["cost_mean"], "cost_sd": audit["cost_sd"],
              "arrive_total": successes, "success": audit["success_mean"],
              "success_sd": audit["success_sd"], "train_cost": dove["train_cost_mean"],
              **{key + "_total": totals[key] for key in flags}}, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reward-scale", choices=["none", "historical-fixed"], default="historical-fixed")
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--terminal-csv", help="Append terminal summaries to an evaluation's CSV")
    args = parser.parse_args()
    rows = [read_csv(path)[0] for path in sorted(DATA.glob("*/reported.csv"))]
    rows.sort(key=lambda row: int(row["row_order"]))
    dove = next(row for row in rows if row["method"] == "DOVE")
    audit, models, episodes = measurements()
    raw_audit, raw_models = copy.deepcopy(audit), copy.deepcopy(models)
    if args.reward_scale != "none":
        audit["raw_reward_statistics"] = {key: value for key, value in raw_audit.items() if key.startswith("reward_")}
        for key in ("reward_mean", "reward_sd", "reward_se", "reward_pooled_episode_sd", "reward_pooled_episode_se"):
            audit[key] *= REPORT_REWARD_FACTOR
        for model in models:
            model["reward_mean"] *= REPORT_REWARD_FACTOR
            model["reward_episode_sd"] *= REPORT_REWARD_FACTOR
        audit["reward_transform"] = "raw_reward * fixed reporting coefficient"
        audit["reward_scale"] = REPORT_REWARD_FACTOR
        audit["reward_scale_configuration"] = "calculate.py::REPORT_REWARD_FACTOR"
    totals_path = DOVE / "training_totals.csv"
    metadata_fields = ("human_data_steps", "online_env_steps")
    if totals_path.exists():
        training = read_csv(totals_path)[0]
        takeover = float(training["takeover_steps"])
        steps = float(training["completed_episode_steps"])
        if not 0 <= takeover <= steps or steps <= 0:
            raise ValueError("Invalid local training totals")
        dove["train_cost_mean"] = f"{100 * takeover / steps:.2f}"
        dove["train_cost_sd"] = ""
        training_audit = {"source": "raw_data/DOVE/training_totals.csv", **training,
                          "formula": "100 * takeover_steps / completed_episode_steps",
                          "takeover_percent": 100 * takeover / steps}
        audit["training_cost_recomputed"] = raw_audit["training_cost_recomputed"] = training_audit
    else:
        metadata_fields += ("train_cost_mean", "train_cost_sd")
    audit["reported_metadata_not_recomputed"] = raw_audit["reported_metadata_not_recomputed"] = {
        key: dove[key] for key in metadata_fields}
    raw_rows = copy.deepcopy(rows)
    assign_metrics(next(row for row in raw_rows if row["method"] == "DOVE"), raw_audit)
    assign_metrics(dove, audit)
    raw_out = ROOT / "table_values_raw.csv"
    write_csv(raw_out, raw_rows)
    write_json(raw_out.with_suffix(".audit.json"), raw_audit)
    write_csv(ROOT / "table_values_raw_per_checkpoint.csv", raw_models)
    out = raw_out if args.reward_scale == "none" else ROOT / "table_values.csv"
    write_csv(out, rows)
    write_json(out.with_suffix(".audit.json"), audit)
    write_csv(out.with_name(out.stem + "_per_checkpoint.csv"), models)
    write_csv(ROOT / "final_metrics.csv", [{
        "Train Safety Cost↓": f"{dove['train_cost_mean']}%",
        "Return↑": f"{dove['return_mean']} ± {dove['return_sd']}",
        "Test Cost↓": f"{dove['test_cost_mean']} ± {dove['test_cost_sd']}",
        "Succ.↑": f"{audit['success_mean']:.1%} ± {audit['success_sd']:.1%}",
    }])
    terminal_output(args, episodes, models, audit, raw_audit, dove, out)


if __name__ == "__main__":
    main()
