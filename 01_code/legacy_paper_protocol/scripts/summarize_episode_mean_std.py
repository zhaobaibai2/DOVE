#!/usr/bin/env python3
"""Build table-ready mean±std summaries from DOVEER episode-level eval CSVs.

Expected input layout from runs/40_eval_all_policies_all_modes_seed104.sh:

  <root>/<mode>/episodes_<mode>.csv

Each row is one episode for one policy on one map.  This script computes exact
mean/std across episode rows for the metrics that are usually reported in the
paper tables.  It is intentionally read-only: it never runs evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Sequence, Tuple

PAPER_METRICS = [
    "success",
    "crash",
    "out_of_route",
    "timeout",
    "takeover_rate",
    "cost",
    "native_cost",
    "takeover_cost",
    "route_completion",
    "reward",
    "closed_loop_distance",
    "inference_latency_mean",
    "inference_latency_p95",
    "final_gate_accept_rate",
    "final_gate_delta_c_mean",
    "final_gate_delta_e_mean",
    "final_gate_reject_proxy_rate",
    "final_gate_reject_prior_rate",
    "final_gate_reject_uncertainty_rate",
]

DISPLAY = {
    "success": "Succ.",
    "crash": "Coll.",
    "out_of_route": "OOR",
    "timeout": "Timeout",
    "takeover_rate": "Tkov.",
    "cost": "Cost",
    "native_cost": "NativeCost",
    "takeover_cost": "TakeoverCost",
    "route_completion": "Route",
    "reward": "Reward",
    "closed_loop_distance": "Dist.",
    "inference_latency_mean": "Latency",
    "inference_latency_p95": "LatencyP95",
    "final_gate_accept_rate": "GateAcc",
    "final_gate_delta_c_mean": "DeltaC",
    "final_gate_delta_e_mean": "DeltaE",
    "final_gate_reject_proxy_rate": "RejectProxy",
    "final_gate_reject_prior_rate": "RejectPrior",
    "final_gate_reject_uncertainty_rate": "RejectUnc",
}

MODE_ORDER = {
    "vanilla": 0,
    "upvgds_no_gate": 1,
    "upvgds_final_gate_full": 2,
    "final_gate_no_disagreement": 3,
    "final_gate_proxy_only": 4,
    "final_gate_prior_only": 5,
    "topk8_rerank": 6,
}

MODE_DISPLAY = {
    "vanilla": "Vanilla / w/o GDS",
    "upvgds_no_gate": "UPV-GDS w/o Gate",
    "upvgds_final_gate_full": "Full UPV-GDS + Final Gate",
    "final_gate_no_disagreement": "w/o disagreement gate",
    "final_gate_proxy_only": "Proxy-only gate",
    "final_gate_prior_only": "Prior-only gate",
    "topk8_rerank": "Top-K rerank",
}


def _mode_from_path(path: Path) -> str:
    stem = path.stem
    if stem.startswith("episodes_"):
        return stem[len("episodes_") :]
    return path.parent.name


def _step_of(policy_name: str) -> int:
    name = Path(policy_name).name
    if name == "stage1b_offline_bc_policy.pkl":
        return -1
    for pattern in (r"policy-(\d+)-", r"final_policy_(\d+)", r"policy_\D*(\d+)", r"final_bundle_step_(\d+)"):
        m = re.search(pattern, name)
        if m:
            return int(m.group(1))
    return 10**18


def _to_float(value: str) -> float:
    try:
        x = float(value)
        return x if math.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _fmt(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def _mean_std(values: Sequence[float]) -> Tuple[float, float, int]:
    clean = [v for v in values if math.isfinite(v)]
    if not clean:
        return float("nan"), float("nan"), 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return mean(clean), stdev(clean), len(clean)


def _mean_pm_std(values: Sequence[float], digits: int = 3) -> str:
    m, s, n = _mean_std(values)
    if n == 0:
        return "--"
    return f"{_fmt(m, digits)} $\\pm$ {_fmt(s, digits)}"


def _collect_episode_csvs(root: Path, inputs: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    if root:
        paths.extend(sorted(root.glob("**/episodes_*.csv")))
    for item in inputs:
        p = Path(item).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("**/episodes_*.csv")))
        elif p.is_file():
            paths.append(p)
    dedup: Dict[Path, Path] = {}
    for p in paths:
        if p.is_file():
            dedup[p.resolve()] = p
    return list(dedup.values())


def _rank_key(row: Dict[str, str]) -> Tuple[float, float, float, float, float, int]:
    def f(key: str, default: float) -> float:
        x = _to_float(row.get(f"{key}_mean", "nan"))
        return x if math.isfinite(x) else default

    return (
        -f("success", -1.0),
        f("crash", 1e9),
        f("out_of_route", 1e9),
        f("timeout", 1e9),
        -f("reward", -1e9),
        int(row.get("step", str(10**18))),
    )


def read_and_summarize(paths: Sequence[Path], metrics: Sequence[str]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = defaultdict(list)
    source_by_group: Dict[Tuple[str, str], str] = {}
    for src in paths:
        mode = _mode_from_path(src)
        with src.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for ep in reader:
                policy = ep.get("policy_name") or ep.get("policy") or ""
                key = (mode, policy)
                grouped[key].append(ep)
                source_by_group[key] = str(src)

    out: List[Dict[str, str]] = []
    for (mode, policy), episodes in grouped.items():
        row: Dict[str, str] = {
            "mode": mode,
            "mode_display": MODE_DISPLAY.get(mode, mode),
            "policy": policy,
            "policy_basename": Path(policy).name,
            "step": str(_step_of(policy)),
            "n_episodes": str(len(episodes)),
            "source_episode_csv": source_by_group[(mode, policy)],
        }
        for metric in metrics:
            values = [_to_float(ep.get(metric, "nan")) for ep in episodes]
            m, s, n = _mean_std(values)
            row[f"{metric}_mean"] = "" if not math.isfinite(m) else f"{m:.10g}"
            row[f"{metric}_std"] = "" if not math.isfinite(s) else f"{s:.10g}"
            row[f"{metric}_n"] = str(n)
            row[f"{metric}_pm_std"] = _mean_pm_std(values, digits=3)
        out.append(row)
    return sorted(out, key=lambda r: (MODE_ORDER.get(r["mode"], 999), _rank_key(r)))


def write_csv(path: Path, rows: Sequence[Dict[str, str]], metrics: Sequence[str]) -> None:
    fields = ["mode", "mode_display", "policy", "policy_basename", "step", "n_episodes"]
    for metric in metrics:
        fields += [f"{metric}_mean", f"{metric}_std", f"{metric}_n", f"{metric}_pm_std"]
    fields += ["source_episode_csv"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def write_markdown(path: Path, rows: Sequence[Dict[str, str]], top_n: int) -> None:
    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["mode"]].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Table-ready episode mean/std summary\n\n")
        f.write("Values are computed from episode-level CSV rows. Use these for paper tables only after the checkpoint was selected on validation maps/seeds, not by test-set sweeping.\n\n")
        f.write("## Best checkpoint per mode\n\n")
        f.write("| mode | policy | n | Succ. | Coll. | OOR | Route | Reward | Delta C | GateAcc | Latency |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for mode in sorted(grouped, key=lambda m: MODE_ORDER.get(m, 999)):
            row = grouped[mode][0]
            f.write(
                f"| {row['mode_display']} | `{row['policy_basename']}` | {row['n_episodes']} | "
                f"{row['success_pm_std']} | {row['crash_pm_std']} | {row['out_of_route_pm_std']} | "
                f"{row['route_completion_pm_std']} | {row['reward_pm_std']} | "
                f"{row['final_gate_delta_c_mean_pm_std']} | {row['final_gate_accept_rate_pm_std']} | "
                f"{row['inference_latency_mean_pm_std']} |\n"
            )
        f.write("\n## Top checkpoints by mode\n\n")
        for mode in sorted(grouped, key=lambda m: MODE_ORDER.get(m, 999)):
            f.write(f"### {MODE_DISPLAY.get(mode, mode)}\n\n")
            f.write("| rank | policy | n | Succ. | Coll. | OOR | Route | Reward |\n")
            f.write("|---:|---|---:|---:|---:|---:|---:|---:|\n")
            for rank, row in enumerate(grouped[mode][: max(top_n, 1)], 1):
                f.write(
                    f"| {rank} | `{row['policy_basename']}` | {row['n_episodes']} | "
                    f"{row['success_pm_std']} | {row['crash_pm_std']} | {row['out_of_route_pm_std']} | "
                    f"{row['route_completion_pm_std']} | {row['reward_pm_std']} |\n"
                )
            f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize episode-level DOVEER eval CSVs as mean±std tables.")
    ap.add_argument("--root", type=Path, default=None, help="Root directory containing mode subdirs with episodes_*.csv")
    ap.add_argument("--inputs", nargs="*", default=[], help="Specific episode CSV files or directories")
    ap.add_argument("--output", required=True, help="Output table-ready CSV")
    ap.add_argument("--markdown-output", default=None, help="Output Markdown summary")
    ap.add_argument("--metrics", nargs="*", default=PAPER_METRICS, help="Episode metric keys to summarize")
    ap.add_argument("--top-n", type=int, default=20, help="Top N policies per mode in Markdown")
    args = ap.parse_args()

    paths = _collect_episode_csvs(args.root, args.inputs)
    if not paths:
        raise SystemExit("No episodes_*.csv files found. Re-run eval with the patched --episode_output_csv support.")
    rows = read_and_summarize(paths, args.metrics)
    write_csv(Path(args.output), rows, args.metrics)
    if args.markdown_output:
        write_markdown(Path(args.markdown_output), rows, args.top_n)
    print(f"[episode_csvs] {len(paths)}")
    print(f"[policy_mode_rows] {len(rows)}")
    print(f"[output] {args.output}")
    if args.markdown_output:
        print(f"[markdown] {args.markdown_output}")


if __name__ == "__main__":
    main()
