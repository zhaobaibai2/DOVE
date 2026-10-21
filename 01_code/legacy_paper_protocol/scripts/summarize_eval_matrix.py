#!/usr/bin/env python3
"""Summarize DOVEER eval CSVs produced by scripts/eval_pvp_policies_fixed_v3.py.

This helper is intentionally lightweight: it only reads CSV files and writes
combined CSV/Markdown summaries. It does not run evaluation.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


def _float(row: Dict[str, str], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
        return value if math.isfinite(value) else float("nan")
    except Exception:
        return float("nan")


def _fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:.{digits}g}"


def _step_of(policy_name: str) -> int:
    name = Path(policy_name).name
    if name == "stage1b_offline_bc_policy.pkl":
        return -1
    for pattern in (r"policy-(\d+)-", r"final_policy_(\d+)", r"policy_\D*(\d+)", r"final_bundle_step_(\d+)"):
        match = re.search(pattern, policy_name)
        if match:
            return int(match.group(1))
    return 10**18


def _mode_from_csv(path: Path) -> str:
    stem = path.stem
    if stem.startswith("eval_"):
        return stem[len("eval_") :]
    return path.parent.name


def _collect_input_csvs(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    for item in args.inputs or []:
        p = Path(item).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("**/eval_*.csv")))
        else:
            paths.append(p)
    if args.root:
        root = Path(args.root).expanduser()
        paths.extend(sorted(root.glob("**/eval_*.csv")))
    dedup: Dict[Path, Path] = {}
    for p in paths:
        if p.is_file():
            dedup[p.resolve()] = p
    return list(dedup.values())


def _rank_key(row: Dict[str, str]):
    success = _float(row, "avg_success")
    reward = _float(row, "avg_reward")
    crash = _float(row, "avg_crash")
    oor = _float(row, "avg_out_of_route")
    timeout = _float(row, "avg_timeout")
    step = _step_of(row.get("policy", row.get("policy_name", "")))
    return (
        -(success if math.isfinite(success) else -1.0),
        (crash if math.isfinite(crash) else 1e9),
        (oor if math.isfinite(oor) else 1e9),
        (timeout if math.isfinite(timeout) else 1e9),
        -(reward if math.isfinite(reward) else -1e9),
        step,
    )


def read_rows(csv_paths: Sequence[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for src in csv_paths:
        mode = _mode_from_csv(src)
        with src.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                policy = row.get("policy_name") or row.get("policy") or ""
                rows.append(
                    {
                        "mode": mode,
                        "policy": policy,
                        "step": str(_step_of(policy)),
                        "avg_success": _fmt(_float(row, "avg_success")),
                        "avg_reward": _fmt(_float(row, "avg_reward")),
                        "avg_crash": _fmt(_float(row, "avg_crash")),
                        "avg_out_of_route": _fmt(_float(row, "avg_out_of_route")),
                        "avg_timeout": _fmt(_float(row, "avg_timeout")),
                        "avg_takeover_rate": _fmt(_float(row, "avg_takeover_rate")),
                        "avg_route_completion": _fmt(_float(row, "avg_route_completion")),
                        "avg_closed_loop_distance": _fmt(_float(row, "avg_closed_loop_distance")),
                        "avg_inference_latency_mean": _fmt(_float(row, "avg_inference_latency_mean")),
                        "avg_inference_latency_p95": _fmt(_float(row, "avg_inference_latency_p95")),
                        "avg_final_gate_accept_rate": _fmt(_float(row, "avg_final_gate_accept_rate")),
                        "avg_final_gate_delta_c_mean": _fmt(_float(row, "avg_final_gate_delta_c_mean")),
                        "avg_final_gate_delta_e_mean": _fmt(_float(row, "avg_final_gate_delta_e_mean")),
                        "avg_final_gate_reject_proxy_rate": _fmt(_float(row, "avg_final_gate_reject_proxy_rate")),
                        "avg_final_gate_reject_prior_rate": _fmt(_float(row, "avg_final_gate_reject_prior_rate")),
                        "avg_final_gate_reject_uncertainty_rate": _fmt(_float(row, "avg_final_gate_reject_uncertainty_rate")),
                        "source_csv": str(src),
                    }
                )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_markdown(path: Path, best_by_mode: Sequence[Dict[str, str]], top_by_mode: Dict[str, List[Dict[str, str]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Evaluation matrix summary\n\n")
        f.write("## Best checkpoint per mode\n\n")
        f.write("| mode | step | policy | success | reward | crash | OOR | timeout | route | latency | GateAcc | Delta C |\n")
        f.write("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in best_by_mode:
            f.write(
                f"| {row['mode']} | {row['step']} | `{Path(row['policy']).name}` | "
                f"{row['avg_success']} | {row['avg_reward']} | {row['avg_crash']} | "
                f"{row['avg_out_of_route']} | {row['avg_timeout']} | {row['avg_route_completion']} | "
                f"{row['avg_inference_latency_mean']} | {row['avg_final_gate_accept_rate']} | "
                f"{row['avg_final_gate_delta_c_mean']} |\n"
            )
        f.write("\n## Top checkpoints by mode\n\n")
        for mode, rows in top_by_mode.items():
            f.write(f"### {mode}\n\n")
            f.write("| rank | step | policy | success | reward | crash | OOR | timeout | route |\n")
            f.write("|---:|---:|---|---:|---:|---:|---:|---:|---:|\n")
            for rank, row in enumerate(rows, 1):
                f.write(
                    f"| {rank} | {row['step']} | `{Path(row['policy']).name}` | "
                    f"{row['avg_success']} | {row['avg_reward']} | {row['avg_crash']} | "
                    f"{row['avg_out_of_route']} | {row['avg_timeout']} | {row['avg_route_completion']} |\n"
                )
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize one or more DOVEER eval CSV files.")
    parser.add_argument("--root", default=None, help="Directory containing per-mode eval_*.csv files")
    parser.add_argument("--inputs", nargs="*", default=[], help="Specific CSV files or directories to include")
    parser.add_argument("--output", required=True, help="Combined summary CSV path")
    parser.add_argument("--best-output", default=None, help="Best-per-mode CSV path")
    parser.add_argument("--markdown-output", default=None, help="Markdown report path")
    parser.add_argument("--top-n", type=int, default=20, help="Top N policies per mode for the Markdown report")
    args = parser.parse_args()

    csv_paths = _collect_input_csvs(args)
    if not csv_paths:
        raise SystemExit("No eval CSV files found. Pass --root or --inputs.")

    rows = read_rows(csv_paths)
    fields = [
        "mode", "policy", "step", "avg_success", "avg_reward", "avg_crash", "avg_out_of_route", "avg_timeout",
        "avg_takeover_rate", "avg_route_completion", "avg_closed_loop_distance", "avg_inference_latency_mean",
        "avg_inference_latency_p95", "avg_final_gate_accept_rate", "avg_final_gate_delta_c_mean",
        "avg_final_gate_delta_e_mean", "avg_final_gate_reject_proxy_rate", "avg_final_gate_reject_prior_rate",
        "avg_final_gate_reject_uncertainty_rate", "source_csv",
    ]
    rows_sorted = sorted(rows, key=lambda r: (r["mode"], _rank_key(r)))
    write_csv(Path(args.output), rows_sorted, fields)

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows_sorted:
        grouped[row["mode"]].append(row)
    best_by_mode = [items[0] for _, items in sorted(grouped.items()) if items]

    if args.best_output:
        write_csv(Path(args.best_output), best_by_mode, fields)
    if args.markdown_output:
        top_by_mode = {mode: items[: max(args.top_n, 1)] for mode, items in sorted(grouped.items())}
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        write_markdown(Path(args.markdown_output), best_by_mode, top_by_mode)

    print(f"[inputs] {len(csv_paths)} CSV files")
    print(f"[rows] {len(rows_sorted)} policy-mode rows")
    print(f"[combined] {args.output}")
    if args.best_output:
        print(f"[best] {args.best_output}")
    if args.markdown_output:
        print(f"[markdown] {args.markdown_output}")


if __name__ == "__main__":
    main()
