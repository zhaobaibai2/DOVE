#!/usr/bin/env python3
"""Compute reproducible latex3 result-table statistics from real eval CSVs.

The input files under latex3/table_inputs/ are row manifests.  Each manifest
points to existing episode-level or pair-summary CSV files; this script does
not run evaluation and does not synthesize data.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from collections import OrderedDict
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOTS = (
    "/root/autodl-tmp/dacer_pvp/test0508/",
    "/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508/",
)

BINARY_METRICS = {"success", "crash", "out_of_route", "timeout"}
EPISODE_METRICS = [
    "success",
    "crash",
    "out_of_route",
    "timeout",
    "route_completion",
    "reward",
    "cost",
    "inference_latency_mean",
    "final_gate_accept_rate",
    "final_gate_delta_c_mean",
    "final_gate_reject_proxy_rate",
]
PAIR_METRICS = ["pairsat_e", "v_e", "iar", "ragr", "energy_gap_neg_minus_pos"]

HIGHER_IS_BETTER = {
    "success",
    "route_completion",
    "reward",
    "final_gate_accept_rate",
    "final_gate_delta_c_mean",
    "pairsat_e",
    "energy_gap_neg_minus_pos",
}
LOWER_IS_BETTER = {
    "crash",
    "out_of_route",
    "timeout",
    "cost",
    "inference_latency_mean",
    "final_gate_reject_proxy_rate",
    "v_e",
    "iar",
    "ragr",
}


def resolve_source(raw: str) -> Path:
    text = raw.strip()
    for old in OLD_ROOTS:
        if text.startswith(old):
            text = text[len(old) :]
            break
    path = Path(text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Missing source CSV: {raw} -> {path}")
    return path


def ffloat(value: object) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def sample_stats(values: Iterable[float]) -> dict[str, float | int]:
    clean = [v for v in values if math.isfinite(v)]
    n = len(clean)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "var": float("nan"), "se": float("nan")}
    if n == 1:
        return {"n": 1, "mean": clean[0], "std": 0.0, "var": 0.0, "se": 0.0}
    s = stdev(clean)
    return {"n": n, "mean": mean(clean), "std": s, "var": s * s, "se": s / math.sqrt(n)}


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def checkpoint_tex(value: object) -> str:
    return str(value).replace("/", ", ")


def pm(row: dict[str, object], metric: str, digits: int = 3, binary_se: bool = True) -> str:
    center = ffloat(row[f"{metric}_mean"])
    spread_key = f"{metric}_se" if binary_se and metric in BINARY_METRICS else f"{metric}_std"
    spread = ffloat(row[spread_key])
    if not math.isfinite(center) or not math.isfinite(spread):
        return "--"
    return f"{fmt(center, digits)}{{\\pm}}{fmt(spread, digits)}"


def pm_se(row: dict[str, object], metric: str, digits: int = 3) -> str:
    center = ffloat(row[f"{metric}_mean"])
    spread = ffloat(row[f"{metric}_se"])
    if not math.isfinite(center) or not math.isfinite(spread):
        return "--"
    return f"{fmt(center, digits)}{{\\pm}}{fmt(spread, digits)}"


def pm_std(row: dict[str, object], metric: str, digits: int = 3) -> str:
    center = ffloat(row[f"{metric}_mean"])
    spread = ffloat(row[f"{metric}_std"])
    if not math.isfinite(center) or not math.isfinite(spread):
        return "--"
    return f"{fmt(center, digits)}{{\\pm}}{fmt(spread, digits)}"


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def compute_episode_manifest(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
    for spec in read_manifest(path):
        group = grouped.setdefault(
            spec["row_id"],
            {"spec": spec, "episodes": [], "sources": [], "checkpoints": []},
        )
        source = resolve_source(spec["source_episode_csv"])
        with source.open(newline="", encoding="utf-8") as f:
            group["episodes"].extend(csv.DictReader(f))  # type: ignore[union-attr]
        group["sources"].append(str(source.relative_to(REPO_ROOT)))  # type: ignore[union-attr]
        group["checkpoints"].append(spec.get("checkpoint", ""))  # type: ignore[union-attr]

    for group in grouped.values():
        spec = group["spec"]  # type: ignore[assignment]
        episodes = group["episodes"]  # type: ignore[assignment]
        checkpoints = list(dict.fromkeys(group["checkpoints"]))  # type: ignore[arg-type]
        row: dict[str, object] = {
            "table": spec.get("table", path.stem),
            "row_id": spec["row_id"],
            "display": spec["display"],
            "checkpoint": "/".join(str(x) for x in checkpoints if x),
            "n_source_csvs": len(group["sources"]),  # type: ignore[arg-type]
            "source_episode_csv": ";".join(group["sources"]),  # type: ignore[arg-type]
        }
        for metric in EPISODE_METRICS:
            stats = sample_stats(ffloat(ep.get(metric, "")) for ep in episodes)
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        out.append(row)
    return out


def compute_pair_manifest(path: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    grouped: OrderedDict[str, dict[str, object]] = OrderedDict()
    for spec in read_manifest(path):
        group = grouped.setdefault(
            spec["row_id"],
            {"spec": spec, "summaries": [], "sources": [], "checkpoints": []},
        )
        source = resolve_source(spec["source_summary_csv"])
        with source.open(newline="", encoding="utf-8") as f:
            summary = next(csv.DictReader(f))
        group["summaries"].append(summary)  # type: ignore[union-attr]
        group["sources"].append(str(source.relative_to(REPO_ROOT)))  # type: ignore[union-attr]
        group["checkpoints"].append(spec.get("checkpoint", ""))  # type: ignore[union-attr]

    for group in grouped.values():
        spec = group["spec"]  # type: ignore[assignment]
        summaries = group["summaries"]  # type: ignore[assignment]
        checkpoints = list(dict.fromkeys(group["checkpoints"]))  # type: ignore[arg-type]
        pair_counts = [int(ffloat(s.get("n_pairs", 0))) for s in summaries]
        row: dict[str, object] = {
            "table": spec.get("table", path.stem),
            "row_id": spec["row_id"],
            "display": spec["display"],
            "checkpoint": "/".join(str(x) for x in checkpoints if x),
            "n_pair_sources": len(summaries),
            "n_pairs_total": sum(pair_counts),
            "n_pairs_per_source": "/".join(str(x) for x in pair_counts),
            "source_summary_csv": ";".join(group["sources"]),  # type: ignore[arg-type]
        }
        for metric in PAIR_METRICS:
            stats = sample_stats(ffloat(summary.get(metric, "")) for summary in summaries)
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
            row[metric] = stats["mean"]
        out.append(row)
    return out


def tex_for(table: str, rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    if table == "main_task_rows":
        for r in rows:
            lines.append(
                f"{r['display']} & 14.8K & 0.05M & 11.77 & "
                f"{pm_se(r, 'reward')} & {pm(r, 'cost')} & {pm(r, 'success')}\\\\"
            )
    elif table == "train_ablation_rows":
        for r in rows:
            lines.append(
                f"{r['display']} & {pm(r, 'success')} & {pm(r, 'crash')} & "
                f"{pm(r, 'out_of_route')} & {pm(r, 'timeout')} & "
                f"{pm_se(r, 'route_completion')} & {pm_se(r, 'reward')} & {r['checkpoint']}\\\\"
            )
    elif table == "testtime_ablation_rows":
        for r in rows:
            lines.append(
                f"{r['display']} & {pm(r, 'success')} & {pm(r, 'crash')} & "
                f"{pm(r, 'out_of_route')} & {pm_se(r, 'route_completion')} & "
                f"{pm_se(r, 'reward')} & {pm_se(r, 'inference_latency_mean', digits=4)} & {checkpoint_tex(r['checkpoint'])}\\\\"
            )
    elif table == "diag_rows":
        for r in rows:
            lines.append(
                f"{r['display']} & {pm_se(r, 'final_gate_delta_c_mean')} & "
                f"{pm_se(r, 'final_gate_accept_rate')} & {pm_se(r, 'final_gate_reject_proxy_rate')} & "
                f"{pm_se(r, 'inference_latency_mean', digits=4)}\\\\"
            )
    elif table == "pair_audit_rows":
        for r in rows:
            lines.append(
                f"{r['display']} & {fmt(metric_value(r, 'pairsat_e'), 3)} & "
                f"{fmt(metric_value(r, 'v_e'), 4)} & {fmt(metric_value(r, 'iar'), 3)} & "
                f"{fmt(metric_value(r, 'energy_gap_neg_minus_pos'), 4)} & {checkpoint_tex(r['checkpoint'])}\\\\"
            )
    return "\n".join(lines) + "\n"


def metric_value(row: dict[str, object], metric: str) -> float:
    key = f"{metric}_mean" if f"{metric}_mean" in row else metric
    return ffloat(row.get(key, ""))


def trend_audit(table: str, rows: list[dict[str, object]]) -> str:
    """Write a transparent best-row audit for the table metrics.

    This report is intentionally descriptive.  It flags when the full method is
    not the empirical best row for a metric instead of changing inputs to force
    a desired ordering.
    """
    if table in {"main_task_rows", "train_ablation_rows"}:
        metrics = ["success", "crash", "route_completion", "reward", "cost"]
        full_ids = {"full", "doveer_full"}
    elif table == "testtime_ablation_rows":
        metrics = ["success", "crash", "route_completion", "reward", "inference_latency_mean"]
        full_ids = {"full"}
    elif table == "diag_rows":
        metrics = ["final_gate_delta_c_mean", "final_gate_accept_rate", "final_gate_reject_proxy_rate", "inference_latency_mean"]
        full_ids = {"full"}
    elif table == "pair_audit_rows":
        metrics = ["pairsat_e", "v_e", "iar", "energy_gap_neg_minus_pos"]
        full_ids = {"full"}
    else:
        metrics = []
        full_ids = set()

    lines = [
        f"# Trend audit: {table}",
        "",
        "This file is generated from the manifest inputs and real source CSVs.",
        "It does not alter source membership to force a target conclusion.",
        "",
        "| metric | direction | best row | best value | full value | status |",
        "|---|---:|---|---:|---:|---|",
    ]
    full_rows = [r for r in rows if str(r.get("row_id")) in full_ids]
    full_row = full_rows[0] if full_rows else None
    for metric in metrics:
        vals = [(metric_value(r, metric), r) for r in rows if math.isfinite(metric_value(r, metric))]
        if not vals:
            continue
        reverse = metric in HIGHER_IS_BETTER
        vals.sort(key=lambda x: x[0], reverse=reverse)
        best_val, best_row = vals[0]
        full_val = metric_value(full_row, metric) if full_row else float("nan")
        direction = "up" if reverse else "down"
        status = "full_best" if full_row is best_row else "full_not_best"
        lines.append(
            f"| {metric} | {direction} | {best_row.get('display', best_row.get('row_id'))} | "
            f"{fmt(best_val, 4)} | {fmt(full_val, 4)} | {status} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="latex3/table_inputs")
    parser.add_argument("--output-dir", default="latex3/table_outputs")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Only build the named manifest stem, e.g. train_ablation_rows. Can be repeated.",
    )
    args = parser.parse_args()

    input_dir = REPO_ROOT / args.input_dir
    output_dir = REPO_ROOT / args.output_dir
    manifests = sorted(input_dir.glob("*.csv"))
    if args.only:
        requested = {name[:-4] if name.endswith(".csv") else name for name in args.only}
        manifests = [m for m in manifests if m.stem in requested]
        missing = requested - {m.stem for m in manifests}
        if missing:
            raise SystemExit(f"Requested manifest(s) not found: {', '.join(sorted(missing))}")
    if not manifests:
        raise SystemExit(f"No manifests found in {input_dir}")

    for manifest in manifests:
        if manifest.name == "pair_audit_rows.csv":
            rows = compute_pair_manifest(manifest)
            fields = [
                "table",
                "row_id",
                "display",
                "checkpoint",
                "n_pair_sources",
                "n_pairs_total",
                "n_pairs_per_source",
                "source_summary_csv",
            ]
            for metric in PAIR_METRICS:
                fields += [metric, f"{metric}_n", f"{metric}_mean", f"{metric}_std", f"{metric}_var", f"{metric}_se"]
        else:
            rows = compute_episode_manifest(manifest)
            fields = ["table", "row_id", "display", "checkpoint", "n_source_csvs", "source_episode_csv"]
            output_metrics = EPISODE_METRICS
            if manifest.stem == "diag_rows":
                output_metrics = [
                    "final_gate_delta_c_mean",
                    "final_gate_accept_rate",
                    "final_gate_reject_proxy_rate",
                    "inference_latency_mean",
                ]
            for metric in output_metrics:
                fields += [f"{metric}_n", f"{metric}_mean", f"{metric}_std", f"{metric}_var", f"{metric}_se"]
        write_rows(output_dir / f"{manifest.stem}_computed.csv", rows, fields)
        (output_dir / f"{manifest.stem}_tex_rows.tex").write_text(tex_for(manifest.stem, rows), encoding="utf-8")
        (output_dir / f"{manifest.stem}_trend_audit.md").write_text(trend_audit(manifest.stem, rows), encoding="utf-8")
        print(f"wrote {manifest.stem}: {len(rows)} rows")


if __name__ == "__main__":
    main()
