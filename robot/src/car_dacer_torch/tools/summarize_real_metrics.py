#!/usr/bin/env python3
"""Summarize real-car DOVE train/eval CSV logs without pandas."""

import argparse
import csv
import glob
import math
import os
from typing import Dict, Iterable, List, Optional


def _float(row: Dict[str, str], name: str) -> Optional[float]:
    value = row.get(name, "")
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except Exception:
        return None
    if not math.isfinite(x):
        return None
    return x


def _values(rows: Iterable[Dict[str, str]], name: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        x = _float(row, name)
        if x is not None:
            out.append(x)
    return out


def _mean(xs: List[float]) -> Optional[float]:
    if not xs:
        return None
    return sum(xs) / float(len(xs))


def _pctl(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * p / 100.0
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - k) + ys[hi] * (k - lo)


def _rate(xs: List[float], threshold: float = 0.5) -> Optional[float]:
    if not xs:
        return None
    return sum(1 for x in xs if x > threshold) / float(len(xs))


def _fmt(value: Optional[float], scale: float = 1.0, suffix: str = "") -> str:
    if value is None:
        return "NA"
    return f"{value * scale:.4f}{suffix}"


def _expand_inputs(patterns: List[str]) -> List[str]:
    files: List[str] = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
    return sorted(set(files))


def _read_rows(files: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for path in files:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["_source_file"] = path
                rows.append(row)
    return rows


def _filtered_rows(rows: List[Dict[str, str]], phase: Optional[int]) -> List[Dict[str, str]]:
    if phase is None:
        return rows
    out: List[Dict[str, str]] = []
    for row in rows:
        x = _float(row, "phase")
        if x is not None and int(x) == int(phase):
            out.append(row)
    return out


def _print_metric(name: str, value: str) -> None:
    print(f"{name:<34} {value}")


def summarize(rows: List[Dict[str, str]], *, title: str) -> None:
    print(f"\n== {title} ==")
    _print_metric("rows", str(len(rows)))
    if not rows:
        return

    for field, label, scale, suffix in [
        ("process_time", "process_time mean", 1000.0, " ms"),
        ("process_time", "process_time p95", 1000.0, " ms"),
        ("infer_time", "infer_time mean", 1000.0, " ms"),
        ("infer_time", "infer_time p95", 1000.0, " ms"),
        ("loop_period", "loop_period mean", 1000.0, " ms"),
        ("loop_period", "loop_period p95", 1000.0, " ms"),
        ("sensor_age", "sensor_age mean", 1000.0, " ms"),
        ("sensor_age", "sensor_age p95", 1000.0, " ms"),
    ]:
        xs = _values(rows, field)
        value = _mean(xs) if label.endswith("mean") else _pctl(xs, 95.0)
        _print_metric(label, _fmt(value, scale, suffix))

    _print_metric("deadline_miss_rate", _fmt(_mean(_values(rows, "deadline_miss")), 100.0, "%"))
    _print_metric("intervention_rate", _fmt(_mean(_values(rows, "intervention")), 100.0, "%"))
    _print_metric("safety_override_rate", _fmt(_mean(_values(rows, "safety_override")), 100.0, "%"))
    _print_metric("gate_accept_rate", _fmt(_mean(_values(rows, "gate_accepted")), 100.0, "%"))
    _print_metric("gate_fallback_rate", _fmt(_mean(_values(rows, "gate_fallback")), 100.0, "%"))
    _print_metric("gate_action_shift_l2 mean", _fmt(_mean(_values(rows, "gate_action_shift_l2"))))
    _print_metric("gate_proxy_gain mean", _fmt(_mean(_values(rows, "gate_proxy_gain"))))
    _print_metric("gate_delta_c mean", _fmt(_mean(_values(rows, "gate_delta_c"))))
    _print_metric("gate_unc_guided mean", _fmt(_mean(_values(rows, "gate_unc_guided"))))
    _print_metric("guidance_q_uncertainty mean", _fmt(_mean(_values(rows, "guidance_q_uncertainty"))))
    _print_metric("avg_speed", _fmt(_mean(_values(rows, "carspeed")), 1.0, " m/s"))
    _print_metric("abs_lateral_error mean", _fmt(_mean([abs(x) for x in _values(rows, "error_distance")]), 1.0, " m"))
    _print_metric("abs_yaw_error mean", _fmt(_mean([abs(x) for x in _values(rows, "error_yaw")])))
    _print_metric("abs_jerk mean", _fmt(_mean([abs(x) for x in _values(rows, "speed_jerk")])))

    accepted = [r for r in rows if (_float(r, "gate_accepted") or 0.0) > 0.5]
    fallback = [r for r in rows if (_float(r, "gate_fallback") or 0.0) > 0.5]
    _print_metric("accepted_abs_lat_err mean", _fmt(_mean([abs(x) for x in _values(accepted, "error_distance")]), 1.0, " m"))
    _print_metric("fallback_abs_lat_err mean", _fmt(_mean([abs(x) for x in _values(fallback, "error_distance")]), 1.0, " m"))
    _print_metric("accepted_abs_jerk mean", _fmt(_mean([abs(x) for x in _values(accepted, "speed_jerk")])))
    _print_metric("fallback_abs_jerk mean", _fmt(_mean([abs(x) for x in _values(fallback, "speed_jerk")])))


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize DOVE real-car CSV logs.")
    parser.add_argument("csv", nargs="+", help="CSV path(s) or glob(s), for example '/path/to/eval_*.csv'")
    parser.add_argument("--phase", type=int, default=None, help="Only summarize rows from one training phase, e.g. --phase 3")
    args = parser.parse_args()

    files = _expand_inputs(args.csv)
    if not files:
        raise SystemExit("no CSV files matched")
    rows = _read_rows(files)
    rows = _filtered_rows(rows, args.phase)

    print("files:")
    for path in files:
        print(f"  {path}")
    title = "all rows" if args.phase is None else f"phase {args.phase}"
    summarize(rows, title=title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
