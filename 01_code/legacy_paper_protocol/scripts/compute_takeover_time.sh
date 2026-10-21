#!/usr/bin/env bash
set -euo pipefail

DEFAULT_ROOT="/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508/logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0/tb"
ROOT="${1:-$DEFAULT_ROOT}"

# Event parsing is CPU/I/O bound, but keep CUDA visible so the same conda
# environment and GPU job wrapper can be used consistently.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [[ -f "$ROOT" ]]; then
  SEARCH_ROOT="$ROOT"
elif [[ -d "$ROOT" ]]; then
  SEARCH_ROOT="$ROOT"
else
  echo "ERROR: path not found: $ROOT" >&2
  exit 1
fi

if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "/opt/conda/etc/profile.d/conda.sh"
else
  echo "ERROR: conda.sh not found" >&2
  exit 1
fi

conda activate dacer_pvp

python - "$SEARCH_ROOT" <<'PY'

import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

root = Path(sys.argv[1]).expanduser().resolve()
event_files = sorted(root.rglob("events.out.tfevents*"))
if root.is_file() and root.name.startswith("events.out.tfevents"):
    event_files = [root]

if not event_files:
    raise SystemExit(f"ERROR: no TensorBoard event files under {root}")

TAKEOVER_TAGS = ("episode/takeover_rate", "stage2_metrics/episode/takeover_rate")
INTERVENTION_TAGS = ("episode/intervention_rate", "stage2_metrics/episode/intervention_rate")
LENGTH_TAGS = ("episode/length", "stage2_metrics/episode/length")
ACTIVE_TAGS = (
    "takeover/active_rate",
    "stage2_metrics/takeover/active_rate",
    "update/takeover/active_rate",
    "stage2_metrics/update/takeover/active_rate",
)
TARGETS = set(TAKEOVER_TAGS + INTERVENTION_TAGS + LENGTH_TAGS + ACTIVE_TAGS)


def scalar_value(summary_value) -> float:
    if summary_value.HasField("tensor"):
        tensor = summary_value.tensor
        if tensor.float_val:
            return float(tensor.float_val[0])
        if tensor.double_val:
            return float(tensor.double_val[0])
        if tensor.int_val:
            return float(tensor.int_val[0])
        if tensor.int64_val:
            return float(tensor.int64_val[0])
    return float(summary_value.simple_value)


def choose_tag(values_by_tag, tags):
    present = [tag for tag in tags if values_by_tag.get(tag)]
    if not present:
        return None
    return max(present, key=lambda tag: len(values_by_tag[tag]))


def integrate_active_rate(points):
    points = sorted(points)
    if len(points) < 2:
        return None
    active_sec = 0.0
    total_sec = 0.0
    for (step0, _wall0, value0), (step1, _wall1, _value1) in zip(points, points[1:]):
        dt = max(0, step1 - step0)
        active_sec += value0 * dt
        total_sec += dt
    if total_sec <= 0:
        return None
    return active_sec, total_sec, active_sec / total_sec


def summarize_event(path):
    values_by_tag = defaultdict(list)
    event_count = 0
    for event in EventFileLoader(str(path)).Load():
        event_count += 1
        for value in event.summary.value:
            if value.tag in TARGETS:
                values_by_tag[value.tag].append((int(event.step), float(event.wall_time), scalar_value(value)))

    length_tag = choose_tag(values_by_tag, LENGTH_TAGS)
    rate_tag = choose_tag(values_by_tag, TAKEOVER_TAGS)
    source = "takeover_rate"

    if rate_tag is None:
        rate_tag = choose_tag(values_by_tag, INTERVENTION_TAGS)
        source = "intervention_rate"

    if rate_tag is not None and length_tag is not None:
        by_step = defaultdict(dict)
        for step, _wall, value in values_by_tag[rate_tag]:
            by_step[step][rate_tag] = value
        for step, _wall, value in values_by_tag[length_tag]:
            by_step[step][length_tag] = value

        pairs = [(step, item[rate_tag], item[length_tag]) for step, item in sorted(by_step.items()) if rate_tag in item and length_tag in item]
        if pairs:
            total_sec = sum(length for _step, _rate, length in pairs)
            takeover_sec = sum(rate * length for _step, rate, length in pairs)
            ratio = takeover_sec / total_sec if total_sec else math.nan
            return {
                "path": path,
                "source": source,
                "rate_tag": rate_tag,
                "length_tag": length_tag,
                "pairs": len(pairs),
                "takeover_sec": takeover_sec,
                "total_sec": total_sec,
                "ratio": ratio,
                "event_count": event_count,
            }

    active_tag = choose_tag(values_by_tag, ACTIVE_TAGS)
    if active_tag is not None:
        integrated = integrate_active_rate(values_by_tag[active_tag])
        if integrated is not None:
            takeover_sec, total_sec, ratio = integrated
            return {
                "path": path,
                "source": "active_rate_step_integral",
                "rate_tag": active_tag,
                "length_tag": "-",
                "pairs": len(values_by_tag[active_tag]),
                "takeover_sec": takeover_sec,
                "total_sec": total_sec,
                "ratio": ratio,
                "event_count": event_count,
            }

    return None


rows = []
for event_file in event_files:
    row = summarize_event(event_file)
    if row is not None:
        rows.append(row)

if not rows:
    raise SystemExit("ERROR: no takeover/intervention metrics found")

print(f"search_root: {root}")
print(f"event_files_with_metrics: {len(rows)} / {len(event_files)}")
print(f"cuda_visible_devices: {__import__('os').environ.get('CUDA_VISIBLE_DEVICES', '')}")
print()
print("per_policy_or_event:")
print("ratio_pct,takeover_sec,total_sec,n_points,source,event_file")
for row in rows:
    print(
        f"{row['ratio'] * 100:.6f},"
        f"{row['takeover_sec']:.6f},"
        f"{row['total_sec']:.6f},"
        f"{row['pairs']},"
        f"{row['source']},"
        f"{row['path']}"
    )

ratios = [float(row["ratio"]) for row in rows]
takeover_secs = [float(row["takeover_sec"]) for row in rows]
total_secs = [float(row["total_sec"]) for row in rows]

mean_ratio = statistics.mean(ratios)
std_ratio = statistics.stdev(ratios) if len(ratios) > 1 else 0.0
mean_takeover = statistics.mean(takeover_secs)
std_takeover = statistics.stdev(takeover_secs) if len(takeover_secs) > 1 else 0.0
mean_total = statistics.mean(total_secs)
std_total = statistics.stdev(total_secs) if len(total_secs) > 1 else 0.0

pooled_takeover = sum(takeover_secs)
pooled_total = sum(total_secs)
pooled_ratio = pooled_takeover / pooled_total if pooled_total else math.nan

print()
print("summary_across_policies_or_events:")
print(f"n = {len(rows)}")
print(f"mean_ratio = {mean_ratio:.8f} ({mean_ratio * 100:.4f}%)")
print(f"std_ratio = {std_ratio:.8f} ({std_ratio * 100:.4f}%)")
print(f"mean_takeover_sec = {mean_takeover:.4f}")
print(f"std_takeover_sec = {std_takeover:.4f}")
print(f"mean_total_sec = {mean_total:.4f}")
print(f"std_total_sec = {std_total:.4f}")
print(f"pooled_takeover_sec = {pooled_takeover:.4f}")
print(f"pooled_total_sec = {pooled_total:.4f}")
print(f"pooled_ratio = {pooled_ratio:.8f} ({pooled_ratio * 100:.4f}%)")
PY
