#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508"
TRAIN_TB="${ROOT}/logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0/tb/events.out.tfevents.1779417685.aaa-MS-7E44.31580.0"

# TensorBoard event parsing is CPU/I/O bound. CUDA is exported so this can run
# under the same GPU-oriented environment wrapper, but GPU will not materially
# accelerate event-file decoding.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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

python - "$TRAIN_TB" <<'PY'
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

event_file = Path(sys.argv[1])
if not event_file.exists():
    raise SystemExit(f"missing event file: {event_file}")

# These are the same four checkpoints the paper/eval matrix uses.
# The calculation below uses the first step range in the policy name, i.e.
# policy-12000-23998 means episode records whose TensorBoard step is in
# [12000, 23998]. This matches the earlier 23.18% calculation style:
# sum(takeover_rate_i * length_i) / sum(length_i).
CHECKPOINT_WINDOWS = {
    "policy-12000-23998": (12000, 23998),
    "policy-13000-25998": (13000, 25998),
    "policy-14000-27998": (14000, 27998),
    "policy-32000-63998": (32000, 63998),
}

TARGET_TAGS = {
    "episode/takeover_rate",
    "episode/length",
    "stage2_metrics/episode/takeover_rate",
    "stage2_metrics/episode/length",
}


def scalar_value(summary_value):
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


by_tag = defaultdict(list)
event_count = 0
for event in EventFileLoader(str(event_file)).Load():
    event_count += 1
    for value in event.summary.value:
        if value.tag in TARGET_TAGS:
            by_tag[value.tag].append((int(event.step), scalar_value(value)))


def paired_rows(rate_tag, length_tag):
    by_step = defaultdict(dict)
    for step, value in by_tag[rate_tag]:
        by_step[step][rate_tag] = value
    for step, value in by_tag[length_tag]:
        by_step[step][length_tag] = value
    return [
        (step, item[rate_tag], item[length_tag])
        for step, item in sorted(by_step.items())
        if rate_tag in item and length_tag in item
    ]


episode_rows = paired_rows("episode/takeover_rate", "episode/length")
stage2_rows = paired_rows("stage2_metrics/episode/takeover_rate", "stage2_metrics/episode/length")

if not episode_rows and not stage2_rows:
    raise SystemExit("no paired takeover_rate/length episode records found")


def summarize(rows, start=None, end=None):
    if start is not None and end is not None:
        rows = [(s, r, l) for s, r, l in rows if start <= s <= end]
    if not rows:
        return None
    takeover_sec = sum(rate * length for _step, rate, length in rows)
    total_sec = sum(length for _step, _rate, length in rows)
    ratios = [rate for _step, rate, _length in rows]
    return {
        "n_episode_points": len(rows),
        "step_first": rows[0][0],
        "step_last": rows[-1][0],
        "takeover_sec": takeover_sec,
        "total_sec": total_sec,
        "ratio": takeover_sec / total_sec if total_sec else math.nan,
        "mean_episode_rate": statistics.mean(ratios),
        "std_episode_rate": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
    }


print("train_tensorboard_episode_takeover")
print(f"event_file = {event_file}")
print(f"event_count = {event_count}")
print()

all_summary = summarize(episode_rows)
print("overall_same_as_previous_23pct_style")
print("source,n_episode_points,step_first,step_last,takeover_sec,total_sec,ratio_pct,mean_episode_rate_pct,std_episode_rate_pct")
print(
    "episode,"
    f"{all_summary['n_episode_points']},"
    f"{all_summary['step_first']},"
    f"{all_summary['step_last']},"
    f"{all_summary['takeover_sec']:.6f},"
    f"{all_summary['total_sec']:.6f},"
    f"{all_summary['ratio'] * 100:.6f},"
    f"{all_summary['mean_episode_rate'] * 100:.6f},"
    f"{all_summary['std_episode_rate'] * 100:.6f}"
)
if stage2_rows:
    stage2_summary = summarize(stage2_rows)
    print(
        "stage2_episode,"
        f"{stage2_summary['n_episode_points']},"
        f"{stage2_summary['step_first']},"
        f"{stage2_summary['step_last']},"
        f"{stage2_summary['takeover_sec']:.6f},"
        f"{stage2_summary['total_sec']:.6f},"
        f"{stage2_summary['ratio'] * 100:.6f},"
        f"{stage2_summary['mean_episode_rate'] * 100:.6f},"
        f"{stage2_summary['std_episode_rate'] * 100:.6f}"
    )

print()
print("fixed_checkpoint_windows_using_episode_takeover_rate")
print("policy,window_start,window_end,n_episode_points,step_first,step_last,takeover_sec,total_sec,ratio_pct,mean_episode_rate_pct,std_episode_rate_pct")
window_summaries = []
for policy, (start, end) in CHECKPOINT_WINDOWS.items():
    summary = summarize(episode_rows, start, end)
    if summary is None:
        print(f"{policy},{start},{end},0,NA,NA,NA,NA,NA,NA,NA")
        continue
    window_summaries.append((policy, summary))
    print(
        f"{policy},"
        f"{start},"
        f"{end},"
        f"{summary['n_episode_points']},"
        f"{summary['step_first']},"
        f"{summary['step_last']},"
        f"{summary['takeover_sec']:.6f},"
        f"{summary['total_sec']:.6f},"
        f"{summary['ratio'] * 100:.6f},"
        f"{summary['mean_episode_rate'] * 100:.6f},"
        f"{summary['std_episode_rate'] * 100:.6f}"
    )

if window_summaries:
    ratios = [item[1]["ratio"] for item in window_summaries]
    takeover_secs = [item[1]["takeover_sec"] for item in window_summaries]
    total_secs = [item[1]["total_sec"] for item in window_summaries]
    pooled_takeover = sum(takeover_secs)
    pooled_total = sum(total_secs)
    print()
    print("summary_across_fixed_windows")
    print(f"n_windows = {len(window_summaries)}")
    print(f"mean_ratio = {statistics.mean(ratios):.8f} ({statistics.mean(ratios) * 100:.6f}%)")
    print(f"std_ratio = {(statistics.stdev(ratios) if len(ratios) > 1 else 0.0):.8f} ({(statistics.stdev(ratios) if len(ratios) > 1 else 0.0) * 100:.6f}%)")
    print(f"pooled_takeover_sec = {pooled_takeover:.6f}")
    print(f"pooled_total_sec = {pooled_total:.6f}")
    print(f"pooled_ratio = {(pooled_takeover / pooled_total if pooled_total else math.nan):.8f} ({(pooled_takeover / pooled_total * 100 if pooled_total else math.nan):.6f}%)")
PY
