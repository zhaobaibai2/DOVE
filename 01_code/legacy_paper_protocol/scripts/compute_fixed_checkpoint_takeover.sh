#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508"
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
cd "$ROOT"

python - <<'PY'
import math
import statistics
from pathlib import Path

from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

EVENTS = {
    "policy-12000-23998": "logs/50_all_useful_evals_gpu_seed104_20260525_205921/closed_loop_matrix/12_train_doveer_er_seed104_50000_20260522_104112_pvp_dacer_20260522_104125_s0/upvgds_final_gate_full/policy-12000-23998/tb/events.out.tfevents.1779823963.aaa-MS-7E44.369603.0",
    "policy-13000-25998": "logs/50_all_useful_evals_gpu_seed104_20260525_205921/closed_loop_matrix/12_train_doveer_er_seed104_50000_20260522_104112_pvp_dacer_20260522_104125_s0/upvgds_final_gate_full/policy-13000-25998/tb/events.out.tfevents.1779824289.aaa-MS-7E44.369889.0",
    "policy-14000-27998": "logs/50_all_useful_evals_gpu_seed104_20260525_205921/closed_loop_matrix/12_train_doveer_er_seed104_50000_20260522_104112_pvp_dacer_20260522_104125_s0/upvgds_final_gate_full/policy-14000-27998/tb/events.out.tfevents.1779824592.aaa-MS-7E44.370157.0",
    "policy-32000-63998": "logs/50_all_useful_evals_gpu_seed104_20260525_205921/closed_loop_matrix/12_train_doveer_er_seed104_50000_20260522_104112_pvp_dacer_20260522_104125_s0/upvgds_final_gate_full/policy-32000-63998/tb/events.out.tfevents.1779829173.aaa-MS-7E44.374971.0",
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


def read_scalars(path):
    scalars = {}
    for event in EventFileLoader(str(path)).Load():
        for value in event.summary.value:
            scalars[value.tag] = scalar_value(value)
    return scalars


rows = []
for policy, event_path in EVENTS.items():
    path = Path(event_path)
    if not path.exists():
        raise SystemExit(f"missing event file: {path}")

    scalars = read_scalars(path)
    prefix = f"eval/{policy}.pkl"
    avg_prefix = f"{prefix}/avg"
    map_prefix = f"{prefix}/map_104"

    takeover_rate = scalars[f"{avg_prefix}/takeover_rate"]
    intervention_count = scalars[f"{avg_prefix}/intervention_count"]
    takeover_cost = scalars[f"{avg_prefix}/takeover_cost"]
    length = scalars[f"{map_prefix}/length"]
    num_episodes = scalars.get("eval_config/num_episodes", 100.0)

    total_sec = length * num_episodes
    takeover_sec = takeover_rate * total_sec
    rows.append(
        {
            "policy": policy,
            "takeover_rate": takeover_rate,
            "takeover_sec": takeover_sec,
            "total_sec": total_sec,
            "avg_length": length,
            "num_episodes": num_episodes,
            "intervention_count": intervention_count,
            "takeover_cost": takeover_cost,
        }
    )

print("fixed_checkpoint_takeover_from_tensorboard")
print("policy,takeover_rate_pct,takeover_sec,total_sec,avg_length,num_episodes,intervention_count,takeover_cost")
for row in rows:
    print(
        f"{row['policy']},"
        f"{row['takeover_rate'] * 100:.6f},"
        f"{row['takeover_sec']:.6f},"
        f"{row['total_sec']:.6f},"
        f"{row['avg_length']:.6f},"
        f"{row['num_episodes']:.0f},"
        f"{row['intervention_count']:.6f},"
        f"{row['takeover_cost']:.6f}"
    )

rates = [row["takeover_rate"] for row in rows]
takeover_secs = [row["takeover_sec"] for row in rows]
total_secs = [row["total_sec"] for row in rows]
pooled_takeover = sum(takeover_secs)
pooled_total = sum(total_secs)

print()
print("summary")
print(f"n_checkpoints = {len(rows)}")
print(f"mean_takeover_rate = {statistics.mean(rates):.8f} ({statistics.mean(rates) * 100:.6f}%)")
print(f"std_takeover_rate = {(statistics.stdev(rates) if len(rates) > 1 else 0.0):.8f} ({(statistics.stdev(rates) if len(rates) > 1 else 0.0) * 100:.6f}%)")
print(f"mean_takeover_sec = {statistics.mean(takeover_secs):.6f}")
print(f"std_takeover_sec = {(statistics.stdev(takeover_secs) if len(takeover_secs) > 1 else 0.0):.6f}")
print(f"mean_total_sec = {statistics.mean(total_secs):.6f}")
print(f"std_total_sec = {(statistics.stdev(total_secs) if len(total_secs) > 1 else 0.0):.6f}")
print(f"pooled_takeover_sec = {pooled_takeover:.6f}")
print(f"pooled_total_sec = {pooled_total:.6f}")
print(f"pooled_takeover_rate = {(pooled_takeover / pooled_total if pooled_total else math.nan):.8f} ({(pooled_takeover / pooled_total * 100 if pooled_total else math.nan):.6f}%)")
PY
