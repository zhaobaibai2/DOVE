#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/critic_reliability_matrix_allpolicies_20260604_1603}"
WATCH_LOG="${WATCH_LOG:-${OUTPUT_ROOT}_watchdog.log}"
SUMMARY_DIR="${SUMMARY_DIR:-${OUTPUT_ROOT}/summary}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAPS="${MAPS:-104}"
RELIABILITY_HORIZON="${RELIABILITY_HORIZON:-25}"
STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-1}"
FINAL_GATE_ENERGY_SAMPLES="${FINAL_GATE_ENERGY_SAMPLES:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dacer_pvp}"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda.sh not found: $CONDA_SH" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION

echo "[watchdog] start $(date)" | tee -a "$WATCH_LOG"
echo "[watchdog] output_root=$OUTPUT_ROOT workers=$NUM_WORKERS episodes=$NUM_EPISODES maps=$MAPS" | tee -a "$WATCH_LOG"

while true; do
  if [[ -f "$OUTPUT_ROOT/results.csv" ]]; then
    failed_count="$(python - "$OUTPUT_ROOT/results.csv" <<'PY'
import csv, sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print(sum(1 for r in rows if r.get("status") == "failed"))
PY
)"
    total_count="$(python - "$OUTPUT_ROOT/results.csv" <<'PY'
import csv, sys
with open(sys.argv[1], newline="", encoding="utf-8") as f:
    print(sum(1 for _ in csv.DictReader(f)))
PY
)"
    if [[ "$total_count" == "146" && "$failed_count" == "0" ]]; then
      echo "[watchdog] matrix already complete; aggregating $(date)" | tee -a "$WATCH_LOG"
      break
    fi
  fi

  echo "[watchdog] launch/resume matrix $(date)" | tee -a "$WATCH_LOG"
  set +e
  python scripts/run_critic_reliability_matrix.py \
    --output_root "$OUTPUT_ROOT" \
    --num_workers "$NUM_WORKERS" \
    --num_episodes "$NUM_EPISODES" \
    --maps $MAPS \
    --reliability_horizon "$RELIABILITY_HORIZON" \
    --step_diagnostics_stride "$STEP_DIAGNOSTICS_STRIDE" \
    --final_gate_energy_samples "$FINAL_GATE_ENERGY_SAMPLES" \
    --cuda_visible_devices "$CUDA_VISIBLE_DEVICES" \
    --xla_mem_fraction "$XLA_PYTHON_CLIENT_MEM_FRACTION" \
    >> "$WATCH_LOG" 2>&1
  rc=$?
  set -e

  if [[ "$rc" == "0" ]]; then
    echo "[watchdog] matrix complete $(date)" | tee -a "$WATCH_LOG"
    break
  fi

  echo "[watchdog] matrix exited rc=$rc; sleep ${POLL_SECONDS}s then resume $(date)" | tee -a "$WATCH_LOG"
  sleep "$POLL_SECONDS"
done

mapfile -t step_csvs < <(find "$OUTPUT_ROOT" -mindepth 3 -maxdepth 3 -type f -name 'steps_upvgds_final_gate_full.csv' | sort)
if [[ "${#step_csvs[@]}" -eq 0 ]]; then
  echo "ERROR: no step CSVs found under $OUTPUT_ROOT" | tee -a "$WATCH_LOG"
  exit 1
fi

variants=()
for csv in "${step_csvs[@]}"; do
  rel="${csv#$OUTPUT_ROOT/}"
  variant="${rel%%/*}"
  policy_dir="$(basename "$(dirname "$csv")")"
  variants+=("${variant}_${policy_dir}")
done

mkdir -p "$SUMMARY_DIR"
python scripts/aggregate_critic_reliability.py \
  --step_csv "${step_csvs[@]}" \
  --variant "${variants[@]}" \
  --output_dir "$SUMMARY_DIR" \
  --bins 10 \
  >> "$WATCH_LOG" 2>&1

echo "[watchdog] aggregate complete $(date)" | tee -a "$WATCH_LOG"
