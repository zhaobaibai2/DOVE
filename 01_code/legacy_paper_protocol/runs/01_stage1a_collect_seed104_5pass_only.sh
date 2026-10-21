#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

TRAINING_SEED="${TRAINING_SEED:-0}"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="01_dove_stage1a_trainseed${TRAINING_SEED}_20maps_5pass_${TS}"
LOG_ROOT="${REPO_DIR}/logs/${RUN_NAME}"

echo "[run] $RUN_NAME"
echo "[log] $LOG_ROOT"
echo "[data] auto: actual training log path/data"
echo "[env] conda=$CONDA_DEFAULT_ENV python=$(which python)"
check_nvidia_gpu

python scripts/train_pvp_dacer_metadrive_off.py \
  --controller steering_wheel \
  --manual_control \
  --use_render \
  --force_demo_collection \
  --start_seed 104 \
  --num_scenarios 20 \
  --traffic_density 0.06 \
  --seed "$TRAINING_SEED" \
  --demo_passes_per_map 5 \
  --stage1b_updates 0 \
  --total_step 0 \
  --save_policy_every 1000 \
  --log_dir "$LOG_ROOT"

DATA_DIR="$(find "$LOG_ROOT" -type f -name "stage1a_merged_demo.pkl" -printf '%h\n' | sort | tail -n 1)"
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="$(find "$LOG_ROOT" -type d -name "recorded_data" -printf '%h\n' | sort | tail -n 1)"
fi
if [[ -z "$DATA_DIR" ]]; then
  DATA_DIR="${LOG_ROOT}/<actual_pvp_dacer_run>/data"
fi

echo
echo "[next] Use demo root:"
echo "TRAINING_SEED=$TRAINING_SEED DEMO_ROOT=\"$DATA_DIR\" bash runs/12_train_doveer_er_seed104_50k.sh"
