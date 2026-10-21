#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

# Edit these when needed.
DEMO_ROOT="${DEMO_ROOT:-}"
MAP_SEED="${MAP_SEED:-104}"
NUM_SCENARIOS="${NUM_SCENARIOS:-20}"
TRAINING_SEED="${TRAINING_SEED:-0}"
TRAFFIC_DENSITY="${TRAFFIC_DENSITY:-0.06}"
STAGE1B_UPDATES="${STAGE1B_UPDATES:-10000}"
STAGE1B_LAMBDA_BC="${STAGE1B_LAMBDA_BC:-50.0}"
CONTROL_FLAGS="${CONTROL_FLAGS:-}"
CONTROL_FLAGS_ARRAY=()
if [[ -n "$CONTROL_FLAGS" ]]; then
  read -r -a CONTROL_FLAGS_ARRAY <<< "$CONTROL_FLAGS"
fi

if [[ -z "$DEMO_ROOT" ]]; then
  echo "ERROR: set DEMO_ROOT to the Stage1a data directory, e.g."
  echo "  DEMO_ROOT=/path/to/logs/01_stage1a.../data bash $0"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="10_positive_only_initialization_trainseed${TRAINING_SEED}_20maps_bc${STAGE1B_UPDATES}_${TS}"
LOG_ROOT="${REPO_DIR}/logs/${RUN_NAME}"
mkdir -p "$LOG_ROOT"

echo "[run] $RUN_NAME"
echo "[meaning] Positive-only diffusion initialization checkpoint only; use the paper task generator for the matched-budget reference."
echo "[demo_root] $DEMO_ROOT"
echo "[log] $LOG_ROOT"
echo "[control_flags] ${CONTROL_FLAGS:-<none>}"
echo "[env] conda=$CONDA_DEFAULT_ENV python=$(which python)"
check_nvidia_gpu

python scripts/train_pvp_dacer_metadrive_off.py \
  "${CONTROL_FLAGS_ARRAY[@]}" \
  --demo_root "$DEMO_ROOT" \
  --start_seed "$MAP_SEED" \
  --num_scenarios "$NUM_SCENARIOS" \
  --traffic_density "$TRAFFIC_DENSITY" \
  --seed "$TRAINING_SEED" \
  --stage1b_updates "$STAGE1B_UPDATES" \
  --stage1b_lambda_bc "$STAGE1B_LAMBDA_BC" \
  --total_step 0 \
  --save_policy_every 1000 \
  --log_dir "$LOG_ROOT" \
  2>&1 | tee "$LOG_ROOT/console.log"

echo "[policy candidates]"
find "$LOG_ROOT" -name "stage1b_offline_bc_policy.pkl" -print | sort
