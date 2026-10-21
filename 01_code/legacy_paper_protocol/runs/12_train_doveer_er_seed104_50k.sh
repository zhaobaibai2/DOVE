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
TOTAL_STEP="${TOTAL_STEP:-50000}"
STAGE1B_UPDATES="${STAGE1B_UPDATES:-10000}"
if [[ -z "${CONTROL_FLAGS+x}" ]]; then
  CONTROL_FLAGS="--controller steering_wheel --use_render"
fi
CONTROL_FLAGS_ARRAY=()
if [[ -n "$CONTROL_FLAGS" ]]; then
  read -r -a CONTROL_FLAGS_ARRAY <<< "$CONTROL_FLAGS"
fi

if [[ -z "$DEMO_ROOT" ]]; then
  echo "ERROR: set DEMO_ROOT to the Stage1a data directory."
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="12_train_DOVE_trainseed${TRAINING_SEED}_20maps_budget${TOTAL_STEP}_${TS}"
LOG_ROOT="${REPO_DIR}/logs/${RUN_NAME}"
mkdir -p "$LOG_ROOT"

echo "[run] $RUN_NAME"
echo "[meaning] DOVE: Positive-only + EnergyRank + CPCal under the paper protocol."
echo "[demo_root] $DEMO_ROOT"
echo "[log] $LOG_ROOT"
echo "[control_flags] $CONTROL_FLAGS"
echo "[env] conda=$CONDA_DEFAULT_ENV python=$(which python)"
check_nvidia_gpu

python -u scripts/train_pvp_dacer_metadrive_off.py \
  "${CONTROL_FLAGS_ARRAY[@]}" \
  --demo_root "$DEMO_ROOT" \
  --start_seed "$MAP_SEED" \
  --num_scenarios "$NUM_SCENARIOS" \
  --traffic_density "$TRAFFIC_DENSITY" \
  --seed "$TRAINING_SEED" \
  --stage1b_updates "$STAGE1B_UPDATES" \
  --stage1b_lambda_bc 50.0 \
  --total_step "$TOTAL_STEP" \
  --total_interaction_budget 50000 \
  --save_policy_every 1000 \
  --lambda_pv 1.0 \
  --B 2.0 \
  --lambda_bc 12.0 \
  --policy_mode hybrid_dacer \
  --lambda_rl 0.1 \
  --lambda_reg 2.0 \
  --critic_objective cost \
  --use_energy_rank \
  --lambda_er 1.0 \
  --er_margin 0.05 \
  --er_min_action_gap 0.03 \
  --er_positive_action behavior \
  --er_use_primal_dual \
  --er_budget 0.01 \
  --er_dual_lr 0.001 \
  --er_eta_init 1.0 \
  --lambda_pv_constraint 1.0 \
  --pv_constraint_margin 0.1 \
  --stage2_critic_warmup_steps 1000 \
  --stage2_rl_warmup_steps 0 \
  --stage2_rl_ramp_steps 0 \
  --log_dir "$LOG_ROOT" \
  2>&1 | tee "$LOG_ROOT/console.log"

echo "[policy candidates]"
find "$LOG_ROOT" -name "final_policy_*.pkl" -print | sort | tail -n 5
