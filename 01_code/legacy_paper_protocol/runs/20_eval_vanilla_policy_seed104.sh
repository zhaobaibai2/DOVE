#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

POLICY="${POLICY:-}"
MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-20}"
if [[ -z "$POLICY" ]]; then
  echo "ERROR: set POLICY=/path/to/policy.pkl"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_PARENT="${OUT_PARENT:-${REPO_DIR}/logs}"
RUN_NAME="20_eval_vanilla_seed${MAPS// /_}_${TS}"
OUT_DIR="${OUT_DIR:-${OUT_PARENT}/${RUN_NAME}}"
mkdir -p "$OUT_DIR"
POLICY_DIR="$(cd "$(dirname "$POLICY")" && pwd)"
POLICY_NAME="$(basename "$POLICY")"

echo "[run] $RUN_NAME"
echo "[meaning] DOVE actor without test-time critic guidance; 20 held-out episodes total by default."
echo "[policy] $POLICY"
echo "[out] $OUT_DIR"
echo "[env] conda=$CONDA_DEFAULT_ENV python=$(which python)"
check_nvidia_gpu

python scripts/eval_pvp_policies_fixed_v3.py \
  --log_dir "$POLICY_DIR" \
  --policy "$POLICY_NAME" \
  --output_csv "$OUT_DIR/eval_vanilla.csv" \
  --episode_output_csv "$OUT_DIR/episodes_vanilla.csv" \
  --maps $MAPS \
  --num_episodes "$NUM_EPISODES" \
  --step_diagnostics_stride "$STEP_DIAGNOSTICS_STRIDE" \
  --no_use_render \
  2>&1 | tee "$OUT_DIR/console.log"
