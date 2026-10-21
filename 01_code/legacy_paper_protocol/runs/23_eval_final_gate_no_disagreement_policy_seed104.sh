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
RUN_NAME="23_eval_final_gate_no_disagreement_seed${MAPS// /_}_${TS}"
OUT_DIR="${OUT_DIR:-${OUT_PARENT}/${RUN_NAME}}"
mkdir -p "$OUT_DIR"
POLICY_DIR="$(cd "$(dirname "$POLICY")" && pwd)"
POLICY_NAME="$(basename "$POLICY")"

echo "[run] $RUN_NAME"
echo "[meaning] Ablation: UPV-GDS + Final Gate but disagreement gate disabled with tau_u=-1."
echo "[policy] $POLICY"
echo "[out] $OUT_DIR"
echo "[env] conda=$CONDA_DEFAULT_ENV python=$(which python)"
check_nvidia_gpu

python scripts/eval_pvp_policies_fixed_v3.py \
  --log_dir "$POLICY_DIR" \
  --policy "$POLICY_NAME" \
  --output_csv "$OUT_DIR/eval_final_gate_no_disagreement.csv" \
  --episode_output_csv "$OUT_DIR/episodes_final_gate_no_disagreement.csv" \
  --maps $MAPS \
  --num_episodes "$NUM_EPISODES" \
  --step_diagnostics_stride "$STEP_DIAGNOSTICS_STRIDE" \
  --no_use_render \
  --use_value_guidance \
  --guidance_lambda0 0.5 \
  --guidance_beta_unc 1.0 \
  --guidance_p_decay 1.0 \
  --guidance_grad_clip 1.0 \
  --guidance_mode proxy_value \
  --guidance_q_agg conservative \
  --guidance_injection clean_x0 \
  --guidance_schedule noise_level \
  --guidance_target x0 \
  --use_final_gate \
  --final_gate_mode proxy_prior \
  --final_gate_proxy_compare robust \
  --final_gate_min_proxy_gain 0.02 \
  --final_gate_max_action_shift 0.2 \
  --final_gate_tau_prior 0.02 \
  --final_gate_tau_u -1.0 \
  --final_gate_project_guidance 1 \
  --final_gate_energy_samples "${FINAL_GATE_ENERGY_SAMPLES:-${ENERGY_SAMPLES:-4}}" \
  2>&1 | tee "$OUT_DIR/console.log"
