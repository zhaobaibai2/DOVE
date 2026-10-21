#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

RUN_DIR="${RUN_DIR:-}"
MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
TOPK="${TOPK:-8}"
RUN_TOPK="${RUN_TOPK:-0}"
RUN_LABEL="${RUN_LABEL:-custom}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find logs/12_train_doveer_er_seed104_50000_* -type d -name 'pvp_dacer_*_s0' | sort | tail -n 1)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: cannot find policy checkpoint directory. Set RUN_DIR=/path/to/pvp_dacer_xxx_s0"
  exit 2
fi

POLICY_COUNT="$(find "$RUN_DIR" -maxdepth 1 \( -name 'policy-*.pkl' -o -name 'policy_*.pkl' -o -name 'final_policy_*.pkl' -o -name 'stage1b_offline_bc_policy.pkl' \) ! -name '*.meta.pkl' | wc -l | tr -d ' ')"
TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
OUT_ROOT="${REPO_DIR}/logs/40_eval_all_policies_all_modes_${RUN_LABEL}_seed${MAP_TAG}_${TS}"
mkdir -p "$OUT_ROOT"

cat > "$OUT_ROOT/manifest.txt" <<EOF
run_label=$RUN_LABEL
run_dir=$RUN_DIR
maps=$MAPS
num_episodes=$NUM_EPISODES
policy_count=$POLICY_COUNT
run_topk=$RUN_TOPK
topk=$TOPK
conda_env=${CONDA_DEFAULT_ENV:-}
python=$(which python)
EOF

cat "$OUT_ROOT/manifest.txt"
check_nvidia_gpu | tee "$OUT_ROOT/nvidia-smi.txt"

run_eval() {
  local mode="$1"
  shift
  local mode_dir="$OUT_ROOT/$mode"
  mkdir -p "$mode_dir"
  echo
  echo "========================================================================"
  echo "[mode] $mode"
  echo "[run_dir] $RUN_DIR"
  echo "[out] $mode_dir"
  echo "========================================================================"
  python -u scripts/eval_pvp_policies_fixed_v3.py \
    --log_dir "$RUN_DIR" \
    --output_csv "$mode_dir/eval_${mode}.csv" \
    --episode_output_csv "$mode_dir/episodes_${mode}.csv" \
    --tensorboard_dir "$mode_dir/tb" \
    --maps $MAPS \
    --num_episodes "$NUM_EPISODES" \
    --no_use_render \
    "$@" \
    2>&1 | tee "$mode_dir/console.log"
}

run_eval vanilla

run_eval upvgds_no_gate \
  --use_value_guidance \
  --guidance_lambda0 0.5 \
  --guidance_beta_unc 1.0 \
  --guidance_p_decay 1.0 \
  --guidance_grad_clip 1.0 \
  --guidance_mode proxy_value \
  --guidance_q_agg conservative \
  --guidance_injection clean_x0 \
  --guidance_schedule noise_level \
  --guidance_target x0

run_eval upvgds_final_gate_full \
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
  --final_gate_min_proxy_gain 0.0 \
  --final_gate_max_action_shift 0.2 \
  --final_gate_tau_prior 0.02 \
  --final_gate_tau_u 1.0 \
  --final_gate_project_guidance 1 \
  --final_gate_energy_samples 4

run_eval final_gate_no_disagreement \
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
  --final_gate_min_proxy_gain 0.0 \
  --final_gate_max_action_shift 0.2 \
  --final_gate_tau_prior 0.02 \
  --final_gate_tau_u -1.0 \
  --final_gate_project_guidance 1 \
  --final_gate_energy_samples 4

run_eval final_gate_proxy_only \
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
  --final_gate_mode proxy_only \
  --final_gate_proxy_compare robust \
  --final_gate_min_proxy_gain 0.0 \
  --final_gate_max_action_shift 0.2 \
  --final_gate_tau_u 1.0 \
  --final_gate_project_guidance 1

run_eval final_gate_prior_only \
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
  --final_gate_mode prior_only \
  --final_gate_max_action_shift 0.2 \
  --final_gate_tau_prior 0.02 \
  --final_gate_tau_u 1.0 \
  --final_gate_project_guidance 1 \
  --final_gate_energy_samples 4

if [[ "$RUN_TOPK" == "1" ]]; then
  run_eval topk${TOPK}_rerank \
    --rerank_topk "$TOPK" \
    --critic_objective cost
else
  echo
  echo "[skip] top-k rerank for all policies. Set RUN_TOPK=1 to include it."
  echo "[why] top-k rerank is much slower because it samples/ranks K actions at every step."
fi

python scripts/summarize_eval_matrix.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/all_modes_all_policies_summary.csv" \
  --best-output "$OUT_ROOT/best_policy_by_mode.csv" \
  --markdown-output "$OUT_ROOT/best_policy_by_mode.md" \
  --top-n 20

python scripts/summarize_episode_mean_std.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/table_ready_episode_mean_std.csv" \
  --markdown-output "$OUT_ROOT/table_ready_episode_mean_std.md" \
  --top-n 20

echo

echo "[done] out_root=$OUT_ROOT"
echo "[done] combined_csv=$OUT_ROOT/all_modes_all_policies_summary.csv"
echo "[done] best_csv=$OUT_ROOT/best_policy_by_mode.csv"
echo "[done] best_md=$OUT_ROOT/best_policy_by_mode.md"
echo "[done] table_ready_csv=$OUT_ROOT/table_ready_episode_mean_std.csv"
echo "[done] table_ready_md=$OUT_ROOT/table_ready_episode_mean_std.md"
