#!/usr/bin/env bash
# One-click GPU/JAX runner for the exhaustive script-driven matrix:
# every selected training run x every policy x every single-policy eval script in runs/.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source /home/aaa/miniconda3/etc/profile.d/conda.sh
conda activate dacer_pvp

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
export PYTHONUNBUFFERED=1

export MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
export NUM_EPISODES="${NUM_EPISODES:-2}"
DIAGNOSTIC_MAPS="${DIAGNOSTIC_MAPS:-104}"
export RUN_IDS="${RUN_IDS:-10 11 12 13}"
export ALLOW_MISSING="${ALLOW_MISSING:-1}"
export RUN_TOPK="${RUN_TOPK:-0}"
export RUN_ALL_MAIN="${RUN_ALL_MAIN:-0}"
export TOPK="${TOPK:-8}"
export MAX_POLICIES="${MAX_POLICIES:-0}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# Pair diagnostics are separate because PairSat/V_E/IAR require intervention pairs.
export RUN_PAIR_DIAG="${RUN_PAIR_DIAG:-0}"
export DIAG_ALL_POLICIES="${DIAG_ALL_POLICIES:-0}"
export MAX_PAIRS="${MAX_PAIRS:-512}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export ENERGY_SAMPLES="${ENERGY_SAMPLES:-1}"
export ER_MARGIN="${ER_MARGIN:-0.05}"
export POSITIVE_ACTION="${POSITIVE_ACTION:-behavior}"
export STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-20}"

TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
MASTER_LOG_DIR="logs/99_every_eval_script_and_pair_diag_seed${MAP_TAG}_${TS}"
mkdir -p "$MASTER_LOG_DIR"

cat > "$MASTER_LOG_DIR/manifest.txt" <<EOF
maps=$MAPS
num_episodes=$NUM_EPISODES
run_ids=$RUN_IDS
allow_missing=$ALLOW_MISSING
run_topk=$RUN_TOPK
run_all_main=$RUN_ALL_MAIN
topk=$TOPK
max_policies=$MAX_POLICIES
continue_on_error=$CONTINUE_ON_ERROR
run_pair_diag=$RUN_PAIR_DIAG
diag_all_policies=$DIAG_ALL_POLICIES
max_pairs=$MAX_PAIRS
batch_size=$BATCH_SIZE
energy_samples=$ENERGY_SAMPLES
er_margin=$ER_MARGIN
positive_action=$POSITIVE_ACTION
step_diagnostics_stride=$STEP_DIAGNOSTICS_STRIDE
cuda_visible_devices=$CUDA_VISIBLE_DEVICES
jax_platforms=$JAX_PLATFORMS
python=$(which python)
conda_env=${CONDA_DEFAULT_ENV:-}
master_log_dir=$MASTER_LOG_DIR
EOF
cat "$MASTER_LOG_DIR/manifest.txt"

nvidia-smi | tee "$MASTER_LOG_DIR/nvidia-smi.txt"
python - <<'PY' | tee "$MASTER_LOG_DIR/gpu_jax_check.txt"
import torch
print('[torch.cuda]', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[torch.gpu]', torch.cuda.get_device_name(0))
import jax
print('[jax.backend]', jax.default_backend())
print('[jax.devices]', jax.devices())
PY

chmod +x runs/20_eval_vanilla_policy_seed104.sh || true
chmod +x runs/21_eval_upvgds_no_gate_policy_seed104.sh || true
chmod +x runs/22_eval_upvgds_final_gate_full_policy_seed104.sh || true
chmod +x runs/23_eval_final_gate_no_disagreement_policy_seed104.sh || true
chmod +x runs/24_eval_final_gate_proxy_only_policy_seed104.sh || true
chmod +x runs/25_eval_final_gate_prior_only_policy_seed104.sh || true
chmod +x runs/26_eval_topk_rerank_policy_seed104.sh || true
chmod +x runs/43_run_intervention_pair_diagnostics_seed104.sh || true
chmod +x runs/45_eval_every_eval_script_all_policies_seed104.sh || true
chmod +x runs/46_eval_training_runs_every_eval_script_all_policies_seed104.sh || true

# 1) Exhaustive closed-loop matrix using the actual runs/*_eval_*_policy_seed104.sh files.
MATRIX_ROOT="$MASTER_LOG_DIR/closed_loop_matrix"
MASTER_ROOT="$MATRIX_ROOT" bash runs/46_eval_training_runs_every_eval_script_all_policies_seed104.sh \
  2>&1 | tee "$MASTER_LOG_DIR/46_every_eval_script_matrix.log"

# 2) Intervention-pair diagnostics for PairSat/V_E/IAR.
if [[ "$RUN_PAIR_DIAG" == "1" ]]; then
  for rid in $RUN_IDS; do
    echo
    echo "============================================================"
    echo "[pair_diag] RUN_ID=$rid"
    echo "============================================================"
    RUN_ID="$rid" MAPS="$DIAGNOSTIC_MAPS" OUT_DIR="$MASTER_LOG_DIR/pair_diag_run${rid}" \
      bash runs/43_run_intervention_pair_diagnostics_seed104.sh \
      2>&1 | tee "$MASTER_LOG_DIR/43_pair_diag_run${rid}.log" || {
        echo "[warn] pair diagnostics failed for run $rid; continuing because some runs may lack intervention pairs/policies" >&2
      }
  done
fi

find "$MASTER_LOG_DIR" -type f \( -name '*.csv' -o -name '*.md' -o -name 'console.log' -o -name '*.txt' -o -name '*.log' \) | sort > "$MASTER_LOG_DIR/result_files.txt"

echo "[done] $(date)"
echo "[master_log_dir] $MASTER_LOG_DIR"
echo "[closed_loop_matrix] $MATRIX_ROOT"
echo "[result_files] $MASTER_LOG_DIR/result_files.txt"
