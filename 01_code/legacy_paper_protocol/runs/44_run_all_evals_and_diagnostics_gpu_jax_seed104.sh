#!/usr/bin/env bash
# One-click GPU/JAX runner: closed-loop eval matrix + intervention-pair diagnostics.
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
export TOPK="${TOPK:-8}"

# Diagnostic defaults.  Keep DIAG_ALL_POLICIES=0 by default because PairSat/V_E/IAR
# are needed for the selected method row; full-policy diagnostics can be very slow.
export RUN_PAIR_DIAG="${RUN_PAIR_DIAG:-1}"
export DIAG_ALL_POLICIES="${DIAG_ALL_POLICIES:-0}"
export MAX_PAIRS="${MAX_PAIRS:-512}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export ENERGY_SAMPLES="${ENERGY_SAMPLES:-4}"
export ER_MARGIN="${ER_MARGIN:-0.05}"
export POSITIVE_ACTION="${POSITIVE_ACTION:-behavior}"

TS="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG_DIR="logs/99_all_eval_and_pair_diag_seed${MAPS}_${TS}"
mkdir -p "$MASTER_LOG_DIR"

echo "[start] $(date)"
echo "[cwd] $(pwd)"
echo "[python] $(which python)"
echo "[gpu] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[maps] $MAPS"
echo "[episodes] $NUM_EPISODES"
echo "[run_ids] $RUN_IDS"
echo "[run_topk] $RUN_TOPK"
echo "[run_pair_diag] $RUN_PAIR_DIAG"
echo "[master_log_dir] $MASTER_LOG_DIR"

nvidia-smi
python - <<'PY'
import torch
print('[torch.cuda]', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[torch.gpu]', torch.cuda.get_device_name(0))
import jax
print('[jax.backend]', jax.default_backend())
print('[jax.devices]', jax.devices())
PY

chmod +x runs/40_eval_all_policies_all_modes_seed104.sh || true
chmod +x runs/41_eval_training_runs_all_policies_seed104.sh || true
chmod +x runs/42_run_all_evals_gpu_jax_seed104.sh || true
chmod +x runs/43_run_intervention_pair_diagnostics_seed104.sh || true

# 1) Closed-loop all-policy eval matrix and episode-level mean/std summaries.
bash runs/42_run_all_evals_gpu_jax_seed104.sh 2>&1 | tee "$MASTER_LOG_DIR/42_closed_loop_eval_matrix.log"

# 2) Intervention-pair diagnostics for PairSat/V_E/IAR.
if [[ "$RUN_PAIR_DIAG" == "1" ]]; then
  for rid in $RUN_IDS; do
    echo
    echo "============================================================"
    echo "[pair_diag] RUN_ID=$rid"
    echo "============================================================"
    RUN_ID="$rid" MAPS="$DIAGNOSTIC_MAPS" bash runs/43_run_intervention_pair_diagnostics_seed104.sh \
      2>&1 | tee "$MASTER_LOG_DIR/43_pair_diag_run${rid}.log" || {
        echo "[warn] pair diagnostics failed for run $rid; continuing because some baselines may lack valid intervention pairs/policies" >&2
      }
  done
fi

find logs -type f \( -name '*.csv' -o -name '*.md' -o -name 'console.log' \) | sort > "$MASTER_LOG_DIR/result_files.txt"
find logs -maxdepth 1 -type d \( -name '40_eval_all_policies_all_modes_*' -o -name '43_intervention_pair_diag_*' -o -name '99_*' \) | sort > "$MASTER_LOG_DIR/result_dirs.txt"

echo "[done] $(date)"
echo "[result files] $MASTER_LOG_DIR/result_files.txt"
echo "[result dirs] $MASTER_LOG_DIR/result_dirs.txt"
