#!/usr/bin/env bash
# One-command GPU runner for all useful evals from runs/20 onward.
# Default behavior:
#   - auto-discover every training checkpoint directory under logs/*/pvp_dacer_*_s0
#   - run the script-driven eval definitions 20--26 over every policy checkpoint
#   - use 100 episodes per policy/mode
#   - write complete logs plus Table-III-ready Test Cost summaries
# Notes:
#   40--47 are runner/summary wrappers, not independent eval modes; this script replaces them as the master runner.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
export PYTHONUNBUFFERED=1

export MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
export NUM_EPISODES="${NUM_EPISODES:-2}"
TRAINING_MAP_START="${TRAINING_MAP_START:-104}"
DIAGNOSTIC_MAPS="${DIAGNOSTIC_MAPS:-104}"

# RUN_IDS=auto means: do not limit to 10/11/12/13; discover all training runs with pkl checkpoints.
# You can still override, e.g. RUN_IDS="10 11 12 13" bash runs/50...
export RUN_IDS="${RUN_IDS:-auto}"
export ALLOW_MISSING="${ALLOW_MISSING:-1}"

# Include top-k by default because the user asked to rerun all useful evals.
# Disable with RUN_TOPK=0 if it is too slow.
export RUN_TOPK="${RUN_TOPK:-1}"

# 30_eval_all_main_for_policy_seed104.sh is a duplicate wrapper of 20--26.
# Keep off by default to avoid double-running the same modes. Set RUN_ALL_MAIN=1 to also run it.
export RUN_ALL_MAIN="${RUN_ALL_MAIN:-0}"

export TOPK="${TOPK:-8}"
export MAX_POLICIES="${MAX_POLICIES:-0}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-20}"

# Pair diagnostics are mechanism diagnostics, not closed-loop Table III eval. Enable when needed.
export RUN_PAIR_DIAG="${RUN_PAIR_DIAG:-0}"
export DIAG_ALL_POLICIES="${DIAG_ALL_POLICIES:-0}"
export MAX_PAIRS="${MAX_PAIRS:-512}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export ENERGY_SAMPLES="${ENERGY_SAMPLES:-1}"
export ER_MARGIN="${ER_MARGIN:-0.05}"
export POSITIVE_ACTION="${POSITIVE_ACTION:-behavior}"

TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
MASTER_LOG_DIR="${MASTER_LOG_DIR:-$REPO_DIR/logs/50_all_useful_evals_gpu_seed${MAP_TAG}_${TS}}"
MATRIX_ROOT="$MASTER_LOG_DIR/closed_loop_matrix"
mkdir -p "$MASTER_LOG_DIR" "$MATRIX_ROOT"

chmod +x runs/[2-4][0-9]*.sh runs/_common_env.sh 2>/dev/null || true

find_run_dir_by_id() {
  local run_id="$1"
  case "$run_id" in
    10) find logs/10_train_dbp_only_seed${TRAINING_MAP_START}_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1 ;;
    11) find logs/11_train_pvp_dbp_critic_seed${TRAINING_MAP_START}_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1 ;;
    12) find logs/12_train_doveer_er_seed${TRAINING_MAP_START}_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1 ;;
    13) find logs/13_train_doveer_er_no_pv_constraint_seed${TRAINING_MAP_START}_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1 ;;
    *)  find logs/${run_id}_train_*seed${TRAINING_MAP_START}* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1 ;;
  esac
}

has_policy_checkpoint() {
  local d="$1"
  find "$d" -maxdepth 1 -type f \
    \( -name 'policy-*.pkl' -o -name 'policy_*.pkl' -o -name 'final_policy_*.pkl' -o -name 'stage1b_offline_bc_policy.pkl' \) \
    ! -name '*.meta.pkl' | grep -q .
}

sanitize_label() {
  echo "$1" | sed -E 's#[/ ]+#_#g; s/[^A-Za-z0-9._-]+/_/g; s/^_+//; s/_+$//'
}

RUN_DIR_LIST="$MASTER_LOG_DIR/discovered_training_runs.txt"
: > "$RUN_DIR_LIST"

if [[ "$RUN_IDS" == "auto" || "$RUN_IDS" == "all" ]]; then
  # Auto-discover all completed training/eval checkpoint directories, not just run IDs 10--13.
  while IFS= read -r d; do
    [[ -d "$d" ]] || continue
    if has_policy_checkpoint "$d"; then
      parent="$(basename "$(dirname "$d")")"
      leaf="$(basename "$d")"
      label="$(sanitize_label "${parent}_${leaf}")"
      printf '%s\t%s\n' "$label" "$d" >> "$RUN_DIR_LIST"
    fi
  done < <(find logs -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort -V)
else
  for run_id in $RUN_IDS; do
    d="$(find_run_dir_by_id "$run_id" || true)"
    if [[ -z "$d" || ! -d "$d" ]]; then
      msg="No checkpoint directory found for RUN_ID=$run_id"
      if [[ "$ALLOW_MISSING" == "1" ]]; then
        echo "[skip] $msg"
        continue
      fi
      echo "ERROR: $msg" >&2
      exit 2
    fi
    if has_policy_checkpoint "$d"; then
      printf 'run%s\t%s\n' "$run_id" "$d" >> "$RUN_DIR_LIST"
    else
      echo "[skip] RUN_ID=$run_id has no policy checkpoint: $d" >&2
    fi
  done
fi

RUN_COUNT="$(wc -l < "$RUN_DIR_LIST" | tr -d ' ')"
if [[ "$RUN_COUNT" == "0" ]]; then
  echo "ERROR: no training checkpoint directories with policy .pkl files were found." >&2
  echo "Looked under: $REPO_DIR/logs" >&2
  exit 2
fi

cat > "$MASTER_LOG_DIR/manifest.txt" <<EOF2
maps=$MAPS
num_episodes=$NUM_EPISODES
run_ids=$RUN_IDS
run_count=$RUN_COUNT
allow_missing=$ALLOW_MISSING
run_topk=$RUN_TOPK
run_all_main=$RUN_ALL_MAIN
topk=$TOPK
max_policies=$MAX_POLICIES
continue_on_error=$CONTINUE_ON_ERROR
step_diagnostics_stride=$STEP_DIAGNOSTICS_STRIDE
run_pair_diag=$RUN_PAIR_DIAG
diag_all_policies=$DIAG_ALL_POLICIES
max_pairs=$MAX_PAIRS
batch_size=$BATCH_SIZE
energy_samples=$ENERGY_SAMPLES
er_margin=$ER_MARGIN
positive_action=$POSITIVE_ACTION
cuda_visible_devices=$CUDA_VISIBLE_DEVICES
jax_platforms=$JAX_PLATFORMS
python=$(which python)
conda_env=${CONDA_DEFAULT_ENV:-}
master_log_dir=$MASTER_LOG_DIR
matrix_root=$MATRIX_ROOT
discovered_training_runs=$RUN_DIR_LIST
EOF2
cat "$MASTER_LOG_DIR/manifest.txt"
echo
echo "[discovered training runs]"
cat "$RUN_DIR_LIST"

check_nvidia_gpu | tee "$MASTER_LOG_DIR/nvidia-smi.txt"
python - <<'PYCHECK' | tee "$MASTER_LOG_DIR/gpu_jax_check.txt"
try:
    import torch
    print('[torch.cuda]', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('[torch.gpu]', torch.cuda.get_device_name(0))
except Exception as exc:
    print('[torch.cuda] check failed:', exc)
import jax
print('[jax.backend]', jax.default_backend())
print('[jax.devices]', jax.devices())
PYCHECK

# Main closed-loop matrix: run eval scripts 20--26 over every policy in every discovered training directory.
while IFS=$'\t' read -r run_label run_dir; do
  [[ -n "$run_label" && -n "$run_dir" ]] || continue
  echo
  echo "########################################################################"
  echo "[training-run] $run_label"
  echo "[run_dir] $run_dir"
  echo "[master_root] $MATRIX_ROOT"
  echo "########################################################################"

  RUN_DIR="$run_dir" \
  RUN_LABEL="$run_label" \
  OUT_ROOT="$MATRIX_ROOT/$run_label" \
  MAPS="$MAPS" \
  NUM_EPISODES="$NUM_EPISODES" \
  RUN_TOPK="$RUN_TOPK" \
  RUN_ALL_MAIN="$RUN_ALL_MAIN" \
  TOPK="$TOPK" \
  STEP_DIAGNOSTICS_STRIDE="$STEP_DIAGNOSTICS_STRIDE" \
  MAX_POLICIES="$MAX_POLICIES" \
  CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR" \
  bash runs/45_eval_every_eval_script_all_policies_seed104.sh \
    2>&1 | tee "$MASTER_LOG_DIR/45_${run_label}.log"
done < "$RUN_DIR_LIST"

# Optional mechanism diagnostics for PairSat/V_E/IAR.
if [[ "$RUN_PAIR_DIAG" == "1" ]]; then
  while IFS=$'\t' read -r run_label run_dir; do
    echo
    echo "============================================================"
    echo "[pair_diag] $run_label"
    echo "============================================================"
    RUN_DIR="$run_dir" RUN_LABEL="$run_label" MAPS="$DIAGNOSTIC_MAPS" OUT_DIR="$MASTER_LOG_DIR/pair_diag_${run_label}" \
      bash runs/43_run_intervention_pair_diagnostics_seed104.sh \
      2>&1 | tee "$MASTER_LOG_DIR/43_pair_diag_${run_label}.log" || {
        echo "[warn] pair diagnostics failed for $run_label; continuing" >&2
      }
  done < "$RUN_DIR_LIST"
fi

# Aggregate the full matrix across all discovered runs.
python scripts/summarize_eval_matrix.py \
  --root "$MATRIX_ROOT" \
  --output "$MASTER_LOG_DIR/all_runs_all_eval_scripts_all_policies_summary.csv" \
  --best-output "$MASTER_LOG_DIR/all_runs_best_policy_by_eval_script.csv" \
  --markdown-output "$MASTER_LOG_DIR/all_runs_best_policy_by_eval_script.md" \
  --top-n 50 || echo "[warn] summarize_eval_matrix.py failed" >&2

python scripts/summarize_episode_mean_std.py \
  --root "$MATRIX_ROOT" \
  --output "$MASTER_LOG_DIR/all_runs_table_ready_episode_mean_std.csv" \
  --markdown-output "$MASTER_LOG_DIR/all_runs_table_ready_episode_mean_std.md" \
  --top-n 50 || echo "[warn] summarize_episode_mean_std.py failed" >&2

# Table III cost extraction. Broad column-name matching keeps working if cost is named cost/test_cost/takeover_cost/etc.
python - "$MASTER_LOG_DIR" "$MATRIX_ROOT" <<'PYTABLE'
import csv, math, re, statistics, sys
from pathlib import Path
root = Path(sys.argv[1])
matrix = Path(sys.argv[2])

COST_KEYS = ('test_cost','cost','takeover_cost','total_cost','episode_cost','mean_cost','avg_cost','eval_cost','takeover_rate','takeover')
RETURN_KEYS = ('return','reward','episode_return','mean_return','avg_return')
SUCCESS_KEYS = ('success','succ','success_rate')

def num(x):
    if x is None: return None
    s = str(x).strip()
    if not s or s.lower() in {'nan','none','null','--'}: return None
    try: return float(s)
    except Exception: return None

def pick(row, keys):
    lowers = {k.lower(): k for k in row.keys()}
    for want in keys:
        for lk, orig in lowers.items():
            if want == lk or want in lk:
                v = num(row.get(orig))
                if v is not None:
                    return v, orig
    return None, ''

def mean_std(vals):
    vals=[v for v in vals if v is not None and math.isfinite(v)]
    if not vals: return '', '', 0
    if len(vals)==1: return vals[0], 0.0, 1
    return statistics.fmean(vals), statistics.stdev(vals), len(vals)

rows=[]
for csv_path in sorted(matrix.rglob('eval_*.csv')):
    if csv_path.name.startswith(('executed_', 'all_', 'best_')):
        continue
    try:
        with csv_path.open(newline='', encoding='utf-8') as f:
            data=list(csv.DictReader(f))
    except Exception:
        continue
    if not data:
        continue
    rel_parts=csv_path.relative_to(matrix).parts
    run_label=rel_parts[0] if len(rel_parts) > 0 else ''
    mode=rel_parts[1] if len(rel_parts) > 1 else ''
    policy=rel_parts[2] if len(rel_parts) > 2 else ''
    vals_cost=[]; vals_ret=[]; vals_succ=[]; cost_col=''; ret_col=''; succ_col=''
    for r in data:
        v,c=pick(r,COST_KEYS); vals_cost.append(v); cost_col=cost_col or c
        v,c=pick(r,RETURN_KEYS); vals_ret.append(v); ret_col=ret_col or c
        v,c=pick(r,SUCCESS_KEYS); vals_succ.append(v); succ_col=succ_col or c
    m_cost,s_cost,n_cost=mean_std(vals_cost)
    m_ret,s_ret,n_ret=mean_std(vals_ret)
    m_succ,s_succ,n_succ=mean_std(vals_succ)
    rows.append({
        'run_label': run_label, 'mode': mode, 'policy': policy, 'csv': str(csv_path.relative_to(root)),
        'test_cost_mean': m_cost, 'test_cost_std': s_cost, 'test_cost_n': n_cost, 'test_cost_column': cost_col,
        'return_mean': m_ret, 'return_std': s_ret, 'return_n': n_ret, 'return_column': ret_col,
        'success_mean': m_succ, 'success_std': s_succ, 'success_n': n_succ, 'success_column': succ_col,
    })

def key(r):
    cost = r['test_cost_mean'] if isinstance(r['test_cost_mean'], float) else float('inf')
    ret = r['return_mean'] if isinstance(r['return_mean'], float) else -float('inf')
    succ = r['success_mean'] if isinstance(r['success_mean'], float) else -float('inf')
    return (cost, -succ, -ret, r['run_label'], r['mode'], r['policy'])

rows_sorted=sorted(rows, key=key)
out_csv=root/'table_iii_train_test_costs.csv'
fields=['run_label','mode','policy','test_cost_mean','test_cost_std','test_cost_n','test_cost_column','return_mean','return_std','return_n','return_column','success_mean','success_std','success_n','success_column','csv']
with out_csv.open('w', newline='', encoding='utf-8') as f:
    w=csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows_sorted)

out_md=root/'table_iii_train_test_costs.md'
with out_md.open('w', encoding='utf-8') as f:
    f.write('# Table III Train/Test Cost Candidates\n\n')
    f.write('Train Cost should be copied from the corresponding training run logs. Test Cost is extracted from eval CSV cost-like columns.\n\n')
    f.write('| run | mode | policy | Test Cost mean±std | Return mean±std | Succ. mean±std | source |\n')
    f.write('|---|---|---|---:|---:|---:|---|\n')
    for r in rows_sorted[:200]:
        def fmt(m,s,n):
            if not isinstance(m,float): return ''
            return f'{m:.6g}±{s:.6g} (n={n})'
        f.write(f"| {r['run_label']} | {r['mode']} | {r['policy']} | {fmt(r['test_cost_mean'],r['test_cost_std'],r['test_cost_n'])} | {fmt(r['return_mean'],r['return_std'],r['return_n'])} | {fmt(r['success_mean'],r['success_std'],r['success_n'])} | `{r['csv']}` |\n")
print(f'[table-iii] wrote {out_csv}')
print(f'[table-iii] wrote {out_md}')
PYTABLE

find "$MASTER_LOG_DIR" -type f \( -name '*.csv' -o -name '*.md' -o -name 'console.log' -o -name '*.txt' -o -name '*.log' \) | sort > "$MASTER_LOG_DIR/result_files.txt"

echo "[done] $(date)"
echo "[master_log_dir] $MASTER_LOG_DIR"
echo "[closed_loop_matrix] $MATRIX_ROOT"
echo "[discovered_training_runs] $RUN_DIR_LIST"
echo "[table_iii_cost_csv] $MASTER_LOG_DIR/table_iii_train_test_costs.csv"
echo "[table_iii_cost_md] $MASTER_LOG_DIR/table_iii_train_test_costs.md"
echo "[result_files] $MASTER_LOG_DIR/result_files.txt"
