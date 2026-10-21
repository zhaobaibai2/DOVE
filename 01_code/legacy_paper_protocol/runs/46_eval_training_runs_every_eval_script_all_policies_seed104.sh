#!/usr/bin/env bash
# For each selected training run, run every runs/*_eval_*_policy_seed104.sh script over every policy.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
RUN_IDS="${RUN_IDS:-10 11 12 13}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"
RUN_TOPK="${RUN_TOPK:-0}"
RUN_ALL_MAIN="${RUN_ALL_MAIN:-0}"
TOPK="${TOPK:-8}"
MAX_POLICIES="${MAX_POLICIES:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-20}"

TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
MASTER_ROOT="${MASTER_ROOT:-${REPO_DIR}/logs/46_eval_training_runs_every_eval_script_all_policies_seed${MAP_TAG}_${TS}}"
mkdir -p "$MASTER_ROOT"

find_run_dir() {
  local run_id="$1"
  case "$run_id" in
    10)
      find logs/10_train_dbp_only_seed${MAPS}_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    11)
      find logs/11_train_pvp_dbp_critic_seed${MAPS}_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    12)
      find logs/12_train_doveer_er_seed${MAPS}_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    13)
      find logs/13_train_doveer_er_no_pv_constraint_seed${MAPS}_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    *)
      find logs/${run_id}_train_*seed${MAPS}* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
  esac
}

cat > "$MASTER_ROOT/manifest.txt" <<EOF
maps=$MAPS
num_episodes=$NUM_EPISODES
run_ids=$RUN_IDS
allow_missing=$ALLOW_MISSING
run_topk=$RUN_TOPK
topk=$TOPK
max_policies=$MAX_POLICIES
continue_on_error=$CONTINUE_ON_ERROR
step_diagnostics_stride=$STEP_DIAGNOSTICS_STRIDE
run_all_main=$RUN_ALL_MAIN
master_root=$MASTER_ROOT
EOF
cat "$MASTER_ROOT/manifest.txt"

for run_id in $RUN_IDS; do
  run_dir="$(find_run_dir "$run_id" || true)"
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    msg="No checkpoint directory found for RUN_ID=$run_id"
    if [[ "$ALLOW_MISSING" == "1" ]]; then
      echo "[skip] $msg"
      continue
    fi
    echo "ERROR: $msg" >&2
    exit 2
  fi

  echo
  echo "########################################################################"
  echo "[training-run] $run_id"
  echo "[run_dir] $run_dir"
  echo "[master_root] $MASTER_ROOT"
  echo "########################################################################"

  RUN_DIR="$run_dir" \
  RUN_LABEL="run${run_id}" \
  OUT_ROOT="$MASTER_ROOT/run${run_id}" \
  MAPS="$MAPS" \
  NUM_EPISODES="$NUM_EPISODES" \
  RUN_TOPK="$RUN_TOPK" \
  RUN_ALL_MAIN="$RUN_ALL_MAIN" \
  TOPK="$TOPK" \
  STEP_DIAGNOSTICS_STRIDE="$STEP_DIAGNOSTICS_STRIDE" \
  MAX_POLICIES="$MAX_POLICIES" \
  CONTINUE_ON_ERROR="$CONTINUE_ON_ERROR" \
  bash runs/45_eval_every_eval_script_all_policies_seed104.sh
done

python scripts/summarize_eval_matrix.py \
  --root "$MASTER_ROOT" \
  --output "$MASTER_ROOT/all_runs_all_eval_scripts_all_policies_summary.csv" \
  --best-output "$MASTER_ROOT/all_runs_best_policy_by_eval_script.csv" \
  --markdown-output "$MASTER_ROOT/all_runs_best_policy_by_eval_script.md" \
  --top-n 20

python scripts/summarize_episode_mean_std.py \
  --root "$MASTER_ROOT" \
  --output "$MASTER_ROOT/all_runs_table_ready_episode_mean_std.csv" \
  --markdown-output "$MASTER_ROOT/all_runs_table_ready_episode_mean_std.md" \
  --top-n 20

find "$MASTER_ROOT" -type f \( -name '*.csv' -o -name '*.md' -o -name 'console.log' -o -name 'manifest.txt' \) | sort > "$MASTER_ROOT/result_files.txt"

echo
echo "[done] master_root=$MASTER_ROOT"
echo "[done] all_runs_summary=$MASTER_ROOT/all_runs_all_eval_scripts_all_policies_summary.csv"
echo "[done] all_runs_best=$MASTER_ROOT/all_runs_best_policy_by_eval_script.csv"
echo "[done] all_runs_table_ready=$MASTER_ROOT/all_runs_table_ready_episode_mean_std.csv"
echo "[done] result_files=$MASTER_ROOT/result_files.txt"
