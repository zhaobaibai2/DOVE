#!/usr/bin/env bash
# Resume runner for runs/50_run_all_useful_evals_gpu_seed104.sh
#
# Purpose:
#   Continue the interrupted run without re-running completed evals.
#
# How it decides "completed":
#   For each (training run, eval mode, policy), it checks the expected output dir:
#     $MASTER_LOG_DIR/closed_loop_matrix/$run_label/$mode/$policy_tag
#   Completed means:
#     - at least one non-empty eval*.csv / eval_*.csv exists anywhere under that dir
#     - and at least one non-empty episodes*.csv / episodes_*.csv exists anywhere under that dir
#   This handles topk_rerank correctly, because topk writes CSVs in a subdirectory.
#
# Typical usage from repo root:
#   chmod +x runs/50_resume_useful_evals_gpu_seed104.sh
#   MASTER_LOG_DIR=logs/50_all_useful_evals_gpu_seed104_20260525_205921 \
#     bash runs/50_resume_useful_evals_gpu_seed104.sh
#
# Preview only:
#   DRY_RUN=1 MASTER_LOG_DIR=logs/50_all_useful_evals_gpu_seed104_20260525_205921 \
#     bash runs/50_resume_useful_evals_gpu_seed104.sh

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
export RUN_IDS="${RUN_IDS:-auto}"
export ALLOW_MISSING="${ALLOW_MISSING:-1}"
export RUN_TOPK="${RUN_TOPK:-1}"
export RUN_ALL_MAIN="${RUN_ALL_MAIN:-0}"
export TOPK="${TOPK:-8}"
export MAX_POLICIES="${MAX_POLICIES:-0}"
export CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
export STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-20}"
export DRY_RUN="${DRY_RUN:-0}"

# IMPORTANT:
# Set this to the interrupted run dir if you want to continue in-place.
# If omitted, the script creates a new resume dir, but then it cannot see old outputs to skip them.
TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
export MASTER_LOG_DIR="${MASTER_LOG_DIR:-$REPO_DIR/logs/50_resume_useful_evals_gpu_seed${MAP_TAG}_${TS}}"
export MATRIX_ROOT="${MATRIX_ROOT:-$MASTER_LOG_DIR/closed_loop_matrix}"
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

sanitize_policy_tag() {
  local value="$1"
  value="${value%.pkl}"
  echo "$value" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

mode_from_script() {
  local base="$1"
  base="${base#??_eval_}"
  base="${base%_policy_seed104.sh}"
  echo "$base"
}

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

has_nonempty_match() {
  local dir="$1"
  local pattern1="$2"
  local pattern2="$3"
  [[ -d "$dir" ]] || return 1
  find "$dir" -type f \( -name "$pattern1" -o -name "$pattern2" \) -size +0c | grep -q .
}

is_eval_complete() {
  local out_dir="$1"
  has_nonempty_match "$out_dir" 'eval*.csv' 'eval_*.csv' || return 1
  has_nonempty_match "$out_dir" 'episodes*.csv' 'episodes_*.csv' || return 1
  return 0
}

list_eval_csvs() {
  local out_dir="$1"
  find "$out_dir" -type f \( -name 'eval*.csv' -o -name 'eval_*.csv' \) -size +0c 2>/dev/null | sort | paste -sd ';' -
}

list_episode_csvs() {
  local out_dir="$1"
  find "$out_dir" -type f \( -name 'episodes*.csv' -o -name 'episodes_*.csv' \) -size +0c 2>/dev/null | sort | paste -sd ';' -
}

RUN_DIR_LIST="$MASTER_LOG_DIR/discovered_training_runs.txt"
: > "$RUN_DIR_LIST"

if [[ "$RUN_IDS" == "auto" || "$RUN_IDS" == "all" ]]; then
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
      parent="$(basename "$(dirname "$d")")"
      leaf="$(basename "$d")"
      label="$(sanitize_label "${parent}_${leaf}")"
      printf '%s\t%s\n' "$label" "$d" >> "$RUN_DIR_LIST"
    else
      echo "[skip] RUN_ID=$run_id has no policy checkpoint: $d" >&2
    fi
  done
fi

RUN_COUNT="$(wc -l < "$RUN_DIR_LIST" | tr -d ' ')"
if [[ "$RUN_COUNT" == "0" ]]; then
  echo "ERROR: no training checkpoint directories with policy .pkl files were found under $REPO_DIR/logs" >&2
  exit 2
fi

if [[ -n "${EVAL_SCRIPTS:-}" ]]; then
  read -r -a EVAL_SCRIPT_LIST <<< "$EVAL_SCRIPTS"
else
  mapfile -t EVAL_SCRIPT_LIST < <(find runs -maxdepth 1 -type f -name '[0-9][0-9]_eval_*_policy_seed104.sh' | sort)
fi

FILTERED_SCRIPTS=()
for script in "${EVAL_SCRIPT_LIST[@]}"; do
  [[ -f "$script" ]] || { echo "[warn] eval script not found, skip: $script" >&2; continue; }
  base="$(basename "$script")"
  if [[ "$base" == *all_main_for_policy* && "$RUN_ALL_MAIN" != "1" ]]; then
    echo "[skip] $script because RUN_ALL_MAIN=$RUN_ALL_MAIN"
    continue
  fi
  if [[ "$base" == *topk* && "$RUN_TOPK" != "1" ]]; then
    echo "[skip] $script because RUN_TOPK=$RUN_TOPK"
    continue
  fi
  FILTERED_SCRIPTS+=("$script")
done

if [[ ${#FILTERED_SCRIPTS[@]} -eq 0 ]]; then
  echo "ERROR: no eval scripts selected." >&2
  exit 2
fi

RESUME_STATUS="$MASTER_LOG_DIR/resume_executed_eval_script_policy_matrix.csv"
if [[ ! -f "$RESUME_STATUS" ]]; then
  printf 'run_label,mode,eval_script,policy,policy_basename,out_dir,action,exit_code,start_time,end_time,elapsed_seconds,eval_csv,episode_csv\n' > "$RESUME_STATUS"
fi

cat > "$MASTER_LOG_DIR/resume_manifest.txt" <<EOF
maps=$MAPS
num_episodes=$NUM_EPISODES
run_ids=$RUN_IDS
run_count=$RUN_COUNT
run_topk=$RUN_TOPK
run_all_main=$RUN_ALL_MAIN
topk=$TOPK
max_policies=$MAX_POLICIES
continue_on_error=$CONTINUE_ON_ERROR
dry_run=$DRY_RUN
master_log_dir=$MASTER_LOG_DIR
matrix_root=$MATRIX_ROOT
discovered_training_runs=$RUN_DIR_LIST
resume_status=$RESUME_STATUS
eval_scripts=${FILTERED_SCRIPTS[*]}
EOF

echo "[resume manifest]"
cat "$MASTER_LOG_DIR/resume_manifest.txt"
echo
echo "[discovered training runs]"
cat "$RUN_DIR_LIST"

if [[ "$DRY_RUN" != "1" ]]; then
  check_nvidia_gpu | tee "$MASTER_LOG_DIR/resume_nvidia-smi.txt"
fi

total=0
skipped=0
ran=0
failed=0

while IFS=$'\t' read -r run_label run_dir; do
  [[ -n "$run_label" && -n "$run_dir" ]] || continue

  mapfile -t POLICIES < <(
    find "$run_dir" -maxdepth 1 -type f \
      \( -name 'policy-*.pkl' -o -name 'policy_*.pkl' -o -name 'final_policy_*.pkl' -o -name 'stage1b_offline_bc_policy.pkl' \) \
      ! -name '*.meta.pkl' | sort -V
  )
  if [[ ${#POLICIES[@]} -eq 0 ]]; then
    echo "[warn] no policies found in $run_dir" >&2
    continue
  fi
  if [[ "$MAX_POLICIES" != "0" ]]; then
    POLICIES=("${POLICIES[@]:0:$MAX_POLICIES}")
  fi

  echo
  echo "########################################################################"
  echo "[training-run] $run_label"
  echo "[run_dir] $run_dir"
  echo "[policies] ${#POLICIES[@]}"
  echo "########################################################################"

  RUN_OUT_ROOT="$MATRIX_ROOT/$run_label"
  mkdir -p "$RUN_OUT_ROOT"
  RUN_STATUS="$RUN_OUT_ROOT/executed_eval_script_policy_matrix.csv"
  if [[ ! -f "$RUN_STATUS" ]]; then
    printf 'run_label,mode,eval_script,policy,policy_basename,out_dir,exit_code,start_time,end_time,elapsed_seconds,eval_csv,episode_csv\n' > "$RUN_STATUS"
  fi

  for script in "${FILTERED_SCRIPTS[@]}"; do
    mode="$(mode_from_script "$(basename "$script")")"
    mode_dir="$RUN_OUT_ROOT/$mode"
    mkdir -p "$mode_dir"

    echo
    echo "========================================================================"
    echo "[eval-script] $script"
    echo "[mode] $mode"
    echo "========================================================================"

    for policy in "${POLICIES[@]}"; do
      total=$((total + 1))
      policy_base="$(basename "$policy")"
      policy_tag="$(sanitize_policy_tag "$policy_base")"
      out_dir="$mode_dir/$policy_tag"
      mkdir -p "$out_dir"

      if is_eval_complete "$out_dir"; then
        skipped=$((skipped + 1))
        eval_csv="$(list_eval_csvs "$out_dir")"
        episode_csv="$(list_episode_csvs "$out_dir")"
        now_iso="$(date -Is)"
        echo "[skip-complete] mode=$mode policy=$policy_base out=$out_dir"

        {
          csv_escape "$run_label"; printf ','
          csv_escape "$mode"; printf ','
          csv_escape "$script"; printf ','
          csv_escape "$policy"; printf ','
          csv_escape "$policy_base"; printf ','
          csv_escape "$out_dir"; printf ','
          csv_escape "skip_complete"; printf ','
          printf '0,'
          csv_escape "$now_iso"; printf ','
          csv_escape "$now_iso"; printf ','
          printf '0,'
          csv_escape "$eval_csv"; printf ','
          csv_escape "$episode_csv"; printf '\n'
        } >> "$RESUME_STATUS"
        continue
      fi

      echo
      echo "------------------------------------------------------------------------"
      echo "[run-missing] mode=$mode policy=$policy_base"
      echo "[out] $out_dir"
      echo "------------------------------------------------------------------------"

      start_iso="$(date -Is)"
      start_sec="$(date +%s)"

      if [[ "$DRY_RUN" == "1" ]]; then
        rc=0
        action="dry_run_missing"
        eval_csv=""
        episode_csv=""
      else
        set +e
        POLICY="$policy" \
        MAPS="$MAPS" \
        NUM_EPISODES="$NUM_EPISODES" \
        TOPK="$TOPK" \
        STEP_DIAGNOSTICS_STRIDE="$STEP_DIAGNOSTICS_STRIDE" \
        OUT_PARENT="$mode_dir" \
        OUT_DIR="$out_dir" \
        bash "$script"
        rc=$?
        set -e

        eval_csv="$(list_eval_csvs "$out_dir")"
        episode_csv="$(list_episode_csvs "$out_dir")"

        if [[ "$rc" == "0" ]] && is_eval_complete "$out_dir"; then
          action="ran_complete"
          ran=$((ran + 1))
        else
          action="ran_incomplete_or_failed"
          failed=$((failed + 1))
        fi
      fi

      end_iso="$(date -Is)"
      end_sec="$(date +%s)"
      elapsed=$((end_sec - start_sec))

      {
        csv_escape "$run_label"; printf ','
        csv_escape "$mode"; printf ','
        csv_escape "$script"; printf ','
        csv_escape "$policy"; printf ','
        csv_escape "$policy_base"; printf ','
        csv_escape "$out_dir"; printf ','
        csv_escape "$action"; printf ','
        printf '%s,' "$rc"
        csv_escape "$start_iso"; printf ','
        csv_escape "$end_iso"; printf ','
        printf '%s,' "$elapsed"
        csv_escape "$eval_csv"; printf ','
        csv_escape "$episode_csv"; printf '\n'
      } >> "$RESUME_STATUS"

      # Also append to the per-run status table so old summary workflows still see resumed jobs.
      {
        csv_escape "$run_label"; printf ','
        csv_escape "$mode"; printf ','
        csv_escape "$script"; printf ','
        csv_escape "$policy"; printf ','
        csv_escape "$policy_base"; printf ','
        csv_escape "$out_dir"; printf ','
        printf '%s,' "$rc"
        csv_escape "$start_iso"; printf ','
        csv_escape "$end_iso"; printf ','
        printf '%s,' "$elapsed"
        csv_escape "$eval_csv"; printf ','
        csv_escape "$episode_csv"; printf '\n'
      } >> "$RUN_STATUS"

      if [[ "$rc" != "0" || "$action" == "ran_incomplete_or_failed" ]]; then
        echo "[warn] eval incomplete/failed: mode=$mode policy=$policy_base rc=$rc" >&2
        if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
          exit "${rc:-1}"
        fi
      fi
    done
  done
done < "$RUN_DIR_LIST"

echo
echo "[resume counts] total=$total skipped_complete=$skipped ran_now=$ran failed_or_incomplete=$failed dry_run=$DRY_RUN"

if [[ "$DRY_RUN" != "1" ]]; then
  echo "[summarize] rebuilding summaries from $MATRIX_ROOT"

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

  find "$MASTER_LOG_DIR" -type f \( -name '*.csv' -o -name '*.md' -o -name 'console.log' -o -name '*.txt' -o -name '*.log' \) | sort > "$MASTER_LOG_DIR/result_files.txt"
fi

echo "[done] $(date)"
echo "[master_log_dir] $MASTER_LOG_DIR"
echo "[closed_loop_matrix] $MATRIX_ROOT"
echo "[resume_status] $RESUME_STATUS"
echo "[result_files] $MASTER_LOG_DIR/result_files.txt"
