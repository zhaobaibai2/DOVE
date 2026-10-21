#!/usr/bin/env bash
# Run every single-policy eval/ablation script in runs/ over every policy checkpoint.
# This is intentionally script-driven: it uses runs/20_eval_... through runs/26_eval_...
# as the source of truth for the ablation definitions, rather than re-encoding modes here.
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

RUN_DIR="${RUN_DIR:-}"
RUN_LABEL="${RUN_LABEL:-custom}"
MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
TOPK="${TOPK:-8}"
RUN_TOPK="${RUN_TOPK:-0}"
RUN_ALL_MAIN="${RUN_ALL_MAIN:-0}"
MAX_POLICIES="${MAX_POLICIES:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-20}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find logs/12_train_doveer_er_seed104_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: cannot find policy checkpoint directory. Set RUN_DIR=/path/to/pvp_dacer_xxx_s0" >&2
  exit 2
fi

# Discover the real ablation scripts in runs/. These scripts each expect POLICY=... .
# By default this finds 20_eval_... through 26_eval_... and skips top-k unless RUN_TOPK=1.
if [[ -n "${EVAL_SCRIPTS:-}" ]]; then
  # Space-separated list, e.g. EVAL_SCRIPTS="runs/20_eval...sh runs/22_eval...sh".
  read -r -a EVAL_SCRIPT_LIST <<< "$EVAL_SCRIPTS"
else
  mapfile -t EVAL_SCRIPT_LIST < <(find runs -maxdepth 1 -type f -name '[0-9][0-9]_eval_*_policy_seed104.sh' | sort)
fi

FILTERED_SCRIPTS=()
for script in "${EVAL_SCRIPT_LIST[@]}"; do
  if [[ ! -f "$script" ]]; then
    echo "[warn] eval script not found, skip: $script" >&2
    continue
  fi
  base="$(basename "$script")"
  if [[ "$base" == *all_main_for_policy* && "$RUN_ALL_MAIN" != "1" ]]; then
    echo "[skip] $script because RUN_ALL_MAIN=$RUN_ALL_MAIN. It duplicates the individual 20-25 eval scripts; set RUN_ALL_MAIN=1 to include it."
    continue
  fi
  if [[ "$base" == *topk* && "$RUN_TOPK" != "1" ]]; then
    echo "[skip] $script because RUN_TOPK=$RUN_TOPK. Set RUN_TOPK=1 to include top-k rerank."
    continue
  fi
  FILTERED_SCRIPTS+=("$script")
done
if [[ ${#FILTERED_SCRIPTS[@]} -eq 0 ]]; then
  echo "ERROR: no eval scripts selected." >&2
  exit 2
fi

mapfile -t POLICIES < <(
  find "$RUN_DIR" -maxdepth 1 -type f \
    \( -name 'policy-*.pkl' -o -name 'policy_*.pkl' -o -name 'final_policy_*.pkl' -o -name 'stage1b_offline_bc_policy.pkl' \) \
    ! -name '*.meta.pkl' | sort -V
)
if [[ ${#POLICIES[@]} -eq 0 ]]; then
  echo "ERROR: no policy checkpoints found under $RUN_DIR" >&2
  exit 2
fi
if [[ "$MAX_POLICIES" != "0" ]]; then
  POLICIES=("${POLICIES[@]:0:$MAX_POLICIES}")
fi

TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
OUT_ROOT="${OUT_ROOT:-${REPO_DIR}/logs/45_eval_every_eval_script_all_policies_${RUN_LABEL}_seed${MAP_TAG}_${TS}}"
mkdir -p "$OUT_ROOT"
STATUS_CSV="$OUT_ROOT/executed_eval_script_policy_matrix.csv"

mode_from_script() {
  local base="$1"
  base="${base#??_eval_}"
  base="${base%_policy_seed104.sh}"
  echo "$base"
}

sanitize() {
  local value="$1"
  value="${value%.pkl}"
  echo "$value" | sed -E 's/[^A-Za-z0-9._-]+/_/g'
}

csv_escape() {
  local value="${1:-}"
  value="${value//\"/\"\"}"
  printf '"%s"' "$value"
}

cat > "$OUT_ROOT/manifest.txt" <<EOF
run_label=$RUN_LABEL
run_dir=$RUN_DIR
maps=$MAPS
num_episodes=$NUM_EPISODES
policy_count=${#POLICIES[@]}
eval_script_count=${#FILTERED_SCRIPTS[@]}
run_topk=$RUN_TOPK
topk=$TOPK
max_policies=$MAX_POLICIES
continue_on_error=$CONTINUE_ON_ERROR
step_diagnostics_stride=$STEP_DIAGNOSTICS_STRIDE
run_all_main=$RUN_ALL_MAIN
conda_env=${CONDA_DEFAULT_ENV:-}
python=$(which python)
eval_scripts=${FILTERED_SCRIPTS[*]}
EOF
cat "$OUT_ROOT/manifest.txt"
check_nvidia_gpu | tee "$OUT_ROOT/nvidia-smi.txt"

printf 'run_label,mode,eval_script,policy,policy_basename,out_dir,exit_code,start_time,end_time,elapsed_seconds,eval_csv,episode_csv\n' > "$STATUS_CSV"

for script in "${FILTERED_SCRIPTS[@]}"; do
  mode="$(mode_from_script "$(basename "$script")")"
  echo
  echo "########################################################################"
  echo "[eval-script] $script"
  echo "[mode] $mode"
  echo "[policies] ${#POLICIES[@]}"
  echo "########################################################################"

  for policy in "${POLICIES[@]}"; do
    policy_base="$(basename "$policy")"
    policy_tag="$(sanitize "$policy_base")"
    mode_dir="$OUT_ROOT/$mode"
    out_dir="$mode_dir/$policy_tag"
    mkdir -p "$out_dir"

    start_iso="$(date -Is)"
    start_sec="$(date +%s)"
    echo
    echo "========================================================================"
    echo "[mode] $mode"
    echo "[script] $script"
    echo "[policy] $policy_base"
    echo "[out] $out_dir"
    echo "========================================================================"

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

    end_iso="$(date -Is)"
    end_sec="$(date +%s)"
    elapsed=$((end_sec - start_sec))
    eval_csv="$(find "$out_dir" -maxdepth 1 -type f -name 'eval_*.csv' | sort | paste -sd ';' -)"
    episode_csv="$(find "$out_dir" -maxdepth 1 -type f -name 'episodes_*.csv' | sort | paste -sd ';' -)"

    {
      csv_escape "$RUN_LABEL"; printf ','
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
    } >> "$STATUS_CSV"

    if [[ "$rc" != "0" ]]; then
      echo "[warn] eval failed: script=$script policy=$policy_base rc=$rc" >&2
      if [[ "$CONTINUE_ON_ERROR" != "1" ]]; then
        exit "$rc"
      fi
    fi
  done
done

python scripts/summarize_eval_matrix.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/all_eval_scripts_all_policies_summary.csv" \
  --best-output "$OUT_ROOT/best_policy_by_eval_script.csv" \
  --markdown-output "$OUT_ROOT/best_policy_by_eval_script.md" \
  --top-n 20

python scripts/summarize_episode_mean_std.py \
  --root "$OUT_ROOT" \
  --output "$OUT_ROOT/table_ready_episode_mean_std.csv" \
  --markdown-output "$OUT_ROOT/table_ready_episode_mean_std.md" \
  --top-n 20

echo
echo "[done] out_root=$OUT_ROOT"
echo "[done] status_csv=$STATUS_CSV"
echo "[done] summary_csv=$OUT_ROOT/all_eval_scripts_all_policies_summary.csv"
echo "[done] best_csv=$OUT_ROOT/best_policy_by_eval_script.csv"
echo "[done] table_ready_csv=$OUT_ROOT/table_ready_episode_mean_std.csv"
