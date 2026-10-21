#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
RUN_TOPK="${RUN_TOPK:-0}"
TOPK="${TOPK:-8}"
RUN_IDS="${RUN_IDS:-11 12 13}"
ALLOW_MISSING="${ALLOW_MISSING:-0}"

find_run_dir() {
  local run_id="$1"
  case "$run_id" in
    10)
      find logs/10_train_dbp_only_seed104_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    11)
      find logs/11_train_pvp_dbp_critic_seed104_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    12)
      find logs/12_train_doveer_er_seed104_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    13)
      find logs/13_train_doveer_er_no_pv_constraint_seed104_50000_* -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1
      ;;
    *)
      echo "ERROR: unknown RUN_ID=$run_id. Supported: 10 11 12 13" >&2
      return 2
      ;;
  esac
}

for run_id in $RUN_IDS; do
  run_dir="$(find_run_dir "$run_id" || true)"
  if [[ -z "$run_dir" || ! -d "$run_dir" ]]; then
    msg="No checkpoint directory found for RUN_ID=$run_id"
    if [[ "$ALLOW_MISSING" == "1" ]]; then
      echo "[skip] $msg"
      continue
    fi
    echo "ERROR: $msg"
    exit 2
  fi

  echo
  echo "########################################################################"
  echo "[training-run] $run_id"
  echo "[run_dir] $run_dir"
  echo "########################################################################"
  RUN_DIR="$run_dir" \
  RUN_LABEL="run${run_id}" \
  MAPS="$MAPS" \
  NUM_EPISODES="$NUM_EPISODES" \
  RUN_TOPK="$RUN_TOPK" \
  TOPK="$TOPK" \
  bash runs/40_eval_all_policies_all_modes_seed104.sh
done
