#!/usr/bin/env bash
# Compute PairSat / V_E / IAR from true intervention pairs.
# This is separate from closed-loop eval because those metrics require
# same-state intervention pairs (s, a+, a-), not only episode returns.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ID="${RUN_ID:-12}"
MAPS="${MAPS:-104}"
MAX_PAIRS="${MAX_PAIRS:-512}"
BATCH_SIZE="${BATCH_SIZE:-256}"
ENERGY_SAMPLES="${ENERGY_SAMPLES:-4}"
ER_MARGIN="${ER_MARGIN:-0.05}"
POSITIVE_ACTION="${POSITIVE_ACTION:-behavior}"
DIAG_ALL_POLICIES="${DIAG_ALL_POLICIES:-0}"

if [[ -z "${RUN_DIR:-}" ]]; then
  case "$RUN_ID" in
    10) PATTERN="logs/10_train_dbp_only_seed${MAPS}_*" ;;
    11) PATTERN="logs/11_train_pvp_dbp_critic_seed${MAPS}_50k_*" ;;
    12) PATTERN="logs/12_train_doveer_er_seed${MAPS}_50000_*" ;;
    13) PATTERN="logs/13_train_doveer_er_no_pv_constraint_seed${MAPS}_50000_*" ;;
    *)  PATTERN="logs/${RUN_ID}_train_*seed${MAPS}*" ;;
  esac
  RUN_DIR="$(find $PATTERN -type d -name 'pvp_dacer_*_s0' 2>/dev/null | sort | tail -n 1 || true)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "[error] RUN_DIR not found. Set RUN_DIR=/path/to/pvp_dacer_*_s0" >&2
  exit 2
fi

if [[ -z "${DATA_ROOT:-}" ]]; then
  # Prefer current run data, then common project-level demo roots, then previous logs.
  for cand in \
    "$RUN_DIR/data" \
    "$RUN_DIR/recorded_data" \
    "$(dirname "$RUN_DIR")/data" \
    "$(dirname "$RUN_DIR")/recorded_data" \
    "data${MAPS}" \
    "data" \
    "logs"; do
    if [[ -e "$cand" ]] && find "$cand" -name '*.pkl' -print -quit | grep -q .; then
      DATA_ROOT="$cand"
      break
    fi
  done
fi

if [[ -z "${DATA_ROOT:-}" || ! -e "$DATA_ROOT" ]]; then
  echo "[error] DATA_ROOT not found. Set DATA_ROOT to demo/replay pkl directory containing intervention pairs." >&2
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${OUT_DIR:-logs/43_intervention_pair_diag_run${RUN_ID}_seed${MAPS}_${TS}}"
mkdir -p "$OUT_DIR"

POLICY_ARGS=()
if [[ "$DIAG_ALL_POLICIES" == "1" ]]; then
  echo "[diag] evaluating all policies under RUN_DIR"
else
  if [[ -n "${POLICY:-}" ]]; then
    POLICY_ARGS+=(--policy "$POLICY")
  else
    POLICY="$(find "$RUN_DIR" -maxdepth 1 -type f \( -name 'policy-*.pkl' -o -name 'final_policy_*.pkl' -o -name 'stage1b_offline_bc_policy.pkl' \) | sort | tail -n 1 || true)"
    if [[ -z "$POLICY" ]]; then
      echo "[error] POLICY not set and no checkpoint found in $RUN_DIR" >&2
      exit 2
    fi
    POLICY_ARGS+=(--policy "$POLICY")
  fi
fi

echo "[run_id] $RUN_ID"
echo "[run_dir] $RUN_DIR"
echo "[policy] ${POLICY:-ALL_POLICIES}"
echo "[data_root] $DATA_ROOT"
echo "[out_dir] $OUT_DIR"

python -u scripts/run_intervention_pair_diagnostics.py \
  --log_dir "$RUN_DIR" \
  "${POLICY_ARGS[@]}" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUT_DIR" \
  --max_pairs "$MAX_PAIRS" \
  --batch_size "$BATCH_SIZE" \
  --energy_samples "$ENERGY_SAMPLES" \
  --er_margin "$ER_MARGIN" \
  --positive_action "$POSITIVE_ACTION" \
  2>&1 | tee "$OUT_DIR/console.log"

# Show the paper-ready summary at the end.
echo
cat "$OUT_DIR/intervention_pair_summary.md"
