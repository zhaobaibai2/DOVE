#!/usr/bin/env bash
set -euo pipefail

# Reproduce Table IV: Main task results.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[TABLE IV] GPU configuration"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; table aggregation itself only reads CSV files."
fi

echo
echo "[TABLE IV] Method input CSV mapping"
echo "1. DOVEER full (ours, ER+CPC)"
echo "   row manifest: latex3/table_inputs/by_row/main_task_rows/doveer_full.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-12000-23998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-13000-25998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-14000-27998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-32000-63998/episodes_upvgds_final_gate_full.csv"

echo
python scripts/build_latex3_table_main_task.py
