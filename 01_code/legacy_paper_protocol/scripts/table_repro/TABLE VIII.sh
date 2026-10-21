#!/usr/bin/env bash
set -euo pipefail

# Reproduce Table VIII: Actor-prior pair audit.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[TABLE VIII] GPU configuration"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; table aggregation itself only reads CSV files."
fi

echo
echo "[TABLE VIII] Method input CSV mapping"
echo "1. DBP positive-only"
echo "   row manifest: latex3/table_inputs/by_row/pair_audit_rows/dbp_only.csv"
echo "   input csv [checkpoint stage1b]: logs/99_every_eval_script_and_pair_diag_seed104_20260524_092640/pair_diag/run10_stage1b_summary.csv"
echo
echo "2. DOVEER full (ER + CPC)"
echo "   row manifest: latex3/table_inputs/by_row/pair_audit_rows/full.csv"
echo "   input csv [checkpoint 11000]: logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151/run12_full_policy-11000-21998_summary.csv"
echo "   input csv [checkpoint 12000]: logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151/run12_full_policy-12000-23998_summary.csv"
echo "   input csv [checkpoint 13000]: logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151/run12_full_policy-13000-25998_summary.csv"
echo
echo "3. EnergyRank w/o CPC"
echo "   row manifest: latex3/table_inputs/by_row/pair_audit_rows/no_cpc.csv"
echo "   input csv [checkpoint 20000]: logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151/run13_nopv_policy-20000-39998_summary.csv"
echo "   input csv [checkpoint 23000]: logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151/run13_nopv_policy-23000-45998_summary.csv"
echo "   input csv [checkpoint 25000]: logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151/run13_nopv_policy-25000-49998_summary.csv"
echo "   input csv [checkpoint 26000]: logs/106_table_vi_nohup_same_pair_source_gpu_seed104_20260525_112151/run13_nopv_policy-26000-51998_summary.csv"

echo
python scripts/build_latex3_table_pair_audit.py
