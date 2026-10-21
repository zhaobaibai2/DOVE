#!/usr/bin/env bash
set -euo pipefail

# Reproduce Table V: Training-side ablation results.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[TABLE V] GPU configuration"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; table aggregation itself only reads CSV files."
fi

echo
echo "[TABLE V] Method input CSV mapping"
echo "1. DBP only"
echo "   row manifest: latex3/table_inputs/by_row/train_ablation_rows/dbp_only.csv"
echo "   input csv [checkpoint stage1b]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run10_dbp_only/vanilla/stage1b_offline_bc_policy/episodes_vanilla.csv"
echo
echo "2. DBP+critic / w/o ER"
echo "   row manifest: latex3/table_inputs/by_row/train_ablation_rows/dbp_critic.csv"
echo "   input csv [checkpoint 44000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run11_pvp_dbp_critic/upvgds_final_gate_full/policy-44000-87998/episodes_upvgds_final_gate_full.csv"

echo "   input csv [checkpoint final50000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run11_pvp_dbp_critic/upvgds_final_gate_full/final_policy_50000/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 27000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run11_pvp_dbp_critic/upvgds_final_gate_full/policy-27000-53998/episodes_upvgds_final_gate_full.csv"

echo "   input csv [checkpoint 16000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run11_pvp_dbp_critic/upvgds_final_gate_full/policy-16000-31998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 48000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run11_pvp_dbp_critic/upvgds_final_gate_full/policy-48000-95998/episodes_upvgds_final_gate_full.csv"
echo



echo "3. DOVEER w/o CPC"
echo "   row manifest: latex3/table_inputs/by_row/train_ablation_rows/no_cpc.csv"
echo "   input csv [checkpoint 21000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run13_doveer_no_pv_constraint/upvgds_final_gate_full/policy-21000-41998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 23000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run13_doveer_no_pv_constraint/upvgds_final_gate_full/policy-23000-45998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 25000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run13_doveer_no_pv_constraint/upvgds_final_gate_full/policy-25000-49998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 1000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run13_doveer_no_pv_constraint/upvgds_final_gate_full/policy-1000-1998/episodes_upvgds_final_gate_full.csv"

echo "   input csv [checkpoint 18000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run13_doveer_no_pv_constraint/upvgds_final_gate_full/policy-18000-35998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 7000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run13_doveer_no_pv_constraint/upvgds_final_gate_full/policy-7000-13998/episodes_upvgds_final_gate_full.csv"
echo


echo "4. DOVEER full (ER+CPC)"
echo "   row manifest: latex3/table_inputs/by_row/train_ablation_rows/full.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-12000-23998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-13000-25998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-14000-27998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-32000-63998/episodes_upvgds_final_gate_full.csv"

echo


python scripts/build_latex3_table_train_ablation.py
