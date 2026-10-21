#!/usr/bin/env bash
set -euo pipefail

# Reproduce Table VI: Test-time ablation results.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

echo "[TABLE VI] GPU configuration"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; table aggregation itself only reads CSV files."
fi

echo
echo "[TABLE VI] Method input CSV mapping"
echo "1. Vanilla policy (no GDS)"
echo "   row manifest: latex3/table_inputs/by_row/testtime_ablation_rows/vanilla.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/vanilla/policy-12000-23998/episodes_vanilla.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/vanilla/policy-13000-25998/episodes_vanilla.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/vanilla/policy-14000-27998/episodes_vanilla.csv"
echo "   input csv [checkpoint 20000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/vanilla/policy-20000-39998/episodes_vanilla.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/vanilla/policy-32000-63998/episodes_vanilla.csv"
echo
echo "2. Top-K rerank (K=8)"
echo "   row manifest: latex3/table_inputs/by_row/testtime_ablation_rows/topk.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/topk_rerank/policy-12000-23998/episodes_topk_rerank.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/topk_rerank/policy-13000-25998/episodes_topk_rerank.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/topk_rerank/policy-14000-27998/episodes_topk_rerank.csv"
echo "   input csv [checkpoint 20000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/topk_rerank/policy-20000-39998/episodes_topk_rerank.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/topk_rerank/policy-32000-63998/episodes_topk_rerank.csv"
echo
echo "3. UPV-GDS proposal only"
echo "   row manifest: latex3/table_inputs/by_row/testtime_ablation_rows/no_gate.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_no_gate/policy-12000-23998/episodes_upvgds_no_gate.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_no_gate/policy-13000-25998/episodes_upvgds_no_gate.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_no_gate/policy-14000-27998/episodes_upvgds_no_gate.csv"
echo "   input csv [checkpoint 20000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_no_gate/policy-20000-39998/episodes_upvgds_no_gate.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_no_gate/policy-32000-63998/episodes_upvgds_no_gate.csv"
echo
echo "4. UPV-GDS + proxy-only gate"
echo "   row manifest: latex3/table_inputs/by_row/testtime_ablation_rows/proxy_only.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_proxy_only/policy-12000-23998/episodes_final_gate_proxy_only.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_proxy_only/policy-13000-25998/episodes_final_gate_proxy_only.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_proxy_only/policy-14000-27998/episodes_final_gate_proxy_only.csv"
echo "   input csv [checkpoint 20000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_proxy_only/policy-20000-39998/episodes_final_gate_proxy_only.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_proxy_only/policy-32000-63998/episodes_final_gate_proxy_only.csv"
echo
echo "5. UPV-GDS + prior-only gate"
echo "   row manifest: latex3/table_inputs/by_row/testtime_ablation_rows/prior_only.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_prior_only/policy-12000-23998/episodes_final_gate_prior_only.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_prior_only/policy-13000-25998/episodes_final_gate_prior_only.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_prior_only/policy-14000-27998/episodes_final_gate_prior_only.csv"
echo "   input csv [checkpoint 20000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_prior_only/policy-20000-39998/episodes_final_gate_prior_only.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_prior_only/policy-32000-63998/episodes_final_gate_prior_only.csv"
echo
echo "6. UPV-GDS + gate w/o disagreement check"
echo "   row manifest: latex3/table_inputs/by_row/testtime_ablation_rows/no_disagreement.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_no_disagreement/policy-12000-23998/episodes_final_gate_no_disagreement.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_no_disagreement/policy-13000-25998/episodes_final_gate_no_disagreement.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_no_disagreement/policy-14000-27998/episodes_final_gate_no_disagreement.csv"
echo "   input csv [checkpoint 20000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_no_disagreement/policy-20000-39998/episodes_final_gate_no_disagreement.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/final_gate_no_disagreement/policy-32000-63998/episodes_final_gate_no_disagreement.csv"
echo
echo "7. UPV-GDS + full gate"
echo "   row manifest: latex3/table_inputs/by_row/testtime_ablation_rows/full.csv"
echo "   input csv [checkpoint 12000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-12000-23998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 13000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-13000-25998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 14000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-14000-27998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 20000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-20000-39998/episodes_upvgds_final_gate_full.csv"
echo "   input csv [checkpoint 32000]: eval2/tb/all_checkpoints_map104_20260527_200052/evaluation_matrix/run12_doveer_full/upvgds_final_gate_full/policy-32000-63998/episodes_upvgds_final_gate_full.csv"
echo
python scripts/build_latex3_table_testtime_ablation.py
