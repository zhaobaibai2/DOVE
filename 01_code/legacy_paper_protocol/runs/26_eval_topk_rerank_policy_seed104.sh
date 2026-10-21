#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

RUN12_POLICY_ROOT="$REPO_DIR/logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0"
RUN13_POLICY_ROOT="$REPO_DIR/logs/13_train_doveer_er_no_pv_constraint_seed104_50000_20260522_152214/pvp_dacer_20260522_152226_s0"
DEFAULT_POLICY_ROOTS="$RUN12_POLICY_ROOT $RUN13_POLICY_ROOT"
POLICY_ROOTS="${POLICY_ROOTS:-$DEFAULT_POLICY_ROOTS}"
POLICY="${POLICY:-}"
MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
TOPK="${TOPK:-16}"
HORIZON="${HORIZON:-1000}"
OUT_PARENT="${OUT_PARENT:-${REPO_DIR}/logs}"
TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
RUN_NAME="26_eval_topk${TOPK}_rerank_run12_run13_all_policies_seed${MAP_TAG}_${TS}"
OUT_DIR="${OUT_DIR:-${OUT_PARENT}/${RUN_NAME}}"
mkdir -p "$OUT_DIR"

echo "[run] $RUN_NAME"
echo "[meaning] Test-time critic Top-K reranking baseline. No UPV-GDS, no Final Gate."
echo "[topk_impl] batched JAX/JIT path in scripts/eval_pvp_policies_fixed_v3.py; adapter log should include JIT=yes."
echo "[maps] $MAPS"
echo "[num_episodes_per_policy_per_map] $NUM_EPISODES"
echo "[topk] $TOPK"
echo "[horizon] $HORIZON"
echo "[out] $OUT_DIR"
echo "[env] conda=$CONDA_DEFAULT_ENV python=$(which python) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset} JAX_PLATFORMS=${JAX_PLATFORMS:-unset}"
check_nvidia_gpu
python - <<'PYCUDA'
import jax
print('[jax]', jax.default_backend(), jax.devices())
try:
    import torch
    print('[torch_cuda]', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')
except Exception as e:
    print('[torch_cuda] skipped:', e)
PYCUDA

run_one_root() {
  local policy_root="$1"
  local root_tag="$2"
  local sub_out="$OUT_DIR/$root_tag"
  mkdir -p "$sub_out"

  if [[ ! -d "$policy_root" ]]; then
    echo "ERROR: policy root not found for $root_tag: $policy_root"
    exit 2
  fi

  shopt -s nullglob
  local policy_count=0
  for _p in "$policy_root"/policy-*.pkl "$policy_root"/policy_*.pkl "$policy_root"/final_policy_*.pkl "$policy_root"/stage1b_offline_bc_policy.pkl; do
    [[ "$(basename "$_p")" == *.meta.pkl ]] && continue
    policy_count=$((policy_count + 1))
  done
  shopt -u nullglob

  if [[ "$policy_count" -le 0 ]]; then
    echo "ERROR: no policy checkpoints found for $root_tag in $policy_root"
    exit 2
  fi

  echo
  echo "======================================================================"
  echo "[root] $root_tag"
  echo "[policy_root] $policy_root"
  echo "[policy_count] $policy_count"
  echo "[sub_out] $sub_out"
  echo "======================================================================"

  python -u scripts/eval_pvp_policies_fixed_v3.py     --log_dir "$policy_root"     --output_csv "$sub_out/eval_topk_rerank_all_policies.csv"     --episode_output_csv "$sub_out/episodes_topk_rerank_all_policies.csv"     --tensorboard_dir "$sub_out/tb"     --maps $MAPS     --num_episodes "$NUM_EPISODES"     --horizon "$HORIZON"     --no_use_render     --rerank_topk "$TOPK"     --critic_objective cost     2>&1 | tee "$sub_out/console.log"

  echo "[done:$root_tag] summary csv: $sub_out/eval_topk_rerank_all_policies.csv"
  echo "[done:$root_tag] episode csv: $sub_out/episodes_topk_rerank_all_policies.csv"
}

if [[ -n "$POLICY" ]]; then
  POLICY_ROOT="$(cd "$(dirname "$POLICY")" && pwd)"
  POLICY_NAME="$(basename "$POLICY")"
  SINGLE_OUT="$OUT_DIR/single_policy"
  mkdir -p "$SINGLE_OUT"
  echo "[single_policy] $POLICY"
  python -u scripts/eval_pvp_policies_fixed_v3.py     --log_dir "$POLICY_ROOT"     --policy "$POLICY_NAME"     --output_csv "$SINGLE_OUT/eval_topk_rerank_single_policy.csv"     --episode_output_csv "$SINGLE_OUT/episodes_topk_rerank_single_policy.csv"     --tensorboard_dir "$SINGLE_OUT/tb"     --maps $MAPS     --num_episodes "$NUM_EPISODES"     --horizon "$HORIZON"     --no_use_render     --rerank_topk "$TOPK"     --critic_objective cost     2>&1 | tee "$SINGLE_OUT/console.log"
else
  idx=0
  for root in $POLICY_ROOTS; do
    idx=$((idx + 1))
    case "$root" in
      *12_train_doveer_er_seed104_50000*) tag="run12_doveer_full" ;;
      *13_train_doveer_er_no_pv_constraint_seed104_50000*) tag="run13_no_pv_constraint" ;;
      *) tag="root${idx}_$(basename "$root")" ;;
    esac
    run_one_root "$root" "$tag"
  done
fi

echo
printf '%s
' "[done] all outputs under: $OUT_DIR"
find "$OUT_DIR" -maxdepth 2 -type f \( -name '*.csv' -o -name 'console.log' \) | sort
