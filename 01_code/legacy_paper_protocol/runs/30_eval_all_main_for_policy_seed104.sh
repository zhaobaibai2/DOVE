#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

POLICY="${POLICY:-}"
MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"

if [[ -z "$POLICY" ]]; then
  echo "ERROR: set POLICY=/path/to/policy.pkl"
  exit 2
fi

export POLICY MAPS NUM_EPISODES

bash runs/20_eval_vanilla_policy_seed104.sh
bash runs/21_eval_upvgds_no_gate_policy_seed104.sh
bash runs/22_eval_upvgds_final_gate_full_policy_seed104.sh
bash runs/23_eval_final_gate_no_disagreement_policy_seed104.sh
bash runs/24_eval_final_gate_proxy_only_policy_seed104.sh
bash runs/25_eval_final_gate_prior_only_policy_seed104.sh
bash runs/26_eval_topk_rerank_policy_seed104.sh
