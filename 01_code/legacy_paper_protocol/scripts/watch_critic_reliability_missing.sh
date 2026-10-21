#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/critic_reliability_matrix_allpolicies_20260604_1603}"
LOG="${LOG:-${OUTPUT_ROOT}_missing_watch.log}"
NUM_WORKERS="${NUM_WORKERS:-2}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAPS="${MAPS:-104}"
RELIABILITY_HORIZON="${RELIABILITY_HORIZON:-25}"
STEP_DIAGNOSTICS_STRIDE="${STEP_DIAGNOSTICS_STRIDE:-1}"
FINAL_GATE_ENERGY_SAMPLES="${FINAL_GATE_ENERGY_SAMPLES:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.45}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-dacer_pvp}"

mkdir -p "$OUTPUT_ROOT"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "ERROR: conda.sh not found: $CONDA_SH" >&2
  exit 2
fi

# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "$CONDA_ENV"

export CUDA_VISIBLE_DEVICES
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION

write_missing_tasks() {
  python - "$OUTPUT_ROOT" <<'PY'
from pathlib import Path
import re
import sys

out = Path(sys.argv[1])
variants = [
    ("dbp_no_er", Path("logs/11_train_pvp_dbp_critic_seed104_50000_20260521_204732/pvp_dacer_20260521_204744_s0")),
    ("full", Path("logs/12_train_doveer_er_seed104_50000_20260522_104112/pvp_dacer_20260522_104125_s0")),
    ("no_cpc", Path("logs/13_train_doveer_er_no_pv_constraint_seed104_50000_20260522_152214/pvp_dacer_20260522_152226_s0")),
]

def step(path: Path) -> int:
    match = re.search(r"policy-(\d+)-", path.name)
    return int(match.group(1)) if match else 10**18

for variant, policy_dir in variants:
    for policy in sorted(policy_dir.glob("policy-*.pkl"), key=step):
        step_csv = out / variant / policy.stem / "steps_upvgds_final_gate_full.csv"
        if not step_csv.exists():
            print("\t".join([variant, str(policy_dir), policy.name, policy.stem]))
PY
}

run_one() {
  local variant="$1"
  local policy_dir="$2"
  local policy_name="$3"
  local policy_stem="$4"
  local out_dir="$OUTPUT_ROOT/$variant/$policy_stem"
  local step_csv="$out_dir/steps_upvgds_final_gate_full.csv"
  mkdir -p "$out_dir"
  if [[ -f "$step_csv" ]]; then
    echo "[skip] $variant $policy_name" >> "$LOG"
    return 0
  fi
  echo "[run] $(date) $variant $policy_name" | tee -a "$LOG"
  echo "running start=$(date)" > "$out_dir/status.txt"
  set +e
  python scripts/eval_pvp_policies_fixed_v3.py \
    --log_dir "$policy_dir" \
    --policy "$policy_name" \
    --output_csv "$out_dir/eval_upvgds_final_gate_full.csv" \
    --episode_output_csv "$out_dir/episodes_upvgds_final_gate_full.csv" \
    --step_output_csv "$step_csv" \
    --reliability_horizon "$RELIABILITY_HORIZON" \
    --maps $MAPS \
    --num_episodes "$NUM_EPISODES" \
    --step_diagnostics_stride "$STEP_DIAGNOSTICS_STRIDE" \
    --no_use_render \
    --use_value_guidance \
    --guidance_lambda0 0.5 \
    --guidance_beta_unc 1.0 \
    --guidance_p_decay 1.0 \
    --guidance_grad_clip 1.0 \
    --guidance_mode proxy_value \
    --guidance_q_agg conservative \
    --guidance_injection clean_x0 \
    --guidance_schedule noise_level \
    --guidance_target x0 \
    --use_final_gate \
    --final_gate_mode proxy_prior \
    --final_gate_proxy_compare robust \
    --final_gate_min_proxy_gain 0.0 \
    --final_gate_max_action_shift 0.2 \
    --final_gate_tau_prior 0.02 \
    --final_gate_tau_u 1.0 \
    --final_gate_project_guidance 1 \
    --final_gate_energy_samples "$FINAL_GATE_ENERGY_SAMPLES" \
    > "$out_dir/console.log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" == "0" && -f "$step_csv" ]]; then
    echo "ok rc=$rc end=$(date)" > "$out_dir/status.txt"
    echo "[ok] $(date) $variant $policy_name" | tee -a "$LOG"
    return 0
  fi
  echo "failed rc=$rc end=$(date)" > "$out_dir/status.txt"
  echo "[failed] $(date) $variant $policy_name rc=$rc" | tee -a "$LOG"
  return "$rc"
}

echo "[missing-watch] start $(date) output=$OUTPUT_ROOT workers=$NUM_WORKERS" | tee -a "$LOG"

while true; do
  mapfile -t tasks < <(write_missing_tasks)
  echo "[missing-watch] missing=${#tasks[@]} $(date)" | tee -a "$LOG"
  if [[ "${#tasks[@]}" -eq 0 ]]; then
    break
  fi

  active=0
  for task in "${tasks[@]}"; do
    IFS=$'\t' read -r variant policy_dir policy_name policy_stem <<< "$task"
    run_one "$variant" "$policy_dir" "$policy_name" "$policy_stem" &
    active=$((active + 1))
    if [[ "$active" -ge "$NUM_WORKERS" ]]; then
      wait -n || true
      active=$((active - 1))
    fi
  done
  wait || true
done

python - "$OUTPUT_ROOT" <<'PY' >> "$LOG" 2>&1
from pathlib import Path
import glob
import os
import subprocess
import sys

out = Path(sys.argv[1])
step_csvs = sorted(glob.glob(str(out / "*" / "*" / "steps_upvgds_final_gate_full.csv")))
variants = []
for path in step_csvs:
    rel = os.path.relpath(path, out).split(os.sep)
    variants.append(f"{rel[0]}_{rel[1]}")
print(f"[aggregate] step_csvs={len(step_csvs)}")
if len(step_csvs) != 146:
    raise SystemExit(f"expected 146 step CSVs, found {len(step_csvs)}")
subprocess.run(
    [
        sys.executable,
        "scripts/aggregate_critic_reliability.py",
        "--step_csv",
        *step_csvs,
        "--variant",
        *variants,
        "--output_dir",
        str(out / "summary"),
        "--bins",
        "10",
    ],
    check=True,
)
PY

echo "[missing-watch] complete $(date)" | tee -a "$LOG"
