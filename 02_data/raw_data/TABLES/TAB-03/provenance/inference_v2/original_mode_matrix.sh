#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/aaa/data/target/Backup/exp/expB0/0108/DACER-te-new0428/test0508"
STAGE="${1:-all}"
CURRENT_FILE="$ROOT/logs2/energy_sweep_current"
if [[ -n "${SWEEP_DIR:-}" ]]; then
  OUT_ROOT="$SWEEP_DIR"
elif [[ "$STAGE" == "train" || "$STAGE" == "train_guarded" || "$STAGE" == "all" ]]; then
  OUT_ROOT="$ROOT/logs2/energy_lambda_sweep_$(date +%Y%m%d_%H%M%S)"
else
  if [[ ! -f "$CURRENT_FILE" ]]; then
    echo "No current sweep directory found. Run train first, or set SWEEP_DIR=/path/to/logs2/energy_lambda_sweep_*." >&2
    exit 2
  fi
  OUT_ROOT="$(cat "$CURRENT_FILE")"
fi

FIRST_ENERGY="${FIRST_ENERGY:-2.0}"
REMAINING_ENERGY_VALUES="${REMAINING_ENERGY_VALUES:-0.5}"
ENERGY_VALUES="${ENERGY_VALUES:-$FIRST_ENERGY $REMAINING_ENERGY_VALUES}"
ER_MARGIN="${ER_MARGIN:-0.05}"
SEED="${SEED:-0}"
MAPS="${MAPS:-104}"
NUM_EPISODES="${NUM_EPISODES:-5}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TRAIN_RENDER="${TRAIN_RENDER:-1}"

DEMO_ROOT="${DEMO_ROOT:-$ROOT/logs2/energy_lambda_sweep_20260611_183121/train_lambda_er_0p0_margin_0p05/pvp_dacer_20260611_183134_s0/data}"
PAIR_DATA_ROOT="${PAIR_DATA_ROOT:-$DEMO_ROOT}"

export CUDA_VISIBLE_DEVICES
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.5}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"
printf "%s\n" "$OUT_ROOT" > "$CURRENT_FILE"

if [[ "$STAGE" == "train" || "$STAGE" == "train_guarded" || "$STAGE" == "all" || ! -f "$OUT_ROOT/README.md" ]]; then
  cat > "$OUT_ROOT/README.md" <<EOF_README
# EnergyRank lambda sweep

- created_at: $(date '+%Y-%m-%d %H:%M:%S')
- root: $ROOT
- output_root: $OUT_ROOT
- sweep_parameter: lambda_er
- sweep_values: $ENERGY_VALUES
- guarded_train_order: $ENERGY_VALUES
- er_margin: $ER_MARGIN
- previous_full_value: lambda_er=1.0, er_margin=0.05
- previous_no_energy_value: lambda_er=0.0, use_energy_rank=false
- stage1a_demo_collection: skipped; uses fixed collected data
- fixed_demo_root: $DEMO_ROOT
- stage1b_updates: 30000
- pair_data_root: $PAIR_DATA_ROOT
- maps: $MAPS
- num_episodes_per_map: $NUM_EPISODES

Directory layout:

- train_lambda_er_<value>_margin_<margin>/pvp_dacer_*/: training args, checkpoints, TensorBoard, final bundle.
- pair_lambda_er_<value>_margin_<margin>/: PairSat_E, V_E, IAR, energy gaps.
- eval_lambda_er_<value>_margin_<margin>/<mode>/<policy>/: closed-loop episode CSV, aggregate CSV, optional step CSV.
- manifest.tsv: one row per sweep value with resolved train/eval paths.
- guarded_checks/<value>.check.log: post-training integrity checks before the next lambda starts.
EOF_README
fi

if [[ "$STAGE" == "train" || "$STAGE" == "train_guarded" || "$STAGE" == "all" || ! -f "$OUT_ROOT/manifest.tsv" ]]; then
  printf "lambda_er\ter_margin\tuse_energy_rank\ttrain_log_dir\tpair_dir\teval_dir\n" > "$OUT_ROOT/manifest.tsv"
fi

safe_value() {
  printf "%s" "$1" | sed 's/-/m/g; s/\./p/g'
}

latest_run_dir() {
  local log_root="$1"
  find "$log_root" -mindepth 1 -maxdepth 1 -type d -name 'pvp_dacer_*' | sort | tail -1
}

train_one() {
  local lambda_er="$1"
  local safe
  safe="$(safe_value "$lambda_er")"
  local train_root="$OUT_ROOT/train_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  mkdir -p "$train_root"

  local energy_flags=()
  if python - "$lambda_er" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) > 0 else 1)
PY
  then
    energy_flags=(--use_energy_rank --lambda_er "$lambda_er")
  else
    energy_flags=(--lambda_er 0.0)
  fi

  local render_flags=()
  if [[ "$TRAIN_RENDER" == "1" || "$TRAIN_RENDER" == "true" || "$TRAIN_RENDER" == "yes" ]]; then
    render_flags=(--use_render)
  fi

  echo "[train] lambda_er=$lambda_er er_margin=$ER_MARGIN -> $train_root"
  python scripts/train_pvp_dacer_metadrive_off.py \
    --total_step 50000 \
    --seed "$SEED" \
    --start_seed 104 \
    --num_scenarios 1 \
    --traffic_density 0.06 \
    --demo_root "$DEMO_ROOT" \
    --log_dir "$train_root" \
    --save_policy_every 1000 \
    --demo_passes_per_map 5 \
    --stage1b_updates 30000 \
    --stage1b_lambda_bc 50.0 \
    --stage2_critic_warmup_steps 1000 \
    --stage2_rl_warmup_steps 1000 \
    --stage2_rl_ramp_steps 2000 \
    --lambda_pv 1.0 \
    --B 2.0 \
    --lambda_bc 12.0 \
    --lambda_qreg 0.05 \
    --lambda_rl 0.1 \
    --lambda_reg 2.0 \
    --critic_objective cost \
    --lambda_pv_constraint 1.0 \
    --pv_constraint_margin 0.1 \
    --er_margin "$ER_MARGIN" \
    --er_min_action_gap 0.03 \
    --er_positive_action behavior \
    "${render_flags[@]}" \
    "${energy_flags[@]}" \
    2>&1 | tee "$train_root/train_lambda_er_${safe}.log"
}

verify_train_one() {
  local lambda_er="$1"
  local safe train_root run_dir check_dir check_log
  safe="$(safe_value "$lambda_er")"
  train_root="$OUT_ROOT/train_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  run_dir="$(latest_run_dir "$train_root")"
  check_dir="$OUT_ROOT/guarded_checks"
  check_log="$check_dir/lambda_er_${safe}.check.log"
  mkdir -p "$check_dir"
  test -n "$run_dir"

  echo "[check] lambda_er=$lambda_er run=$run_dir" | tee "$check_log"
  python - "$run_dir" "$lambda_er" "$ER_MARGIN" <<'PY' 2>&1 | tee -a "$check_log"
import json
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
lambda_er = float(sys.argv[2])
er_margin = float(sys.argv[3])
errors = []

def require(cond, msg):
    if not cond:
        errors.append(msg)

args_path = run_dir / "args.json"
require(args_path.exists(), f"missing {args_path}")
args = {}
if args_path.exists():
    args = json.loads(args_path.read_text())
    require(int(args.get("total_step", -1)) == 50000, "total_step is not 50000")
    require(int(args.get("stage1b_updates", -1)) == 30000, "stage1b_updates is not 30000")
    require(int(args.get("demo_passes_per_map", -1)) == 5, "demo_passes_per_map is not 5")
    require(not bool(args.get("manual_control", False)), "manual_control should be false for Stage2 agent driving")
    require(abs(float(args.get("er_margin", -1.0)) - er_margin) < 1e-9, "er_margin mismatch")
    require(abs(float(args.get("lambda_er", -1.0)) - lambda_er) < 1e-9, "lambda_er mismatch")
    require(bool(args.get("use_render", False)), "use_render is not true")
    require(str(args.get("demo_root", "")) == str(Path("$DEMO_ROOT")), "demo_root does not match fixed collected data")
    require(not bool(args.get("force_demo_collection", False)), "force_demo_collection should be false")
    if lambda_er > 0:
        require(bool(args.get("use_energy_rank", False)), "use_energy_rank is not true")
    else:
        require(not bool(args.get("use_energy_rank", False)), "lambda_er=0 should not use_energy_rank")

require((run_dir / "stage1b_offline_bc_policy.pkl").exists(), "missing stage1b_offline_bc_policy.pkl")
require((run_dir / "final_policy_50000.pkl").exists(), "missing final_policy_50000.pkl")
require((run_dir / "tb").exists(), "missing tb directory")
require(any((run_dir / "tb").glob("events.out.tfevents.*")), "missing root TensorBoard event file")

demo_root = Path("$DEMO_ROOT")
demo_files = list(demo_root.rglob("*.pkl")) if demo_root.exists() else []
require(len(demo_files) >= 5, f"expected at least 5 fixed demo/data pkl files, got {len(demo_files)} from {demo_root}")

try:
    tags = set()
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    for event_file in sorted((run_dir / "tb").glob("events.out.tfevents.*")):
        for event in EventFileLoader(str(event_file)).Load():
            for value in event.summary.value:
                if value.tag:
                    tags.add(value.tag)
    for tag in [
        "train_cost/episode_cost",
        "train_cost/native_cost",
        "train_cost/takeover_cost",
        "train_cost/intervention_rate",
        "train_cost/intervention_count",
        "episode/cost",
        "episode/native_cost",
        "episode/takeover_cost",
    ]:
        require(tag in tags, f"missing TensorBoard scalar tag {tag}")
except Exception as exc:
    errors.append(f"failed to inspect TensorBoard scalars: {exc}")

if errors:
    print("[check:FAIL]")
    for err in errors:
        print(f"- {err}")
    raise SystemExit(1)

print("[check:OK]")
print(f"run_dir={run_dir}")
print(f"demo_or_data_files={len(demo_files)}")
PY
}

pair_one() {
  local lambda_er="$1"
  local safe train_root run_dir pair_dir
  safe="$(safe_value "$lambda_er")"
  train_root="$OUT_ROOT/train_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  run_dir="$(latest_run_dir "$train_root")"
  pair_dir="$OUT_ROOT/pair_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  mkdir -p "$pair_dir"
  test -n "$run_dir"

  local data_root="${PAIR_DATA_ROOT:-$DEMO_ROOT}"
  echo "[pair] lambda_er=$lambda_er run=$run_dir data_root=$data_root"
  python scripts/run_intervention_pair_diagnostics.py \
    --log_dir "$run_dir" \
    --policy final_policy_50000.pkl \
    --data_root "$data_root" \
    --output_dir "$pair_dir" \
    --max_pairs 512 \
    --batch_size 256 \
    --energy_samples 4 \
    --er_margin "$ER_MARGIN" \
    --min_action_gap 0.03 \
    --positive_action behavior \
    --seed "$SEED" \
    2>&1 | tee "$pair_dir/pair_lambda_er_${safe}.log"
}

eval_policy_mode() {
  local run_dir="$1"
  local policy="$2"
  local mode="$3"
  local out_dir="$4"
  mkdir -p "$out_dir"

  local common=(
    --log_dir "$run_dir"
    --policy "$policy"
    --maps $MAPS
    --num_episodes "$NUM_EPISODES"
    --no_use_render
    --output_csv "$out_dir/eval_${mode}.csv"
    --episode_output_csv "$out_dir/episodes_${mode}.csv"
    --tensorboard_dir "$out_dir/tb"
  )

  case "$mode" in
    vanilla)
      python scripts/eval_pvp_policies_fixed_v3.py "${common[@]}"
      ;;
    upvgds_no_gate)
      python scripts/eval_pvp_policies_fixed_v3.py "${common[@]}" \
        --use_value_guidance \
        --guidance_lambda0 0.5 \
        --guidance_beta_unc 1.0 \
        --guidance_p_decay 1.0 \
        --guidance_grad_clip 1.0 \
        --guidance_mode proxy_value \
        --guidance_target x0 \
        --guidance_step_interval 1 \
        --guidance_injection clean_x0 \
        --guidance_schedule noise_level \
        --guidance_critic_params target \
        --guidance_q_agg conservative \
        --critic_objective lower
      ;;
    upvgds_final_gate_full|final_gate_proxy_only|final_gate_prior_only|final_gate_no_disagreement)
      local gate_mode="proxy_prior"
      local tau_u="1.0"
      if [[ "$mode" == "final_gate_proxy_only" ]]; then gate_mode="proxy_only"; fi
      if [[ "$mode" == "final_gate_prior_only" ]]; then gate_mode="prior_only"; fi
      if [[ "$mode" == "final_gate_no_disagreement" ]]; then tau_u="-1.0"; fi
      python scripts/eval_pvp_policies_fixed_v3.py "${common[@]}" \
        --use_value_guidance \
        --use_final_gate \
        --guidance_lambda0 0.5 \
        --guidance_beta_unc 1.0 \
        --guidance_p_decay 1.0 \
        --guidance_grad_clip 1.0 \
        --guidance_mode proxy_value \
        --guidance_target x0 \
        --guidance_step_interval 1 \
        --guidance_injection clean_x0 \
        --guidance_schedule noise_level \
        --guidance_critic_params target \
        --guidance_q_agg conservative \
        --critic_objective lower \
        --final_gate_mode "$gate_mode" \
        --final_gate_proxy_compare robust \
        --final_gate_margin 0.0 \
        --final_gate_min_proxy_gain 0.0 \
        --final_gate_kappa 1.0 \
        --final_gate_beta_prior 1.0 \
        --final_gate_rho_shift 0.0 \
        --final_gate_rho_sat 0.0 \
        --final_gate_max_action_shift 0.2 \
        --final_gate_tau_prior 0.02 \
        --final_gate_tau_u "$tau_u" \
        --final_gate_project_guidance 1 \
        --final_gate_energy_samples 4
      ;;
    topk_rerank)
      python scripts/eval_pvp_policies_fixed_v3.py "${common[@]}" \
        --rerank_topk 8 \
        --guidance_q_agg conservative \
        --critic_objective lower
      ;;
    *)
      echo "unknown mode: $mode" >&2
      return 2
      ;;
  esac
}

eval_one() {
  local lambda_er="$1"
  local safe train_root run_dir eval_dir policy
  safe="$(safe_value "$lambda_er")"
  train_root="$OUT_ROOT/train_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  run_dir="$(latest_run_dir "$train_root")"
  eval_dir="$OUT_ROOT/eval_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  policy="${POLICY:-final_policy_50000.pkl}"
  mkdir -p "$eval_dir"
  test -n "$run_dir"

  local modes="${EVAL_MODES:-vanilla upvgds_no_gate upvgds_final_gate_full final_gate_proxy_only final_gate_prior_only final_gate_no_disagreement topk_rerank}"
  for mode in $modes; do
    echo "[eval] lambda_er=$lambda_er mode=$mode policy=$policy"
    eval_policy_mode "$run_dir" "$policy" "$mode" "$eval_dir/$mode/$policy" \
      2>&1 | tee "$eval_dir/${mode}_${policy%.pkl}.log"
  done
}

record_manifest_one() {
  local lambda_er="$1"
  local safe train_root run_dir pair_dir eval_dir use_er
  safe="$(safe_value "$lambda_er")"
  train_root="$OUT_ROOT/train_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  run_dir="$(latest_run_dir "$train_root" || true)"
  pair_dir="$OUT_ROOT/pair_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  eval_dir="$OUT_ROOT/eval_lambda_er_${safe}_margin_$(safe_value "$ER_MARGIN")"
  use_er="true"
  if python - "$lambda_er" <<'PY'
import sys
raise SystemExit(0 if float(sys.argv[1]) <= 0 else 1)
PY
  then
    use_er="false"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$lambda_er" "$ER_MARGIN" "$use_er" "$run_dir" "$pair_dir" "$eval_dir" >> "$OUT_ROOT/manifest.tsv"
}

if [[ "$STAGE" == "train_guarded" ]]; then
  echo "[guarded] first lambda_er=$FIRST_ENERGY"
  train_one "$FIRST_ENERGY"
  record_manifest_one "$FIRST_ENERGY"
  echo "[guarded] run finished; checking data/TensorBoard before continuing"
  verify_train_one "$FIRST_ENERGY"
  echo "[guarded] check passed; continuing lambda_er list: $REMAINING_ENERGY_VALUES"
  for v in $REMAINING_ENERGY_VALUES; do
    train_one "$v"
    record_manifest_one "$v"
    verify_train_one "$v"
  done
fi

if [[ "$STAGE" == "train" || "$STAGE" == "all" ]]; then
  for v in $ENERGY_VALUES; do train_one "$v"; record_manifest_one "$v"; done
fi

if [[ "$STAGE" == "pair" || "$STAGE" == "all" ]]; then
  for v in $ENERGY_VALUES; do pair_one "$v"; done
fi

if [[ "$STAGE" == "eval" || "$STAGE" == "all" ]]; then
  for v in $ENERGY_VALUES; do eval_one "$v"; done
fi

if [[ "$STAGE" != "train" && "$STAGE" != "train_guarded" && "$STAGE" != "pair" && "$STAGE" != "eval" && "$STAGE" != "all" ]]; then
  echo "Usage: $0 [train_guarded|train|pair|eval|all]" >&2
  exit 2
fi

echo "[done] $OUT_ROOT"
