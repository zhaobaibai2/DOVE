#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

RUN_DIR="${RUN_DIR:-}"
MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
MAX_POLICIES="${MAX_POLICIES:-0}"

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$(find logs/11_train_pvp_dbp_critic_seed104_50000_* -type d -name 'pvp_dacer_*_s0' | sort | tail -n 1)"
fi

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: cannot find 11 run dir. Set RUN_DIR=/path/to/pvp_dacer_xxx_s0"
  exit 2
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${REPO_DIR}/logs/31_eval_11_all_policies_vanilla_seed104_${TS}"
mkdir -p "$OUT_DIR"

POLICY_COUNT="$(find "$RUN_DIR" -maxdepth 1 \( -name 'policy-*.pkl' -o -name 'final_policy_*.pkl' -o -name 'stage1b_offline_bc_policy.pkl' \) ! -name '*.meta.pkl' | wc -l | tr -d ' ')"

echo "[run] 31_eval_11_all_policies_vanilla_seed104_${TS}"
echo "[meaning] Evaluate all policy checkpoints from run 11 with vanilla no-render autonomous rollout."
echo "[run_dir] $RUN_DIR"
echo "[maps] $MAPS"
echo "[num_episodes] $NUM_EPISODES"
echo "[policy_count] $POLICY_COUNT"
echo "[out] $OUT_DIR"
echo "[env] conda=$CONDA_DEFAULT_ENV python=$(which python)"
check_nvidia_gpu

if [[ "$MAX_POLICIES" != "0" ]]; then
  echo "ERROR: MAX_POLICIES is not supported by this wrapper because eval_pvp_policies_fixed_v3.py scans the whole directory. Use NUM_EPISODES=1 for a faster smoke run."
  exit 2
fi

python -u scripts/eval_pvp_policies_fixed_v3.py \
  --log_dir "$RUN_DIR" \
  --output_csv "$OUT_DIR/eval_11_all_policies_vanilla.csv" \
  --tensorboard_dir "$OUT_DIR/tb" \
  --maps $MAPS \
  --num_episodes "$NUM_EPISODES" \
  --no_use_render \
  2>&1 | tee "$OUT_DIR/console.log"

python - <<'PY' "$OUT_DIR/eval_11_all_policies_vanilla.csv" "$OUT_DIR"
import csv
import math
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
rows = list(csv.DictReader(src.open(newline='', encoding='utf-8')))

def f(row, key):
    try:
        v = float(row.get(key, 'nan'))
        return v if math.isfinite(v) else float('nan')
    except Exception:
        return float('nan')

def step_of(name):
    import re
    if name == 'stage1b_offline_bc_policy.pkl':
        return -1
    m = re.search(r'policy-(\d+)-', name)
    if m:
        return int(m.group(1))
    m = re.search(r'final_policy_(\d+)', name)
    if m:
        return int(m.group(1))
    return 10**18

compact = []
for r in rows:
    name = r.get('policy_name') or r.get('policy', '')
    compact.append({
        'rank_key_success': f(r, 'avg_success'),
        'rank_key_reward': f(r, 'avg_reward'),
        'step': step_of(Path(name).name),
        'policy': name,
        'avg_success': r.get('avg_success', ''),
        'avg_reward': r.get('avg_reward', ''),
        'avg_crash': r.get('avg_crash', ''),
        'avg_out_of_route': r.get('avg_out_of_route', ''),
        'avg_timeout': r.get('avg_timeout', ''),
        'avg_route_completion': r.get('avg_route_completion', ''),
        'avg_closed_loop_distance': r.get('avg_closed_loop_distance', ''),
        'avg_speed_mean': r.get('avg_speed_mean', ''),
        'avg_inference_latency_mean': r.get('avg_inference_latency_mean', ''),
    })

compact.sort(key=lambda r: (-(r['rank_key_success'] if math.isfinite(r['rank_key_success']) else -1), -(r['rank_key_reward'] if math.isfinite(r['rank_key_reward']) else -1), r['step']))
summary_csv = out_dir / 'eval_11_all_policies_vanilla_summary.csv'
fields = ['policy', 'step', 'avg_success', 'avg_reward', 'avg_crash', 'avg_out_of_route', 'avg_timeout', 'avg_route_completion', 'avg_closed_loop_distance', 'avg_speed_mean', 'avg_inference_latency_mean']
with summary_csv.open('w', newline='', encoding='utf-8') as fobj:
    w = csv.DictWriter(fobj, fieldnames=fields)
    w.writeheader()
    for r in compact:
        w.writerow({k: r[k] for k in fields})

summary_md = out_dir / 'eval_11_all_policies_vanilla_top20.md'
with summary_md.open('w', encoding='utf-8') as fobj:
    fobj.write('# Top 20 run-11 vanilla checkpoint eval\n\n')
    fobj.write('| rank | step | policy | success | reward | crash | out_of_route | timeout | route_completion |\n')
    fobj.write('|---:|---:|---|---:|---:|---:|---:|---:|---:|\n')
    for i, r in enumerate(compact[:20], 1):
        fobj.write(f"| {i} | {r['step']} | `{Path(r['policy']).name}` | {r['avg_success']} | {r['avg_reward']} | {r['avg_crash']} | {r['avg_out_of_route']} | {r['avg_timeout']} | {r['avg_route_completion']} |\n")

print(f'[summary_csv] {summary_csv}')
print(f'[summary_md] {summary_md}')
print('[top10]')
for i, r in enumerate(compact[:10], 1):
    print(f"{i:02d} step={r['step']:>6} success={r['avg_success']} reward={r['avg_reward']} policy={Path(r['policy']).name}")
PY

echo "[done] Full CSV: $OUT_DIR/eval_11_all_policies_vanilla.csv"
echo "[done] Summary CSV: $OUT_DIR/eval_11_all_policies_vanilla_summary.csv"
echo "[done] Top20 Markdown: $OUT_DIR/eval_11_all_policies_vanilla_top20.md"
