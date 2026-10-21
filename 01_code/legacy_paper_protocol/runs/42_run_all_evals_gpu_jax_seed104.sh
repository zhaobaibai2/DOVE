#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
export PYTHONUNBUFFERED=1

MAPS="${MAPS:-200 201 202 203 204 205 206 207 208 209}"
NUM_EPISODES="${NUM_EPISODES:-2}"
RUN_IDS="${RUN_IDS:-10 11 12 13}"
RUN_TOPK="${RUN_TOPK:-0}"
TOPK="${TOPK:-8}"
ALLOW_MISSING="${ALLOW_MISSING:-1}"
TS="$(date +%Y%m%d_%H%M%S)"
MAP_TAG="${MAPS// /_}"
MASTER_DIR="$REPO_DIR/logs/99_run_all_evals_gpu_jax_seed${MAP_TAG}_${TS}"
mkdir -p "$MASTER_DIR"
MARKER="$MASTER_DIR/start.marker"
touch "$MARKER"

cat > "$MASTER_DIR/manifest.txt" <<EOF
repo_dir=$REPO_DIR
maps=$MAPS
num_episodes=$NUM_EPISODES
run_ids=$RUN_IDS
run_topk=$RUN_TOPK
topk=$TOPK
allow_missing=$ALLOW_MISSING
cuda_visible_devices=$CUDA_VISIBLE_DEVICES
jax_platforms=$JAX_PLATFORMS
xla_python_client_mem_fraction=$XLA_PYTHON_CLIENT_MEM_FRACTION
conda_env=${CONDA_DEFAULT_ENV:-}
python=$(which python)
EOF

cat "$MASTER_DIR/manifest.txt"
check_nvidia_gpu | tee "$MASTER_DIR/nvidia-smi.txt"

python - <<'PY' | tee "$MASTER_DIR/gpu_jax_check.txt"
import torch
print('[torch cuda]', torch.cuda.is_available())
if torch.cuda.is_available():
    print('[torch gpu]', torch.cuda.get_device_name(0))
import jax
print('[jax backend]', jax.default_backend())
print('[jax devices]', jax.devices())
PY

chmod +x runs/40_eval_all_policies_all_modes_seed104.sh
chmod +x runs/41_eval_training_runs_all_policies_seed104.sh
chmod +x scripts/summarize_eval_matrix.py
chmod +x scripts/summarize_episode_mean_std.py

MAPS="$MAPS" \
NUM_EPISODES="$NUM_EPISODES" \
RUN_IDS="$RUN_IDS" \
RUN_TOPK="$RUN_TOPK" \
TOPK="$TOPK" \
ALLOW_MISSING="$ALLOW_MISSING" \
bash runs/41_eval_training_runs_all_policies_seed104.sh 2>&1 | tee "$MASTER_DIR/41_eval_training_runs_all_policies.log"

find logs -maxdepth 1 -type d -name "40_eval_all_policies_all_modes_run*_seed${MAP_TAG}_*" -newer "$MARKER" | sort > "$MASTER_DIR/new_eval_roots.txt"
find logs -type f \( -name 'eval_*.csv' -o -name 'episodes_*.csv' -o -name '*summary*.csv' -o -name '*.md' -o -name 'console.log' \) -newer "$MARKER" | sort > "$MASTER_DIR/new_result_files.txt"

python - <<'PY' "$MASTER_DIR"
import csv
import sys
from pathlib import Path
master = Path(sys.argv[1])
roots = [Path(x.strip()) for x in (master / 'new_eval_roots.txt').read_text().splitlines() if x.strip()]
for filename, output_name in [
    ('best_policy_by_mode.csv', 'all_runs_best_policy_by_mode.csv'),
    ('table_ready_episode_mean_std.csv', 'all_runs_table_ready_episode_mean_std.csv'),
    ('all_modes_all_policies_summary.csv', 'all_runs_all_modes_all_policies_summary.csv'),
]:
    rows = []
    fields = None
    for root in roots:
        p = root / filename
        if not p.is_file():
            continue
        with p.open(newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if fields is None:
                fields = ['eval_root'] + list(reader.fieldnames or [])
            for row in reader:
                row = {'eval_root': str(root), **row}
                rows.append(row)
    if rows and fields:
        out = master / output_name
        with out.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows)
        print(f'[master_csv] {out}')
PY

echo
echo "[done] master_dir=$MASTER_DIR"
echo "[done] new eval roots: $MASTER_DIR/new_eval_roots.txt"
echo "[done] new result files: $MASTER_DIR/new_result_files.txt"
echo "[done] table-ready master csv: $MASTER_DIR/all_runs_table_ready_episode_mean_std.csv"
