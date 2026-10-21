#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

source "$REPO_DIR/runs/_common_env.sh"
setup_dacer_env

echo "[env] repo=$REPO_DIR"
echo "[env] conda=$CONDA_DEFAULT_ENV"
echo "[env] python=$(which python)"
echo "[env] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

check_nvidia_gpu

python - <<'PY'
import jax
print("[jax] backend:", jax.default_backend())
print("[jax] devices:", jax.devices())
assert jax.default_backend() == "gpu", "JAX GPU backend is required"
PY
