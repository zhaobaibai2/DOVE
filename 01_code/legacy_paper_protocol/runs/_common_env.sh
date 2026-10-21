#!/usr/bin/env bash

setup_dacer_env() {
  local conda_sh="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
  local conda_env="${CONDA_ENV:-dacer_pvp}"

  if [[ ! -f "$conda_sh" ]]; then
    echo "ERROR: conda.sh not found: $conda_sh"
    echo "Set CONDA_SH=/path/to/conda.sh or install Miniconda under \$HOME/miniconda3."
    exit 2
  fi

  # shellcheck source=/dev/null
  source "$conda_sh"
  conda activate "$conda_env"

  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
  export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
  export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.8}"
}

check_nvidia_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi not found. These scripts expect an NVIDIA CUDA environment."
    exit 2
  fi
  nvidia-smi
}
