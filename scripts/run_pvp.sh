#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="python"
    if ! "$PYTHON_BIN" -c 'import jax, metadrive' >/dev/null 2>&1; then
        if [[ -x "$HOME/miniconda3/envs/dacer_pvp/bin/python" ]]; then
            PYTHON_BIN="$HOME/miniconda3/envs/dacer_pvp/bin/python"
        fi
    fi
fi

cd "$PROJECT_ROOT"

usage() {
    echo "Usage: $0 train [training arguments...] | test [--num-episodes N | --replay]"
    echo "  train  Start training; optional arguments are passed to train_pvp_dacer_metadrive_off.py"
    echo "  test   Evaluate the 3 recorded DOVE checkpoints on map 104 (400 episodes each)"
    echo "         --replay uses the 3 archived episode CSVs without running simulation"
}

case "${1:-}" in
    train)
        shift
        exec "$PYTHON_BIN" "$SCRIPT_DIR/train_pvp_dacer_metadrive_off.py" "$@"
        ;;
    test)
        shift
        exec "$PYTHON_BIN" "$SCRIPT_DIR/eval_dove_table01.py" "$@"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
