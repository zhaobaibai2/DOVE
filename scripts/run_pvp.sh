#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$PROJECT_ROOT/02_data/models/final/simulation/DOVE/policy-13000-25998.pkl"

cd "$PROJECT_ROOT"

usage() {
    echo "Usage: $0 train [training arguments...] | test"
    echo "  train  Start training; optional arguments are passed to train_pvp_dacer_metadrive_off.py"
    echo "  test   Evaluate the saved DOVE policy on map 104 (5 episodes by default)"
}

case "${1:-}" in
    train)
        shift
        exec "$PYTHON_BIN" "$SCRIPT_DIR/train_pvp_dacer_metadrive_off.py" "$@"
        ;;
    test)
        if [[ ! -f "$MODEL_PATH" ]]; then
            echo "Checkpoint not found: $MODEL_PATH" >&2
            exit 1
        fi
        exec "$PYTHON_BIN" "$SCRIPT_DIR/eval_pvp_policies_fixed_v3.py" \
            --log_dir "$MODEL_PATH" \
            --maps 104 \
            --num_episodes "${NUM_EPISODES:-5}"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
