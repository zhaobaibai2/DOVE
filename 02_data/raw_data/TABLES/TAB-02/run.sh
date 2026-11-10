#!/usr/bin/env bash
set -euo pipefail
TABLE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$TABLE_DIR/../../../.." && pwd)"
mode="${1:-test}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  test) ;;
  calculate|recompute) exec "${PYTHON_BIN:-python}" "$TABLE_DIR/calculate.py" "$@" ;;
  *) echo "Usage: $0 test | calculate" >&2; exit 2 ;;
esac
if [[ -z "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN=python
  if ! "$PYTHON_BIN" -c 'import jax, metadrive' >/dev/null 2>&1; then
    PYTHON_BIN="$HOME/miniconda3/envs/dacer_pvp/bin/python"
  fi
fi
exec "$PYTHON_BIN" "$PROJECT_ROOT/scripts/eval_table02.py" "$@"
