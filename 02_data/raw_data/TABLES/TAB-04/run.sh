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
exec "${PYTHON_BIN:-python}" "$TABLE_DIR/calculate.py" "$@"
