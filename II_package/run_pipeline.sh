#!/usr/bin/env bash
set -euo pipefail

# Run from package root for relative paths to resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" scripts/run_instance_impact_driver.py "$@"
