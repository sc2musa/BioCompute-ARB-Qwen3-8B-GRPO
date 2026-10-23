#!/usr/bin/env bash
# Oracle: install the reference solution into the workspace.
set -euo pipefail
SOLUTION_DIR="${SOLUTION_DIR:-/solution}"
cp "$SOLUTION_DIR/simulate.py" "${WORKSPACE_DIR:-/app}/simulate.py"
