#!/usr/bin/env bash
# Oracle: install the reference solution into the workspace.
set -euo pipefail
SOLUTION_DIR="${SOLUTION_DIR:-/solution}"
cp "$SOLUTION_DIR/fit.py" "${WORKSPACE_DIR:-/app}/fit.py"
