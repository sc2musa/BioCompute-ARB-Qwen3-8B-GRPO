#!/usr/bin/env bash
# Harbor verifier entrypoint. Paths default to the Harbor layout and can be overridden
# by the local runner (harness/runner.py).
set -euo pipefail
TESTS_DIR="${TESTS_DIR:-/tests}"
"${PYTHON:-python3}" "$TESTS_DIR/grade.py" --workspace "${WORKSPACE_DIR:-/app}" --out "${VERIFIER_LOG_DIR:-/logs/verifier}"
