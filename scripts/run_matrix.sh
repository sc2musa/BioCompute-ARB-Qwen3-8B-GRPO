#!/usr/bin/env bash
# Baseline experiment matrix for the LLM agent (no training).
# Usage (as root, vLLM already serving): [WORKERS=6] bash scripts/run_matrix.sh <out-root> [trials]
# Re-running the same command resumes: trials with a result.json are kept.
#   A_raw          : goal-only,   no thinking, no repeat guard  (plain model + minimal scaffold)
#   B_guard        : goal-only,   no thinking, repeat guard
#   C_method       : method hint, no thinking, repeat guard
#   D_think        : goal-only,   thinking,    repeat guard
#   E_think_method : method hint, thinking,    repeat guard
#   F_check        : goal-only,   no thinking, repeat guard, public self-check script (check.py)
#   G_think_check  : goal-only,   thinking,    repeat guard, public self-check script
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
OUT="${1:?usage: run_matrix.sh <out-root> [trials]}"
TRIALS="${2:-5}"
PY="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}/bin/python"
COMMON=(--agent llm --tasks all --trials "$TRIALS" --sandbox-user "${SANDBOX_USER:-agent}"
        --model "${SERVED_MODEL_NAME:-qwen3-8b}" --base-url "http://127.0.0.1:${VLLM_PORT:-8000}/v1"
        --workers "${WORKERS:-6}" --resume)
declare -A CONFIGS=(
  [A_raw]="--guidance goal"
  [B_guard]="--guidance goal --repeat-guard"
  [C_method]="--guidance method --repeat-guard"
  [D_think]="--guidance goal --repeat-guard --thinking"
  [E_think_method]="--guidance method --repeat-guard --thinking"
  [F_check]="--guidance goal --repeat-guard --public-check"
  [G_think_check]="--guidance goal --repeat-guard --thinking --public-check"
)
cd "$REPO_DIR"
mkdir -p "$OUT"
for name in "${!CONFIGS[@]}"; do
  # shellcheck disable=SC2086
  "$PY" -m harness.runner "${COMMON[@]}" ${CONFIGS[$name]} --out "$OUT/$name" > "$OUT/$name.log" 2>&1 &
done
wait
echo "matrix done: $OUT"
