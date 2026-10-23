#!/usr/bin/env bash
# Rejection-sampling step: run the agent K times on every validated training instance.
# Usage (as root, vLLM replicas serving): bash training/sample_trajectories.sh <train-tasks-dir> \
#          <oracle-run-dir> <out-dir> [K] [base-urls] [served-model-name]
# Only instances whose reference solution passes (reward 1 in <oracle-run-dir>) are used.
# Agent setting = benchmark configuration F: goal-only instruction, repeat guard, check.py.
# GUIDANCE=method appends guidance.md (resampling of unsolved instances; build_sft_data.py
# trains on those with the goal-only instruction). INSTANCES=a,b,... restricts the instances.
# PRESENCE_PENALTY (default 1.0) curbs the degenerate repetition that cuts replies off at the
# token limit; the benchmark itself is run without it. THINKING=1 samples in Qwen3 thinking
# mode (the reasoning is kept in the trajectory and becomes part of the SFT target).
# Many trials wait for the model at once (WORKERS); at most CPU_SLOTS run commands or grading.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
TASKS="${1:?train tasks dir}"; ORACLE="${2:?oracle run dir}"; OUT="${3:?out dir}"
K="${4:-8}"
URLS="${5:-$(cat "$INTERVIEW_ROOT/logs/vllm_urls.txt")}"
NAME="${6:-${SERVED_MODEL_NAME:-qwen3-8b}}"
PY="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}/bin/python"

valid=$("$PY" - "$ORACLE" <<'PY'
import json, sys
from pathlib import Path
ok = sorted({json.loads(p.read_text())["task"] for p in Path(sys.argv[1]).rglob("result.json")
             if json.loads(p.read_text())["reward"] == 1})
print(",".join(ok))
PY
)
echo "valid instances: $(echo "$valid" | tr ',' '\n' | grep -c .)"
if [ -n "${INSTANCES:-}" ]; then
  valid=$(comm -12 <(echo "$valid" | tr ',' '\n' | sort) <(echo "$INSTANCES" | tr ',' '\n' | sort) | paste -sd,)
  echo "restricted to INSTANCES: $(echo "$valid" | tr ',' '\n' | grep -c .)"
fi
cd "$REPO_DIR"
"$PY" -m harness.runner --tasks-root "$TASKS" --tasks "$valid" --agent llm --trials "$K" \
  --guidance "${GUIDANCE:-goal}" --presence-penalty "${PRESENCE_PENALTY:-1.0}" ${THINKING:+--thinking} --repeat-guard --public-check --sandbox-user "${SANDBOX_USER:-agent}" \
  --model "$NAME" --base-url "$URLS" --workers "${WORKERS:-256}" --cpu-slots "${CPU_SLOTS:-144}" \
  --resume --out "$OUT"
