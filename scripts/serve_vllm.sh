#!/usr/bin/env bash
# Serve the baseline model with an OpenAI-compatible API (single GPU).
# Usage: bash scripts/serve_vllm.sh   (reads configs/server.env; logs to $INTERVIEW_ROOT/logs)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
ENV_DIR="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}"
MODEL_DIR="${MODEL_DIR:-$INTERVIEW_ROOT/models/${MODEL_ID##*/}}"
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
# bin/vllm starts with "#!/usr/bin/env python3.x": the env's bin/ must come first on PATH
export PATH="$ENV_DIR/bin:$PATH"

exec "$ENV_DIR/bin/vllm" serve "$MODEL_DIR" \
  --served-model-name "${SERVED_MODEL_NAME:-qwen3-8b}" \
  --host 127.0.0.1 --port "${VLLM_PORT:-8000}" \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --reasoning-parser qwen3 \
  --seed 0
