#!/usr/bin/env bash
# One vLLM server per GPU for high-throughput sampling (data-parallel replicas).
# Usage: [GPUS="0 1"] bash scripts/serve_vllm_multi.sh [n_gpus] [model_dir] [served_name]
# Ports: VLLM_PORT .. VLLM_PORT+n-1; logs: $INTERVIEW_ROOT/logs/vllm_gpu<i>.log
# Prints the comma-separated --base-url list for harness/runner.py.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
N="${1:-8}"
MODEL="${2:-$MODEL_DIR}"
NAME="${3:-${SERVED_MODEL_NAME:-qwen3-8b}}"
BASE_PORT="${VLLM_PORT:-8000}"
VLLM="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}/bin/vllm"
# bin/vllm starts with "#!/usr/bin/env python3.x": the env's bin/ must come first on PATH
export PATH="$(dirname "$VLLM"):$PATH"
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1
urls=()
for ((i = 0; i < N; i++)); do
  port=$((BASE_PORT + i))
  if [[ -n "${GPUS:-}" && " $GPUS " != *" $i "* ]]; then  # GPUS="0 1": (re)start only these
    urls+=("http://127.0.0.1:$port/v1"); continue
  fi
  # Separate compile caches: concurrent replicas sharing one cache dir corrupt each other
  VLLM_CACHE_ROOT="/root/.cache/vllm_gpu$i" CUDA_VISIBLE_DEVICES=$i nohup setsid "$VLLM" serve "$MODEL" \
    --served-model-name "$NAME" --host 127.0.0.1 --port "$port" \
    --max-model-len 40960 --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    --seed "$i" > "$INTERVIEW_ROOT/logs/vllm_gpu$i.log" 2>&1 < /dev/null &
  urls+=("http://127.0.0.1:$port/v1")
done
(IFS=,; echo "${urls[*]}")
