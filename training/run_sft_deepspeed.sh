#!/usr/bin/env bash
# Full-parameter SFT on 8 GPUs with DeepSpeed ZeRO-3 (training/sft_deepspeed.py).
# Usage: [DS_CONFIG=training/ds_zero3_offload.json] bash training/run_sft_deepspeed.sh \
#          <sft-data.jsonl> <output-dir> [extra args...]
# DS_CONFIG defaults to ds_zero3.json; ds_zero3_offload.json keeps optimizer states in CPU RAM.
# Used instead of ROLL's SFT pipeline: our ROLL copy needs megatron-core>=0.15 and
# TransformerEngine, while the shared env has megatron-core 0.12 and no TransformerEngine.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
DATA="$(realpath "${1:?sft data jsonl}")"
OUT="$(realpath -m "${2:?output dir}")"
shift 2
PY="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}/bin/python"
export PATH="$(dirname "$PY"):$PATH"
# CPU Adam (optimizer offload) is JIT-compiled; host CUDA 12.9 vs torch cu128 differ only in minor version
export DS_SKIP_CUDA_CHECK=1
mkdir -p "$OUT"
DS_CONFIG="${DS_CONFIG:-training/ds_zero3.json}"
cp "$REPO_DIR/$DS_CONFIG" "$OUT/ds_config_used.json"
cd "$REPO_DIR"
"$PY" -m torch.distributed.run --nproc_per_node "${NGPUS:-8}" training/sft_deepspeed.py \
  --model "${BASE_MODEL:-$MODEL_DIR}" --data "$DATA" --out "$OUT" \
  --deepspeed "$DS_CONFIG" "$@" 2>&1 | tee "$OUT/train.log"
