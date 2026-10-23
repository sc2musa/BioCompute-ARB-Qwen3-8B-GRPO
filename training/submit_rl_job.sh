#!/usr/bin/env bash
# Entry point of a platform training job (one node, 8 GPUs, fresh container) for the RL run.
# The job command only needs to call this script:
#   bash <repo>/training/submit_rl_job.sh <policy-model-dir> <output-dir> [config-name]
# It prepares what a fresh container lacks (sandbox user, operator build cache), keeps any
# earlier output directory (renamed, never overwritten) and starts training/run_rl.sh.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
POLICY="${1:?policy model dir}"
OUT="$(realpath -m "${2:?output dir}")"
CONFIG_NAME="${3:-rl_reinforcepp_biocompute}"

echo "[job] host=$(hostname) start=$(date -u +%FT%TZ)"
n_gpus=$(nvidia-smi -L | wc -l)
if [[ "$n_gpus" -ne 8 ]]; then
  echo "[job] expected 8 GPUs, found $n_gpus" >&2
  exit 1
fi
# Unprivileged user that runs agent commands and graded submissions (as in scripts/setup_env.sh)
if ! id "${SANDBOX_USER:-agent}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "${SANDBOX_USER:-agent}"
fi
# DeepSpeed's CPU Adam is JIT-compiled once; keep the build on shared storage across jobs
export TORCH_EXTENSIONS_DIR="$INTERVIEW_ROOT/cache/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"
if [[ -e "$OUT" ]]; then
  prev="$OUT.prev_$(date -u +%m%d_%H%M)"
  echo "[job] keeping earlier output: $OUT -> $prev"
  mv "$OUT" "$prev"
fi
exec bash "$REPO_DIR/training/run_rl.sh" "$POLICY" "$OUT" "$CONFIG_NAME"
