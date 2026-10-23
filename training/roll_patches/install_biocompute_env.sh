#!/usr/bin/env bash
# Install the BioCompute-ARB environment into a ROLL checkout and register it with gem as
# "biocompute". Idempotent: re-running refreshes the copied env code.
# Usage: bash training/roll_patches/install_biocompute_env.sh <roll-dir>
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
ROLL_DIR="${1:?roll dir}"
ENV_PKG="$ROLL_DIR/roll/pipeline/agentic/env/biocompute"
REGISTRY="$ROLL_DIR/roll/pipeline/agentic/env/__init__.py"
mkdir -p "$ENV_PKG"
cp "$REPO_DIR/training/roll_env/biocompute_env.py" "$ENV_PKG/env.py"
echo "from roll.pipeline.agentic.env.biocompute.env import BioComputeEnv" > "$ENV_PKG/__init__.py"
LINE='gem.register("biocompute", entry_point="roll.pipeline.agentic.env.biocompute:BioComputeEnv")'
grep -qF "$LINE" "$REGISTRY" || printf '\n%s\n' "$LINE" >> "$REGISTRY"
echo "installed biocompute env into $ENV_PKG"
# Memory-efficient log-probs (full-vocab fp32 logits of 32k-token micro-batches run out of memory)
"${PYTHON:-python}" "$REPO_DIR/training/roll_patches/patch_chunked_logprobs.py" "$ROLL_DIR"
# Async batch collection without a per-round barrier (one slow episode stalled every step)
"${PYTHON:-python}" "$REPO_DIR/training/roll_patches/patch_rollout_get_batch.py" "$ROLL_DIR"
