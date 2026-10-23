#!/usr/bin/env bash
# Launch ROLL full-parameter SFT on 8 GPUs.
# Usage: bash training/run_sft.sh <sft-data.jsonl> <output-dir> [exp-name] [config-name]
# Requires ROLL_DIR (our ROLL copy with the "raw" template patch) in configs/server.env or env.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
export SFT_DATA="$(realpath "${1:?sft data jsonl}")"
export SFT_OUT="$(realpath -m "${2:?output dir}")"
export EXP_NAME="${3:-biocompute-rft-qwen3-8b}"
CONFIG_NAME="${4:-rft_sft_qwen3_8b}"
export BASE_MODEL="${BASE_MODEL:-$MODEL_DIR}"
ROLL_DIR="${ROLL_DIR:?set ROLL_DIR to the ROLL checkout}"
PY="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}/bin/python"
# ROLL starts `ray` (and other console scripts) by name, so the env's bin/ must be on PATH
export PATH="$(dirname "$PY"):$PATH"

mkdir -p "$SFT_OUT"
# Hydra resolves config_path relative to examples/start_sft_pipeline.py
CONF_DIR="$ROLL_DIR/examples/biocompute_rft"
mkdir -p "$CONF_DIR"
cp "$REPO_DIR/training/roll_configs/$CONFIG_NAME.yaml" "$CONF_DIR/"
cp "$REPO_DIR/training/roll_configs/$CONFIG_NAME.yaml" "$SFT_OUT/config_used.yaml"
cd "$ROLL_DIR"
export PYTHONPATH="$ROLL_DIR:$ROLL_DIR/mcore_adapter/src:${PYTHONPATH:-}"
"$PY" examples/start_sft_pipeline.py --config_path biocompute_rft --config_name "$CONFIG_NAME" \
  2>&1 | tee "$SFT_OUT/train.log"
