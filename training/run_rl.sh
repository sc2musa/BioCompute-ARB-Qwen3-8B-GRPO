#!/usr/bin/env bash
# Launch ROLL agentic RL (REINFORCE++, no KL) on BioCompute-ARB training instances, 8 GPUs.
# Usage: [RL_STEPS=100] [RL_TASKS="seq-align-affine reaction-network-stiff"] \
#          bash training/run_rl.sh <policy-model-dir> <output-dir> [config-name]
# Training instances come from training/make_train_tasks.py (TRAIN_TASKS, default
# $INTERVIEW_ROOT/train_tasks); hidden test cases of tasks/ are never used.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
export POLICY_MODEL="$(realpath "${1:?policy model dir}")"
export RL_OUT="$(realpath -m "${2:?output dir}")"
CONFIG_NAME="${3:-rl_reinforcepp_biocompute}"
export REPO_DIR
export TRAIN_TASKS="${TRAIN_TASKS:-$INTERVIEW_ROOT/train_tasks}"
export ENV_WORK_ROOT="${ENV_WORK_ROOT:-$RL_OUT/env_work}"
export RL_STEPS="${RL_STEPS:-100}"
RL_TASKS="${RL_TASKS:-seq-align-affine reaction-network-stiff}"
ROLL_DIR="${ROLL_DIR:?set ROLL_DIR to the ROLL checkout}"
PY="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}/bin/python"
# ROLL starts `ray` (and other console scripts) by name, so the env's bin/ must be on PATH
export PATH="$(dirname "$PY"):$PATH"
export DS_SKIP_CUDA_CHECK=1

mkdir -p "$RL_OUT" "$ENV_WORK_ROOT"
# Training instance list: every generated instance of the selected task families
export TRAIN_TASK_LIST="$RL_OUT/train_tasks.txt"
for t in $RL_TASKS; do
  find "$TRAIN_TASKS" -maxdepth 1 -name "${t}__train*" -printf '%f\n'
done | sort > "$TRAIN_TASK_LIST"
echo "training instances: $(wc -l < "$TRAIN_TASK_LIST") ($RL_TASKS)"

PYTHON="$PY" bash "$REPO_DIR/training/roll_patches/install_biocompute_env.sh" "$ROLL_DIR"
# Hydra resolves config_path relative to examples/start_agentic_pipeline.py
CONF_DIR="$ROLL_DIR/examples/biocompute_rl"
mkdir -p "$CONF_DIR"
cp "$REPO_DIR/training/roll_configs/$CONFIG_NAME.yaml" "$CONF_DIR/"
cp "$REPO_DIR/training/roll_configs/$CONFIG_NAME.yaml" "$RL_OUT/config_used.yaml"
cd "$ROLL_DIR"
export PYTHONPATH="$ROLL_DIR:$ROLL_DIR/mcore_adapter/src:${PYTHONPATH:-}"
"$PY" examples/start_agentic_pipeline.py --config_path biocompute_rl --config_name "$CONFIG_NAME" \
  2>&1 | tee "$RL_OUT/train.log"
