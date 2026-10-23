#!/usr/bin/env bash
# One-time setup on a Linux GPU host: conda env with vLLM, model download, sandbox user.
# Usage: bash scripts/setup_env.sh            (reads configs/server.env)
set -euo pipefail
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_DIR/configs/server.env"
: "${INTERVIEW_ROOT:?set INTERVIEW_ROOT in configs/server.env}"

if [[ -n "${HTTP_PROXY_URL:-}" ]]; then
  export http_proxy="$HTTP_PROXY_URL" https_proxy="$HTTP_PROXY_URL"
  export HTTP_PROXY="$HTTP_PROXY_URL" HTTPS_PROXY="$HTTP_PROXY_URL"
fi
CONDA_DIR="$INTERVIEW_ROOT/miniconda3"
ENV_DIR="${PYTHON_ENV:-$INTERVIEW_ROOT/envs/arb}"
MODEL_DIR="${MODEL_DIR:-$INTERVIEW_ROOT/models/${MODEL_ID##*/}}"
export PIP_CACHE_DIR="$INTERVIEW_ROOT/cache/pip"
export MODELSCOPE_CACHE="$INTERVIEW_ROOT/cache/modelscope"
mkdir -p "$INTERVIEW_ROOT"/{envs,models,cache,runs,logs}

# 1-2. Python env, unless an existing one is given via PYTHON_ENV
if [[ -z "${PYTHON_ENV:-}" ]]; then
  if [[ ! -x "$CONDA_DIR/bin/conda" ]]; then
    installer="${MINICONDA_INSTALLER:-$INTERVIEW_ROOT/cache/Miniconda3-latest-Linux-x86_64.sh}"
    if [[ ! -f "$installer" ]]; then
      curl -fsSL -o "$installer" https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    fi
    bash "$installer" -b -p "$CONDA_DIR"
  fi
  if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    "$CONDA_DIR/bin/python" -m venv "$ENV_DIR"
  fi
  "$ENV_DIR/bin/pip" install -i "${PIP_INDEX_URL:-https://pypi.org/simple}" -r "$REPO_DIR/requirements-server.txt"
fi

# 3. Model weights (skipped when MODEL_DIR already holds a model)
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  "$ENV_DIR/bin/pip" install -i "${PIP_INDEX_URL:-https://pypi.org/simple}" "modelscope>=1.20"
  "$ENV_DIR/bin/modelscope" download --model "$MODEL_ID" --local_dir "$MODEL_DIR"
fi

# 4. Sandbox user for agent commands and submission execution
if ! id "${SANDBOX_USER:-agent}" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "${SANDBOX_USER:-agent}"
fi

"$ENV_DIR/bin/python" -c "import vllm, numpy, scipy, openai; print('vllm', vllm.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__)"
echo "setup done: env=$ENV_DIR model=$MODEL_DIR"
