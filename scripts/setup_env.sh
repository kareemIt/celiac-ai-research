#!/usr/bin/env bash
# Recreate the project venv. Run from the project root inside WSL Ubuntu.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh

[ -d .venv ] || uv venv --python 3.12 .venv

if [ -f requirements.txt ]; then
  # Exact pinned reinstall; torch comes from the PyTorch CUDA 13.0 index.
  uv pip install --python .venv -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match
else
  # Blackwell (sm_120) needs a CUDA >= 12.8 torch build.
  uv pip install --python .venv torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
  uv pip install --python .venv transformers accelerate peft bitsandbytes datasets safetensors tokenizers \
    scikit-learn xgboost rdkit pandas numpy scipy matplotlib jupyterlab pytest tqdm psutil nvidia-ml-py PyTDC
  uv pip freeze --python .venv > requirements.txt
fi
