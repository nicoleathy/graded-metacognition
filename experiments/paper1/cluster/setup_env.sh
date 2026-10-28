#!/bin/bash
# Create the Python environment on the cluster, once, on a node with internet access.
#   PYTHON=python3.12 VENV=$HOME/venvs/paper1 experiments/paper1/cluster/setup_env.sh
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 ...    # only if the GPU driver is too old for the default CUDA build of torch 2.8.0
# The Mac runs used Python 3.12.13, torch 2.8.0 and transformers 4.56.0 (environment/paper1-macos-arm64-freeze.txt). Do not upgrade them.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-$HOME/venvs/paper1}"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), "use Python 3.12: the Mac runs used 3.12.13"'
"$PYTHON" -m venv "$VENV"
source "$VENV/bin/activate"
python -m pip install --upgrade pip
if [ -n "${TORCH_INDEX_URL:-}" ]; then pip install "torch==2.8.0" --index-url "$TORCH_INDEX_URL"; fi
pip install -r requirements-paper1.txt
python - <<'PY'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__, "| cuda", torch.cuda.is_available())
assert torch.__version__.startswith("2.8.0") and transformers.__version__ == "4.56.0", "versions differ from environment/paper1-macos-arm64-freeze.txt"
PY
echo "environment ready: source $VENV/bin/activate"
