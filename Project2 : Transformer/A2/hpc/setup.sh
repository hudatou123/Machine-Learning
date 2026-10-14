#!/bin/bash
# Run once on the Explorer LOGIN node from inside the A2 directory: bash hpc/setup.sh
set -e
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.11 .venv
uv pip install -p .venv/bin/python -r requirements.txt pip
uv pip install -p .venv/bin/python -e .
[ -f data/tokens.npz ] || curl -L https://huggingface.co/datasets/yimingzhang/llms-hw2/resolve/main/tokens.npz -o data/tokens.npz
.venv/bin/python -c "import torch; print('torch', torch.__version__)"
echo "setup done"
