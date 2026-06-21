#!/bin/bash
# DICE / AgentNet local env setup on ece-drf95318s
# venv lives on NFS home (~/dice-env) to avoid filling the small local root disk.
set -e
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$HOME/dice-env"
cd "$HOME/dice-env"
echo "[$(date)] === creating venv (python 3.10) ==="
uv venv --python 3.10 .venv
source .venv/bin/activate
echo "[$(date)] === installing vllm + runtime deps (this is the long part) ==="
uv pip install vllm openai sentence-transformers pyyaml colorlog
echo "[$(date)] === verifying torch/vllm/cuda ==="
python -c "import vllm, torch; print('OK vllm', vllm.__version__, 'torch', torch.__version__, 'cuda_avail', torch.cuda.is_available(), 'ngpu', torch.cuda.device_count())"
echo "=== SETUP DONE ==="
