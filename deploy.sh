#!/usr/bin/env bash
# Deploy the pod server to a RunPod GPU pod over SSH and (re)start it as a real training pod:
#   the tunnel agent listens on 8765 (expose 8765/http on the pod) and the real RF-DETR trainer runs behind it.
#
#   runpod/deploy.sh "root@<ip> -p <port>" [-i ~/.ssh/id_ed25519]      (or: make deploy-pod POD_SSH="root@<ip> -p <port>")
#   VLM=1 runpod/deploy.sh ...      also install vLLM (~/vllm-venv) and download Qwen3.5-9B (~/hf) for the pod's VLM
#                                   (24 GB GPU, 50 GB+ container disk; gone after a pod restart, like the rest)
#
# No secrets here: TUNNEL_KEY, TRAINER_KEY and HF_TOKEN (read-only) are the pod's own environment variables, set in
# the RunPod console; server.py reads them from the container's start environment. The pod has no volume, so run this
# again after every pod restart.
set -euo pipefail
DEST="${1:?usage: deploy.sh \"root@<ip> -p <port>\" [ssh options]}"
shift
HERE="$(cd "$(dirname "$0")" && pwd)"
read -r -a SSH_ARGS <<< "$DEST"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20 "${SSH_ARGS[@]}" "$@")

echo "→ copying runpod/ to ~/runpod on the pod"
COPYFILE_DISABLE=1 tar -C "$HERE" --no-xattrs --exclude .venv --exclude __pycache__ --exclude .pytest_cache \
    --exclude tests -czf - . \
  | "${SSH[@]}" 'mkdir -p ~/runpod && tar -C ~/runpod -xzf -'

echo "→ copying the gateway's detector (the labeller runs the same RF-DETR code as the gateway)"
"${SSH[@]}" 'mkdir -p ~/runpod/shared && cat > ~/runpod/shared/detector.py' < "$HERE/../gateway/labelling/detector.py"

echo "→ installing (a venv that keeps the image's CUDA torch)"
"${SSH[@]}" 'set -e; cd ~/runpod
  [ -x .venv/bin/python ] || python3 -m venv --system-site-packages .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements-train.txt
  .venv/bin/python -c "import torch, rfdetr; print(\"  torch\", torch.__version__, \"cuda\", torch.cuda.is_available(), \"· rfdetr\", rfdetr.__version__ if hasattr(rfdetr, \"__version__\") else \"ok\")"'

if [ "${VLM:-0}" = 1 ]; then
  echo "→ the VLM: vLLM in ~/vllm-venv, Qwen3.5-9B in ~/hf (started on demand by the pod server)"
  "${SSH[@]}" 'set -e
    free=$(df -Pk ~ | awk "NR==2 {print int(\$4 / 1048576)}")
    echo "  ${free} GB free in ~ (vLLM ~10 GB + the model ~19 GB)"
    [ -x ~/vllm-venv/bin/python ] || python3 -m venv ~/vllm-venv
    ~/vllm-venv/bin/pip install -q --upgrade pip
    ~/vllm-venv/bin/pip install -q -r ~/runpod/requirements-vlm.txt
    ~/vllm-venv/bin/python -c "import vllm, torch; print(\"  vllm\", vllm.__version__, \"· torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
    HF_HOME=~/hf ~/vllm-venv/bin/python -c "from huggingface_hub import snapshot_download; print(\"  model in\", snapshot_download(\"Qwen/Qwen3.5-9B\"))"'
fi

echo "→ (re)starting the training pod server (tunnel agent listening on 8765, real trainer)"
"${SSH[@]}" 'cd ~/runpod
  if [ -f pod.pid ] && kill -0 "$(cat pod.pid)" 2>/dev/null; then kill "$(cat pod.pid)"; sleep 3; fi
  pkill -f "[v]llm serve" || true   # a vLLM left from an older server (the pattern does not match this shell)
  nohup .venv/bin/python server.py training --listen 8765 --no-dial --trainer rfdetr > pod.log 2>&1 < /dev/null &
  echo $! > pod.pid
  sleep 6
  tail -n 5 pod.log
  code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/tunnel || true)
  echo "  127.0.0.1:8765/tunnel answers HTTP $code without a key (403 = listening, key required)"'
