#!/usr/bin/env bash
# Install (or update) the pod server on a RunPod GPU pod and start it: the code from GitHub, the pod's keys, the
# trainer + RF-DETR pre-labeller, optionally the VLM (vLLM + Qwen3.5-9B), then the server listening on 8765.
#
#   install.sh [--vlm] [--repo URL] [--branch main] [--dir ~/runpod] [--skip-pip] [--no-start]
#
# The keys come on stdin as KEY=value lines (TUNNEL_KEY, TRAINER_KEY, HF_TOKEN), never as arguments, and go to
# <dir>/.env (mode 600). Keys not sent keep their earlier value. With nothing on stdin (a terminal), nothing changes.
#   printf 'TUNNEL_KEY=...\nTRAINER_KEY=...\nHF_TOKEN=...\n' | bash install.sh --vlm
#
# The gateway's Pods page runs this over SSH. Progress lines for it: "::step <name>", "::ok", "::fail <why>",
# "::warn <text>", "::info <k>=<v>", and at the end "::done pod=<id> tunnel=<wss url>".
set -uo pipefail

REPO="https://github.com/devx01/runpod"
BRANCH="main"
DIR="$HOME/runpod"
VLM=0
SKIP_PIP=0
START=1
while [ $# -gt 0 ]; do
  case "$1" in
    --vlm) VLM=1 ;;
    --repo) REPO="$2"; shift ;;
    --branch) BRANCH="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    --skip-pip) SKIP_PIP=1 ;;
    --no-start) START=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "::fail unknown option $1"; exit 2 ;;
  esac
  shift
done

STEP=""
step() { STEP="$1"; echo "::step $1"; }
ok() { echo "::ok"; }
fail() { echo "::fail $STEP: $*"; exit 1; }

# 1 code -------------------------------------------------------------------------------------------------------------
step code
if [ -d "$DIR/.git" ]; then
  git -C "$DIR" remote set-url origin "$REPO" || fail "git remote"
  git -C "$DIR" fetch -q origin "$BRANCH" || fail "git fetch $REPO $BRANCH"
  git -C "$DIR" reset -q --hard "origin/$BRANCH" || fail "git reset"
else
  [ -e "$DIR" ] && [ -n "$(ls -A "$DIR" 2>/dev/null)" ] && [ ! -d "$DIR/.git" ] && {
    mv "$DIR" "$DIR.old.$(date +%s)" || fail "move the old $DIR aside"; echo "  the old $DIR (not a git checkout) was moved aside"; }
  git clone -q --branch "$BRANCH" "$REPO" "$DIR" || fail "git clone $REPO"
fi
echo "::info commit=$(git -C "$DIR" log -1 --format='%h %s')"
ok

# 2 keys -------------------------------------------------------------------------------------------------------------
step keys
ENVF="$DIR/.env"
umask 077
touch "$ENVF" && chmod 600 "$ENVF" || fail "cannot write $ENVF"
if [ ! -t 0 ]; then
  got=0
  while IFS= read -r line || [ -n "$line" ]; do
    key="${line%%=*}"
    case "$key" in
      TUNNEL_KEY|TRAINER_KEY|HF_TOKEN)
        val="${line#*=}"
        [ -n "$val" ] || continue
        grep -v "^$key=" "$ENVF" > "$ENVF.tmp" 2>/dev/null; printf '%s=%s\n' "$key" "$val" >> "$ENVF.tmp"
        mv "$ENVF.tmp" "$ENVF"; chmod 600 "$ENVF"; got=$((got + 1)) ;;
      "") ;;
      *) echo "::warn ignored a stdin line that is not TUNNEL_KEY, TRAINER_KEY or HF_TOKEN" ;;
    esac
  done
  echo "  $got key(s) written"
fi
for k in TUNNEL_KEY TRAINER_KEY HF_TOKEN; do  # lengths only, never the values
  v="$(grep "^$k=" "$ENVF" | tail -1 | cut -d= -f2-)"
  echo "  $k: ${#v} characters"
done
grep -q "^TUNNEL_KEY=..*" "$ENVF" || fail "no TUNNEL_KEY: the pod's tunnel port would be open with the sample key"
ok

# 3 detector ---------------------------------------------------------------------------------------------------------
step detector
if [ -s "$DIR/shared/detector.py" ]; then
  echo "  shared/detector.py present"
else
  echo "::warn no shared/detector.py: RF-DETR pre-labelling on the pod needs the gateway's labelling/detector.py (the gateway's Pods page sends it)"
fi
ok

# 4 python -----------------------------------------------------------------------------------------------------------
step python
cd "$DIR" || fail "cd $DIR"
if [ "$SKIP_PIP" = 1 ]; then
  echo "  skipped (--skip-pip)"
else
  [ -x .venv/bin/python ] || python3 -m venv --system-site-packages .venv || fail "venv"
  .venv/bin/pip install -q --upgrade pip || fail "pip upgrade"
  .venv/bin/pip install -q -r requirements-train.txt || fail "pip install requirements-train.txt"
  .venv/bin/python -c "import torch, rfdetr; print('  torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "torch / rfdetr do not import"
fi
ok

# 5 vlm --------------------------------------------------------------------------------------------------------------
if [ "$VLM" = 1 ]; then
  step vlm
  if [ "$SKIP_PIP" = 1 ]; then
    echo "  skipped (--skip-pip)"
  else
    free=$(df -Pk "$HOME" | awk 'NR==2 {print int($4 / 1048576)}')
    have=0; [ -d "$HOME/hf/hub/models--Qwen--Qwen3.5-9B" ] && have=1
    echo "  ${free} GB free in $HOME"
    [ "$have" = 1 ] || [ "$free" -ge 35 ] || fail "only ${free} GB free: vLLM (~10 GB) + Qwen3.5-9B (~19 GB) need 35 GB"
    [ -x "$HOME/vllm-venv/bin/python" ] || python3 -m venv "$HOME/vllm-venv" || fail "vllm venv"
    "$HOME/vllm-venv/bin/pip" install -q --upgrade pip || fail "pip upgrade (vllm venv)"
    "$HOME/vllm-venv/bin/pip" install -q -r requirements-vlm.txt || fail "pip install requirements-vlm.txt"
    "$HOME/vllm-venv/bin/python" -c "import vllm, torch; print('  vllm', vllm.__version__, '· torch', torch.__version__, 'cuda', torch.cuda.is_available())" || fail "vllm does not import"
    HF_HOME="$HOME/hf" "$HOME/vllm-venv/bin/python" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3.5-9B')" > download.log 2>&1 \
      || { tail -n 3 download.log; fail "download Qwen/Qwen3.5-9B"; }
    echo "  Qwen/Qwen3.5-9B in $HOME/hf"
  fi
  ok
fi

# 6 start ------------------------------------------------------------------------------------------------------------
POD="${RUNPOD_POD_ID:-$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | grep '^RUNPOD_POD_ID=' | cut -d= -f2)}"
if [ "$START" = 1 ]; then
  step start
  if [ -f pod.pid ] && kill -0 "$(cat pod.pid)" 2>/dev/null; then kill "$(cat pod.pid)"; sleep 3; fi
  pkill -f "[v]llm serve" || true   # a vLLM left from an older server (the pattern does not match this shell)
  PY=.venv/bin/python; [ -x "$PY" ] || PY=python3
  nohup "$PY" server.py training --listen 8765 --no-dial --trainer rfdetr > pod.log 2>&1 < /dev/null &
  echo $! > pod.pid
  code=000
  for _ in $(seq 1 30); do
    sleep 1
    kill -0 "$(cat pod.pid)" 2>/dev/null || { tail -n 5 pod.log; fail "the pod server exited"; }
    code=$(curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8765/tunnel || true)
    [ "$code" = 403 ] && break
  done
  [ "$code" = 403 ] || { tail -n 5 pod.log; fail "port 8765 answers $code, not 403"; }
  echo "  pod server pid $(cat pod.pid), :8765/tunnel answers 403 without the key"
  ok
fi
echo "::done pod=${POD:-unknown} tunnel=${POD:+wss://$POD-8765.proxy.runpod.net/tunnel}"
