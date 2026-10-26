"""The pod's VLM (service name "vllm"): vLLM with Qwen3.5-9B, started on demand, because it takes most of the GPU's
memory and the pod also trains. Key: Bearer TRAINER_KEY (the same pod as the trainer).

  POST /v1/vlm/start            start vLLM (refused while a training job runs); answers at once with the status.
                                {mode: "prelabel" (default) | "inference"}; a start in the other mode restarts it.
                                A start while it runs in that mode only counts as use (the idle timer starts again).
  GET  /v1/vlm/status           {state: stopped | starting | ready | failed, detail, engine, model, mode}
  POST /v1/vlm/stop             stop vLLM (the gateway does this after a pre-label run)
  GET  /v1/models, POST /v1/chat/completions    passed to vLLM (503 while it isn't ready)
  GET  /health

vLLM runs as a subprocess on a free 127.0.0.1 port, from its own venv (VLLM_BIN, default ~/vllm-venv/bin/vllm,
whose torch differs from rfdetr's), its log in VLLM_LOG (default ~/runpod/vllm.log). Without vLLM installed (a Mac,
the tests) the fake vLLM (services/vllm.py) runs instead, and health says engine "fake". VLLM_CMD replaces the command
({port} and {model} are filled in). It stops by itself after VLLM_IDLE_S (default 600) without a request, in case
the gateway goes away mid-run; that only stops vLLM, never the pod.
"""

import atexit
import os
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

from typing import Literal

import httpx
from fastapi import Depends, HTTPException, Request, Response
from pydantic import BaseModel

from . import gpu
from .common import bearer, make_app

app = make_app("VLM service (vLLM, on demand)")
app.title, app.description = "VLM service (vLLM, on demand)", "Starts vLLM on the pod's GPU for pre-labelling."
auth = [Depends(bearer("trainer"))]
HERE = Path(__file__).resolve().parent.parent
MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3.5-9B")
IDLE_S = float(os.environ.get("VLLM_IDLE_S", "600"))
LOG = Path(os.environ.get("VLLM_LOG", HERE / "vllm.log"))
# bf16 Qwen3.5-9B on a 24 GB card: ~18 GB of weights, the rest for the KV cache; one image per request, a few requests
# at a time (the gateway sends one frame at a time; vLLM's default of 256 runs out of memory capturing CUDA graphs).
# --enforce-eager: no torch.compile / CUDA graph capture, so it is ready in ~70 s instead of ~250 s (it starts per run)
FLAGS = ["--max-model-len", "8192", "--limit-mm-per-prompt", '{"image": 1}', "--gpu-memory-utilization", "0.88",
         "--max-num-seqs", "4", "--enforce-eager"]
# inference: RF-DETR stays loaded beside it, so the weights are quantized to fp8 on load (~9 GB instead of ~18 GB) and
# vLLM takes 70 % of the card; a rule hit sends a clip of frames and reference images: up to 8 images per request
INFERENCE_FLAGS = ["--max-model-len", "8192", "--limit-mm-per-prompt", '{"image": 8}', "--gpu-memory-utilization", "0.70",
                   "--quantization", "fp8", "--max-num-seqs", "2", "--enforce-eager"]
MODES = {"prelabel": FLAGS, "inference": INFERENCE_FLAGS}
_lock = threading.Lock()
_s = {"proc": None, "port": None, "used": 0.0, "why": None, "mode": None}  # why: the last stop's reason / exit


def real_bin():
    p = Path(os.environ.get("VLLM_BIN", Path.home() / "vllm-venv" / "bin" / "vllm"))
    return p if p.exists() else None


def engine():
    return "custom" if os.environ.get("VLLM_CMD") else "vllm" if real_bin() else "fake"


def command(port, mode="prelabel"):
    if os.environ.get("VLLM_CMD"):
        return shlex.split(os.environ["VLLM_CMD"].format(port=port, model=MODEL))
    if real_bin():
        return [str(real_bin()), "serve", MODEL, "--served-model-name", MODEL, "--host", "127.0.0.1", "--port", str(port),
                *MODES[mode]]
    return [sys.executable, "-m", "uvicorn", "services.vllm:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def log_tail(n=6):
    try:
        with open(LOG, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - 8192))
            text = f.read().decode(errors="replace")
    except OSError:
        return []
    return [ln.strip() for ln in text.replace("\r", "\n").split("\n") if ln.strip()][-n:]


def running():
    p = _s["proc"]
    return p is not None and p.poll() is None


def ready():
    if not running():
        return False
    try:
        return httpx.get(f"http://127.0.0.1:{_s['port']}/v1/models", timeout=1).status_code == 200
    except httpx.HTTPError:
        return False


def state():
    p = _s["proc"]
    if p is None:
        return {"state": "stopped", "detail": _s["why"] or ""}
    if p.poll() is not None:
        return {"state": "failed", "detail": f"vLLM exited with code {p.returncode}: " + " | ".join(log_tail())}
    if ready():
        return {"state": "ready", "detail": f"{MODEL} on 127.0.0.1:{_s['port']}"}
    tail = log_tail(1)
    return {"state": "starting", "detail": tail[0][:300] if tail else "starting"}


def view():
    return {**state(), "engine": engine(), "model": MODEL, "mode": _s["mode"] if _s["proc"] is not None else None}


gpu.holder("vlm", lambda: f"vLLM is running for {'inference' if _s['mode'] == 'inference' else 'pre-labelling'}"
           if running() else None)
# pre-label mode takes most of the card, so RF-DETR waits; in inference mode it runs beside vLLM
gpu.holder("vlm-exclusive", lambda: "vLLM is running for pre-labelling" if running() and _s["mode"] != "inference" else None)


def stop(why):
    with _lock:
        if _s["proc"] is None:
            return
        _s["why"], p, _s["proc"] = why, _s["proc"], None  # the reason first: status reads it as soon as proc is gone
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)  # vLLM's engine runs in child processes: stop the whole group
                p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(10)
            except ProcessLookupError:
                pass


atexit.register(stop, "the pod server stopped")  # vLLM runs in its own process group: don't leave it holding the GPU


def watch_idle(p):
    while _s["proc"] is p and p.poll() is None:
        time.sleep(min(15.0, max(0.05, IDLE_S / 4)))
        if _s["proc"] is p and time.monotonic() - _s["used"] > IDLE_S:
            stop(f"stopped after {IDLE_S:.0f} s without a request")


@app.get("/health")
def health():
    return {"status": "ok", "service": "vllm", **view()}


class StartIn(BaseModel):
    mode: Literal["prelabel", "inference"] = "prelabel"


@app.post("/v1/vlm/start", dependencies=auth)
def start(body: StartIn | None = None):
    mode = (body or StartIn()).mode
    if running() and _s["mode"] != mode:
        stop(f"restarting for {mode}")
    with _lock:
        if running():
            _s["used"] = time.monotonic()
            return view()
        if reason := gpu.busy("training"):
            raise HTTPException(409, f"{reason}: the VLM can start when it is done")
        if mode == "prelabel":
            gpu.release()  # the labeller's cached RF-DETR models
        port = free_port()
        LOG.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.setdefault("HF_HOME", str(Path.home() / "hf"))  # where deploy.sh VLM=1 downloaded the model (~19 GB)
        with open(LOG, "ab") as log:
            log.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} starting {MODEL} ({engine()}, {mode})\n".encode())
            log.flush()
            p = subprocess.Popen(command(port, mode), stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 cwd=str(HERE), env=env, start_new_session=True)
        _s.update(proc=p, port=port, used=time.monotonic(), why=None, mode=mode)
    threading.Thread(target=watch_idle, args=(p,), name="vllm-idle", daemon=True).start()
    return view()


@app.get("/v1/vlm/status", dependencies=auth)
def status():
    return view()


@app.post("/v1/vlm/stop", dependencies=auth)
def stop_route():
    stop("stopped by the gateway")
    return view()


async def forward(request: Request, path):
    if not ready():
        raise HTTPException(503, {"code": "vlm_not_running", "message": f"vLLM is {state()['state']}: start it first"})
    _s["used"] = time.monotonic()
    async with httpx.AsyncClient(timeout=300) as http:
        r = await http.request(request.method, f"http://127.0.0.1:{_s['port']}{path}", content=await request.body(),
                               headers={"content-type": request.headers.get("content-type", "application/json")})
    _s["used"] = time.monotonic()
    return Response(r.content, r.status_code, media_type=r.headers.get("content-type"))


@app.get("/v1/models", dependencies=auth)
def models():
    """The model this pod serves, running or not (the gateway lists it before starting it)."""
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "vllm", "state": state()["state"]}]}


@app.post("/v1/chat/completions", dependencies=auth)
async def chat(request: Request):
    return await forward(request, "/v1/chat/completions")
