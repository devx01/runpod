"""RF-DETR pre-labelling on the pod's GPU (the gateway sends one frame per request through the tunnel).
Key: Bearer TRAINER_KEY (the same pod as the trainer).

  PUT  /v1/models/{sha256}      the weights in chunks (Content-Range: bytes a-b/total), for models not on the Hub
  POST /v1/models/fetch         {name, location: hf://repo/name, revision, sha256}: the pod downloads a Hub model
  POST /v1/predict              {model: {sha256, args}, image_b64, threshold, tiles: [classes], tile_conf} -> {boxes}
  GET  /health

It runs the gateway's own labelling/detector.py (load, prelabel_frame): copied next to it as shared/detector.py by
deploy.sh, else read from ../gateway/labelling/ when running from the repo. So the boxes are the same as when the
gateway runs the same weights itself. Weights are cached by SHA-256 in LABELLER_CACHE (default ~/runpod-models).
"""

import base64
import hashlib
import importlib.util
import json
import os
import re
import threading
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from pydantic import BaseModel, Field

from . import gpu
from .common import bearer, make_app

app = make_app("Pre-label service (RF-DETR)")
app.title, app.description = "Pre-label service (RF-DETR)", "Runs RF-DETR on the pod's GPU for the gateway."
auth = [Depends(bearer("trainer"))]
CACHE = Path(os.environ.get("LABELLER_CACHE", Path.home() / "runpod-models"))
SHA = re.compile(r"^[0-9a-f]{64}$")
_models, _lock = {}, threading.Lock()  # sha256 -> (predict, name)


def _detector():
    here = Path(__file__).resolve().parent.parent
    for p in (here / "shared" / "detector.py", here.parent / "gateway" / "labelling" / "detector.py"):
        if p.exists():
            spec = importlib.util.spec_from_file_location("shared_detector", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise RuntimeError("no detector.py: run deploy.sh (it copies gateway/labelling/detector.py)")


detector = _detector()


def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(sha):
    if not SHA.match(sha or ""):
        raise HTTPException(422, "a model is named by its SHA-256 (64 hex characters)")
    return CACHE / sha / "model.pth"


def device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    except ImportError:
        return "none"


@app.get("/health")
def health():
    have = sorted(p.parent.name[:12] for p in CACHE.glob("*/model.pth")) if CACHE.exists() else []
    return {"status": "ok", "service": "labeller", "device": device(), "models": have, "loaded": len(_models)}


@app.get("/v1/models/{sha256}", dependencies=auth)
def model_state(sha256: str):
    """{present, bytes}: does the pod have these weights (complete and checked)?"""
    p = model_path(sha256)
    part = p.with_suffix(".part")
    return {"present": p.exists(), "bytes": p.stat().st_size if p.exists() else part.stat().st_size if part.exists() else 0}


@app.put("/v1/models/{sha256}", dependencies=auth)
async def upload(sha256: str, request: Request):
    """One chunk of the weights; the last one completes the file, which must have this SHA-256."""
    p = model_path(sha256)
    if p.exists():
        return {"present": True, "bytes": p.stat().st_size}
    m = re.match(r"^bytes (\d+)-(\d+)/(\d+)$", request.headers.get("content-range", ""))
    if not m:
        raise HTTPException(422, "send each chunk with Content-Range: bytes a-b/total")
    a, b, total = map(int, m.groups())
    body = await request.body()
    if len(body) != b - a + 1 or b >= total:
        raise HTTPException(422, "the chunk doesn't match its Content-Range")
    part = p.with_suffix(".part")
    p.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if a != have:
        raise HTTPException(409, f"expected the chunk starting at byte {have}")
    with open(part, "ab") as f:
        f.write(body)
    if b + 1 == total:
        if sha256_of(part) != sha256:
            part.unlink()
            raise HTTPException(422, "the uploaded weights don't match their SHA-256: send them again")
        part.replace(p)
    return {"present": p.exists(), "bytes": b + 1}


class FetchIn(BaseModel):
    name: str
    location: str = Field(description="hf://<org>/<repo>/<name>")
    revision: str | None = None
    sha256: str


@app.post("/v1/models/fetch", dependencies=auth)
def fetch(body: FetchIn):
    """Download a model from the private Hugging Face model repo (the pod's read-only HF_TOKEN), pinned to its commit."""
    p = model_path(body.sha256)
    if p.exists():
        return {"present": True, "bytes": p.stat().st_size}
    if not body.location.startswith("hf://"):
        raise HTTPException(422, "only hf:// locations are fetched; upload other weights in chunks")
    from huggingface_hub import hf_hub_download
    repo = body.location.removeprefix("hf://").rsplit("/", 1)[0]
    try:
        got = hf_hub_download(repo_id=repo, repo_type="model", filename=f"{body.name}/{body.name}.pth", revision=body.revision,
                              token=os.environ.get("HF_TOKEN") or None, cache_dir=str(CACHE / "hub"),
                              endpoint=os.environ.get("HF_ENDPOINT") or None)  # unset on RunPod: huggingface.co
    except Exception as e:  # noqa: BLE001 - not found, no access, network
        raise HTTPException(502, f"could not fetch {body.location}: {type(e).__name__}: {e}") from e
    if sha256_of(got) != body.sha256:
        raise HTTPException(422, f"{body.location} doesn't match its SHA-256")
    p.parent.mkdir(parents=True, exist_ok=True)
    os.replace(os.path.realpath(got), p)
    return {"present": True, "bytes": p.stat().st_size}


class PredictIn(BaseModel):
    model: dict = Field(description="{sha256, args}: args is the weights' _args.json (model size, classes)")
    image_b64: str
    threshold: float = Field(0.25, ge=0, le=1)
    tiles: list[str] = Field(default_factory=list, description="labels also searched in tiles (marked small)")
    tile_conf: float = Field(0.45, ge=0, le=1)


def loaded(sha, args):
    with _lock:
        if sha not in _models:
            p = model_path(sha)
            if not p.exists():
                raise HTTPException(409, "the pod doesn't have these weights yet: upload or fetch them first")
            p.with_name("model_args.json").write_text(json.dumps(args))
            _models[sha] = detector.load(p)
        return _models[sha]


def unload():
    """Before vLLM starts: drop the loaded models and give their GPU memory back (they load again on the next frame)."""
    with _lock:
        if not _models:
            return
        _models.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


gpu.on_release(unload)


@app.post("/v1/predict", dependencies=auth)
def predict(body: PredictIn):
    if reason := gpu.busy("vlm-exclusive"):  # vLLM in inference mode leaves room for RF-DETR
        raise HTTPException(409, f"{reason}: RF-DETR pre-labels when it is done")
    import cv2
    import numpy as np
    predict_fn, _ = loaded(body.model.get("sha256"), body.model.get("args") or {})
    img = cv2.imdecode(np.frombuffer(base64.b64decode(body.image_b64), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(422, "image_b64 is not a JPEG / PNG image")
    rgb = img[:, :, ::-1].copy()
    return {"boxes": detector.prelabel_frame(predict_fn, rgb, body.threshold, body.tiles, body.tile_conf),
            "device": device()}
