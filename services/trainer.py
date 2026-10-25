"""Fake training service (runs on the rented training pod). Key: Bearer TRAINER_KEY.

A job walks queued → downloading dataset → training → evaluating → done, STEP_SECONDS per stage (0 = done at once).
It is told the dataset's classes (a real trainer reads them from the COCO file) and reports them with the weights, so
whoever installs the weights knows what they detect. The weights file is FAKE_TRAINER_WEIGHTS (a real checkpoint to
stand in for the trained one) or, without it, placeholder bytes that are not a usable checkpoint. It is served with
Range support, so it can be fetched in chunks through the reverse tunnel.
"""

import hashlib
import itertools
import os
import tempfile
import time
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import gpu
from .common import bearer, make_app, scores

app = make_app("Training service")
auth = [Depends(bearer("trainer"))]
STAGES = ["queued", "downloading dataset", "training", "evaluating", "done"]
STEP_SECONDS = float(os.environ.get("FAKE_TRAINER_STEP_SECONDS", "1.0"))
JOBS = {}
ids = itertools.count(1)
PLACEHOLDER = b"fake RF-DETR checkpoint\n" * 64  # served when FAKE_TRAINER_WEIGHTS is not set
WEIGHTS_PATH = os.environ.get("FAKE_TRAINER_WEIGHTS")  # tests may set this module attribute
_sha = {}  # (path, size, mtime) -> sha256


def weights_file_path():
    if WEIGHTS_PATH:
        return Path(WEIGHTS_PATH)
    p = Path(tempfile.gettempdir()) / "fake-trainer-placeholder.pth"
    if not p.exists() or p.read_bytes() != PLACEHOLDER:
        p.write_bytes(PLACEHOLDER)
    return p


def sha256_of(p):
    st = p.stat()
    key = (str(p), st.st_size, st.st_mtime)
    if key not in _sha:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _sha[key] = h.hexdigest()
    return _sha[key]


def reset():
    JOBS.clear()


class JobIn(BaseModel):
    dataset: str
    model: str = "medium"
    epochs: int = Field(50, ge=1)
    pod_id: str | None = None
    classes: list[str] = Field(min_length=1, description="the dataset's classes, in class-id order")


def view(j):
    k = len(STAGES) - 1 if STEP_SECONDS == 0 else min(len(STAGES) - 1, int((time.monotonic() - j["t0"]) / STEP_SECONDS))
    done = STAGES[k] == "done"
    return {"id": j["id"], "status": "done" if done else "queued" if k == 0 else "running", "stage": STAGES[k],
            "progress": round(k / (len(STAGES) - 1), 2), "dataset": j["dataset"], "model": j["model"],
            "epochs": j["epochs"], "pod_id": j["pod_id"], "result_model": f"detector_r2_{j['model']}" if done else None}


def get(job_id):
    if job_id not in JOBS:
        raise HTTPException(404, f"no job {job_id}")
    return JOBS[job_id], view(JOBS[job_id])


@app.get("/health")
def health():
    return {"status": "ok", "jobs": len(JOBS)}


def training():
    busy = [i for i, j in JOBS.items() if view(j)["status"] in ("queued", "running")]
    return f"the pod is training {busy[0]}" if busy else None


gpu.holder("training", training)


@app.post("/v1/jobs", status_code=201, dependencies=auth)
def create(body: JobIn):
    if reason := gpu.busy("vlm"):
        raise HTTPException(409, f"{reason}: train when it is done")
    jid = f"train-{next(ids):04d}"
    JOBS[jid] = {"id": jid, "t0": time.monotonic(), **body.model_dump()}
    return view(JOBS[jid])


@app.get("/v1/jobs", dependencies=auth)
def jobs():
    return [view(j) for j in JOBS.values()]


@app.get("/v1/jobs/{job_id}", dependencies=auth)
def job(job_id: str):
    return get(job_id)[1]


@app.get("/v1/jobs/{job_id}/logs", dependencies=auth)
def logs(job_id: str):
    j, v = get(job_id)
    k = STAGES.index(v["stage"])
    lines = [f"{job_id} queued on pod {j['pod_id'] or '-'}"]
    if k >= 1:
        lines.append(f"downloading {j['dataset']} with the pod's HF token")
    if k >= 2:
        lines += [f"RFDETR{j['model'].title()} · {j['epochs']} epochs · {len(j['classes'])} classes: {', '.join(j['classes'])}",
                  "epoch 30/50  loss 0.33  val mAP50 0.82"]
    if k >= 3:
        lines.append("epoch 50/50  loss 0.24  val mAP50 0.87 · best checkpoint saved")
    if k >= 4:
        lines.append(f"{job_id} done → {v['result_model']}")
    return {"lines": lines}


@app.get("/v1/jobs/{job_id}/metrics", dependencies=auth)
def metrics(job_id: str):
    j, v = get(job_id)
    if v["status"] != "done":
        raise HTTPException(409, f"job {job_id} is not finished yet")
    new, current = scores(j["classes"], "new"), scores(j["classes"], "current")
    for c in new:  # the new model is never worse in this fake
        new[c] = {k: max(new[c][k], current[c][k]) for k in new[c]}
    return {"model": v["result_model"], "split": "heldout", "classes": new, "current": current}


@app.get("/v1/jobs/{job_id}/weights", dependencies=auth)
def weights(job_id: str, request: Request):
    _, v = get(job_id)
    if v["status"] != "done":
        raise HTTPException(409, f"job {job_id} is not finished yet")
    j, _ = get(job_id)
    p = weights_file_path()
    return {"name": v["result_model"], "url": str(request.url_for("weights_file", job_id=job_id)),
            "sha256": sha256_of(p), "bytes": p.stat().st_size,
            # what to save as <name>_args.json next to the weights
            "args": {"model": j["model"], "epochs": j["epochs"], "dataset": j["dataset"], "classes": j["classes"],
                     "resolution": None, "job_id": job_id}}


@app.get("/v1/jobs/{job_id}/weights/file", dependencies=auth)
def weights_file(job_id: str):
    get(job_id)
    return FileResponse(weights_file_path(), media_type="application/octet-stream")  # Range requests → 206
