"""The real training service (on a GPU pod): the same API as the fake one (services/trainer.py), but each job really
fine-tunes RF-DETR, in a subprocess (train_job.py), one job at a time. Key: Bearer TRAINER_KEY.

  POST /v1/jobs {dataset: hf://org/repo/rid, model, epochs, classes, pod_id}   → the job
  GET  /v1/jobs/{id} · /logs · /metrics · /weights · /weights/file (Range → 206)

Job folders live in TRAINER_JOBS_DIR (default ~/runpod-jobs): job.json, status.json, log.txt, rfdetr.log, the dataset,
RF-DETR's output and result.json. They stay on the pod until you terminate it.
"""

import itertools
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import gpu
from .common import bearer, make_app

app = make_app("Training service (RF-DETR)")
app.title, app.description = "Training service (RF-DETR)", "Fine-tunes RF-DETR on the pod's GPU."  # not a fake
auth = [Depends(bearer("trainer"))]
JOBS_DIR = Path(os.environ.get("TRAINER_JOBS_DIR", Path.home() / "runpod-jobs"))
TRAIN_JOB = Path(__file__).resolve().parent.parent / "train_job.py"
JOBS = {}  # id -> {"dir", "proc", "cfg"}
_lock = threading.Lock()
ids = itertools.count(1)


class JobIn(BaseModel):
    dataset: str = Field(description="hf://<org>/<repo>/<dataset id>: the gateway's dataset in a private repo")
    model: str = "medium"
    epochs: int = Field(50, ge=1, le=500)
    pod_id: str | None = None
    classes: list[str] = Field(min_length=1, description="the dataset's classes, in class-id order")
    batch_size: int = Field(4, ge=1, le=64)
    grad_accum_steps: int = Field(4, ge=1, le=64)


def status(j):
    p = j["dir"] / "status.json"
    s = json.loads(p.read_text()) if p.exists() else {"status": "queued", "stage": "queued", "progress": 0.0, "error": None}
    if s["status"] == "running" and j["proc"].poll() not in (None, 0):  # the subprocess died without saying why
        s = {**s, "status": "failed", "error": s.get("error") or f"the training process exited with {j['proc'].returncode}"}
    return s


def view(j):
    s, cfg = status(j), j["cfg"]
    done = s["status"] == "done"
    return {"id": cfg["id"], "status": s["status"], "stage": s["stage"] if s["status"] != "failed" else f"failed: {s['error']}",
            "progress": s["progress"], "dataset": cfg["dataset"], "model": cfg["model"], "epochs": cfg["epochs"],
            "pod_id": cfg.get("pod_id"), "result_model": cfg["result_model"] if done else None}


def get(job_id):
    if job_id not in JOBS:
        raise HTTPException(404, f"no job {job_id}")
    return JOBS[job_id]


def result(j):
    if status(j)["status"] != "done":
        raise HTTPException(409, f"job {j['cfg']['id']} is not finished yet")
    return json.loads((j["dir"] / "result.json").read_text())


@app.get("/health")
def health():
    running = [i for i, j in JOBS.items() if status(j)["status"] in ("queued", "running")]
    return {"status": "ok", "trainer": "rfdetr", "jobs": len(JOBS), "running": running}


def training():
    busy = [i for i, j in JOBS.items() if status(j)["status"] in ("queued", "running")]
    return f"the pod is training {busy[0]}" if busy else None


gpu.holder("training", training)


@app.post("/v1/jobs", status_code=201, dependencies=auth)
def create(body: JobIn):
    if not body.dataset.startswith("hf://"):
        raise HTTPException(422, f"the training pod reads datasets from Hugging Face (hf://org/repo/rid), not {body.dataset!r}")
    if body.model not in ("nano", "small", "medium", "large"):
        raise HTTPException(422, "model must be nano, small, medium or large")
    with _lock:
        if reason := training():
            raise HTTPException(409, f"{reason}: one job at a time")
        if reason := gpu.busy("vlm"):
            raise HTTPException(409, f"{reason}: train when it is done")
        jid = f"train-{next(ids):04d}"
        while (JOBS_DIR / jid).exists():  # job folders survive a service restart
            jid = f"train-{next(ids):04d}"
        d = JOBS_DIR / jid
        d.mkdir(parents=True)
        cfg = {"id": jid, **body.model_dump(), "result_model": f"detector_r2_{body.model}"}
        (d / "job.json").write_text(json.dumps(cfg))
        log = open(d / "rfdetr.log", "ab")
        proc = subprocess.Popen([sys.executable, str(TRAIN_JOB), str(d)], stdout=log, stderr=subprocess.STDOUT,
                                cwd=str(d), env=os.environ.copy())
        JOBS[jid] = {"dir": d, "proc": proc, "cfg": cfg}
    return view(JOBS[jid])


@app.get("/v1/jobs", dependencies=auth)
def jobs():
    return [view(j) for j in JOBS.values()]


@app.get("/v1/jobs/{job_id}", dependencies=auth)
def job(job_id: str):
    return view(get(job_id))


@app.get("/v1/jobs/{job_id}/logs", dependencies=auth)
def logs(job_id: str):
    p = get(job_id)["dir"] / "log.txt"
    return {"lines": p.read_text().splitlines()[-500:] if p.exists() else [f"{job_id} queued"]}


@app.get("/v1/jobs/{job_id}/metrics", dependencies=auth)
def metrics(job_id: str):
    j = get(job_id)
    m = result(j)["metrics"]
    return {"model": j["cfg"]["result_model"], "split": m["split"], "classes": m["classes"], "current": None}


@app.get("/v1/jobs/{job_id}/weights", dependencies=auth)
def weights(job_id: str, request: Request):
    j = get(job_id)
    r = result(j)
    return {"name": j["cfg"]["result_model"], "url": str(request.url_for("weights_file", job_id=job_id)),
            "sha256": r["sha256"], "bytes": r["bytes"], "args": r["args"]}


@app.get("/v1/jobs/{job_id}/weights/file", dependencies=auth)
def weights_file(job_id: str):
    return FileResponse(result(get(job_id))["weights"], media_type="application/octet-stream")  # Range → 206
