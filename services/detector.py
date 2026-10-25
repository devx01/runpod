"""Fake detector + alert pipeline (the inference pod's `serve`, :8091 there, :9091 here). Key: Bearer DETECTOR_KEY."""

from fastapi import Depends, Header, HTTPException, Request, Response
from pydantic import BaseModel

from .common import BOXES, bearer, make_app

app = make_app("Detector + alert pipeline")
auth = [Depends(bearer("detector"))]
STATE = {}


def reset():
    STATE.clear()
    STATE.update(models=[{"name": "detector_r1_medium", "active": True, "job_id": None, "weights_url": None}],
                 prompts=[{"version": 1, "text": "Default VLM prompt (see VLM-prompt.md)."}], active_prompt=1, feedback=[],
                 alerts=[{"id": "demo#1:R1:5-1:41.0", "status": "alert", "rule": "R1",
                          "message": "Orang 1.4 m dari truck yang bergerak", "t": 41.0, "clip": []}])


reset()


def active_model():
    return next(m for m in STATE["models"] if m["active"])


@app.get("/health")
def health():
    return {"status": "ok", "model": active_model()["name"]}


@app.post("/v1/detect", dependencies=auth)
async def detect(request: Request, x_stream: str = Header("app"), x_timestamp: float = Header(0.0)):
    if not await request.body():
        raise HTTPException(400, "empty body: send one JPEG frame")
    alert = {"id": f"{x_stream}#1:R1:5-1:{x_timestamp}", "status": "warn", "rule": "R1",
             "message": "Orang 2.1 m dari truck yang bergerak", "t": x_timestamp, "clip": []}
    STATE["alerts"].append(alert)
    return {"detector": f"rfdetr-{active_model()['name']}", "width": 1280, "height": 720, "timestamp": x_timestamp,
            "detections": BOXES, "safety": {"level": "warning", "min_distance_m": 2.1}, "alerts": [alert]}


@app.get("/v1/alerts", dependencies=auth)
def alerts():
    return STATE["alerts"][-50:]


class Feedback(BaseModel):
    action: str
    reasons: list[str] = []
    note: str = ""
    by: str = ""


@app.post("/v1/alerts/{alert_id}/feedback", status_code=204, dependencies=auth)
def feedback(alert_id: str, body: Feedback):
    STATE["feedback"].append({"alert": alert_id, **body.model_dump()})
    return Response(status_code=204)


def prompt_view():
    v = next(p for p in STATE["prompts"] if p["version"] == STATE["active_prompt"])
    return {"version": v["version"], "text": v["text"], "editable": True}


@app.get("/v1/alerts/prompt", dependencies=auth)
def get_prompt():
    return prompt_view()


class PromptIn(BaseModel):
    text: str
    note: str = ""


@app.put("/v1/alerts/prompt", dependencies=auth)
def put_prompt(body: PromptIn):
    v = max(p["version"] for p in STATE["prompts"]) + 1
    STATE["prompts"].append({"version": v, "text": body.text})
    STATE["active_prompt"] = v
    return prompt_view()


class Version(BaseModel):
    version: int


@app.post("/v1/alerts/prompt/activate", dependencies=auth)
def activate_prompt(body: Version):
    if body.version not in {p["version"] for p in STATE["prompts"]}:
        raise HTTPException(404, f"no prompt version {body.version}")
    STATE["active_prompt"] = body.version
    return prompt_view()


@app.get("/v1/admin/models", dependencies=auth)
def models():
    return [{k: m[k] for k in ("name", "active", "job_id")} for m in STATE["models"]]


class Reload(BaseModel):
    name: str
    weights_url: str
    job_id: str | None = None


@app.post("/v1/admin/reload", dependencies=auth)
def reload(body: Reload):
    prev = active_model()["name"]
    for m in STATE["models"]:
        m["active"] = False
    STATE["models"] = [m for m in STATE["models"] if m["name"] != body.name]
    STATE["models"].append({"name": body.name, "active": True, "job_id": body.job_id, "weights_url": body.weights_url})
    return {"active": body.name, "previous": prev}
