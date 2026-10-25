"""Shared bits of the pod's services (fake for now): credential checks and the synthetic numbers they answer with."""

import json
import os
import re
import secrets
import zlib

from fastapi import FastAPI, Header, HTTPException, Request

# the service keys the gateway sends (through the tunnel); override with the same env names the gateway uses
KEYS = {
    "detector": os.environ.get("DETECTOR_KEY", "fake-detector-key"),
    "trainer": os.environ.get("TRAINER_KEY", "fake-trainer-key"),
}

# boxes the fake detector "sees" on every frame (pixels of a 1280x720 frame); the fake VLM uses their places only
BOXES = [{"class": "truck", "conf": 0.96, "xyxy": [454.8, 461.1, 848.7, 554.3]},
         {"class": "truck", "conf": 0.936, "xyxy": [78.6, 629.8, 383.6, 720.0]},
         {"class": "truck", "conf": 0.826, "xyxy": [250.5, 316.6, 350.8, 393.6]},
         {"class": "person", "conf": 0.71, "xyxy": [402, 532, 452, 624]}]


def make_app(title):
    app = FastAPI(title=f"{title} (fake)", description="Synthetic stand-in for the real service; same API shape.")
    app.state.calls = []  # every request, for tests: (method, path)

    @app.middleware("http")
    async def record(request: Request, call_next):
        app.state.calls.append((request.method, request.url.path))
        return await call_next(request)
    return app


def bearer(service):
    """A dependency: `Authorization: Bearer <key>` must carry this service's key, else 401."""
    def check(authorization: str = Header("")):
        if not secrets.compare_digest(authorization.removeprefix("Bearer ").strip(), KEYS[service]):
            raise HTTPException(401, f"invalid {service} credential")
    return check


def text_of(content):
    """Plain text of an OpenAI message content (a string or a list of parts)."""
    if isinstance(content, str):
        return content
    return " ".join(p.get("text", "") for p in content if isinstance(p, dict))


def prompt_classes(text):
    """The class names a pre-label prompt lists, one per line ("- helmet: hard hats")."""
    return re.findall(r"^- ([a-z0-9][a-z0-9_-]*)", text, re.M)


def scores(classes, seed):
    """Synthetic held-out recall / precision per class, the same for the same class and seed."""
    out = {}
    for c in classes:
        h = zlib.crc32(f"{seed}:{c}".encode())
        out[c] = {"recall": 70 + h % 25, "precision": 70 + (h >> 8) % 25}
    return out


def vlm_answer(text):
    """The pre-label prompt asks for "boxes": one box for each of the first two classes it lists (FAKE_VLM_EXTRA_CLASS=
    <name> adds one box of a class outside the label list, to try the automatic label registration); anything else
    gets an alert judgement. Like Qwen3.5, it answers on a 0-1000 grid when the prompt asks for one ("Frame size WxH")."""
    if "boxes" in text.lower():
        boxes = [{"class": c, "xyxy": b["xyxy"]} for c, b in zip(prompt_classes(text)[:2], BOXES)]
        if os.environ.get("FAKE_VLM_EXTRA_CLASS"):
            boxes.append({"class": os.environ["FAKE_VLM_EXTRA_CLASS"], "xyxy": [402, 532, 452, 624]})
        size = re.search(r"Frame size (\d+)x(\d+)", text)
        if "0-1000 grid" in text and size:
            w, h = int(size.group(1)), int(size.group(2))
            for bx in boxes:
                bx["xyxy"] = [round(v * 1000 / (w if i % 2 == 0 else h)) for i, v in enumerate(bx["xyxy"])]
        return json.dumps({"boxes": boxes})
    return json.dumps({"dangerous": False, "confidence": 0.82, "reason": "synthetic answer"})
