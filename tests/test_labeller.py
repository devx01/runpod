"""The pod's RF-DETR pre-label service: chunked weights upload (checked by SHA-256), the key, and a prediction with the
shipped model (on this machine's CPU / MPS). python -m pytest -q (from runpod/)."""

import base64
import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
WEIGHTS = Path(__file__).resolve().parents[2] / "gateway" / "assets" / "weights" / "detector_r1_medium.pth"
VIDEO = Path(__file__).resolve().parents[2] / "gateway" / "assets" / "videos" / "mining_site_example.mp4"
H = {"Authorization": "Bearer fake-trainer-key"}


@pytest.fixture()
def lab(tmp_path, monkeypatch):
    monkeypatch.setenv("LABELLER_CACHE", str(tmp_path))
    import services.labeller as mod
    return TestClient(importlib.reload(mod).app)


def put(c, sha, data, a, total):
    return c.put(f"/v1/models/{sha}", content=data, headers={**H, "Content-Range": f"bytes {a}-{a + len(data) - 1}/{total}"})


def test_chunked_upload_is_checked(lab):
    data = bytes(range(256)) * 50
    sha = hashlib.sha256(data).hexdigest()
    assert lab.get(f"/v1/models/{sha}").status_code == 401  # the trainer key is needed
    assert lab.get(f"/v1/models/{sha}", headers=H).json() == {"present": False, "bytes": 0}
    assert put(lab, sha, data[:5000], 0, len(data)).json() == {"present": False, "bytes": 5000}
    assert put(lab, sha, data[6000:7000], 6000, len(data)).status_code == 409  # a gap: resume from byte 5000
    assert lab.get(f"/v1/models/{sha}", headers=H).json()["bytes"] == 5000
    assert put(lab, sha, data[5000:], 5000, len(data)).json() == {"present": True, "bytes": len(data)}
    bad = hashlib.sha256(b"other").hexdigest()
    assert put(lab, bad, data, 0, len(data)).status_code == 422  # not what it claims to be: dropped
    assert lab.get(f"/v1/models/{bad}", headers=H).json() == {"present": False, "bytes": 0}
    assert lab.get("/v1/models/not-a-sha", headers=H).status_code == 422


@pytest.mark.skipif(not WEIGHTS.exists() or not VIDEO.exists(), reason="the shipped weights / example video are needed")
def test_predict_with_the_shipped_model(lab):
    import cv2
    ok, img = cv2.VideoCapture(str(VIDEO)).read()
    jpeg = cv2.imencode(".jpg", img)[1].tobytes()
    data = WEIGHTS.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    args = json.loads(WEIGHTS.with_name("detector_r1_medium_args.json").read_text())
    body = {"model": {"sha256": sha, "args": args}, "image_b64": base64.b64encode(jpeg).decode()}
    assert lab.post("/v1/predict", json=body, headers=H).status_code == 409  # not on the pod yet
    for a in range(0, len(data), 4 << 20):
        assert put(lab, sha, data[a:a + (4 << 20)], a, len(data)).status_code == 200
    out = lab.post("/v1/predict", json={**body, "threshold": 0.25}, headers=H).json()
    assert out["device"] in ("cuda", "mps", "cpu")
    assert all(b["class"] in args["classes"] and len(b["xyxy"]) == 4 for b in out["boxes"])
    assert lab.post("/v1/predict", json={**body, "image_b64": base64.b64encode(b"nope").decode()}, headers=H).status_code == 422
