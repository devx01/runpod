"""The agent's listening side (the gateway dials the pod), and the real trainer's helpers that need no GPU.
python -m pytest -q (from runpod/)."""

import asyncio
import base64
import json
import sys
from pathlib import Path

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server  # noqa: E402
import train_job  # noqa: E402


@pytest.fixture()
def listening():
    """A pod with the fake trainer whose agent only listens (no dial-out)."""
    running = server.start(["training"], None, "k", ports={"trainer": 0, "labeller": 0, "vllm": 0}, listen={"training": 0}, pod_ids={"training": "pod-1"})
    agent = running.agents["training"].agent
    for _ in range(200):
        if agent.listening.is_set():
            break
        asyncio.run(asyncio.sleep(0.02))
    yield running, f"ws://127.0.0.1:{agent.listen_port}/tunnel"
    running.stop()


def dial(url, key, frames=()):
    """Dial like the gateway: read the hello, send frames, return (hello, replies) or the refused status."""
    async def go():
        async with connect(url, additional_headers={"Authorization": f"Bearer {key}"}, open_timeout=5) as ws:
            hello = json.loads(await ws.recv())
            replies = []
            for f in frames:
                await ws.send(json.dumps(f))
                replies.append(json.loads(await asyncio.wait_for(ws.recv(), 10)))
            return hello, replies
    try:
        return asyncio.run(go())
    except InvalidStatus as e:
        return e.response.status_code


def test_the_gateway_can_dial_the_pod(listening):
    running, url = listening
    assert dial(url, "wrong") == 403
    assert dial(url.replace("/tunnel", "/x"), "k") == 404
    hello, replies = dial(url, "k", [{"id": "1", "service": "trainer", "method": "GET", "path": "/health", "headers": {}, "body_b64": ""},
                                     {"id": "2", "service": "detector", "method": "GET", "path": "/", "headers": {}, "body_b64": ""}])
    assert hello == {"hello": {"pod": "pod-1", "services": ["trainer", "labeller", "vllm"]}}
    assert replies[0]["status"] == 200 and json.loads(base64.b64decode(replies[0]["body_b64"]))["status"] == "ok"
    assert "no detector service" in replies[1]["error"]


def test_one_tunnel_at_a_time(listening):
    _, url = listening

    async def two():
        async with connect(url, additional_headers={"Authorization": "Bearer k"}, open_timeout=5) as ws:
            await ws.recv()  # the hello: the first tunnel is up
            try:
                async with connect(url, additional_headers={"Authorization": "Bearer k"}, open_timeout=5):
                    return "accepted"
            except InvalidStatus as e:
                return e.response.status_code
    assert asyncio.run(two()) == 409
    assert dial(url, "k")[0]["hello"]["pod"] == "pod-1"  # free again once the first one closed


def test_no_dial_needs_listen():
    with pytest.raises(SystemExit):
        sys.argv = ["server.py", "training", "--no-dial"]
        server.main()


def test_listening_needs_a_real_key(monkeypatch):
    monkeypatch.delenv("TUNNEL_KEY", raising=False)
    monkeypatch.setattr(server, "pod_environment", lambda: None)
    with pytest.raises(SystemExit):  # a public port with the sample key: refused
        sys.argv = ["server.py", "training", "--listen", "0", "--no-dial"]
        server.main()


# ---- the real trainer's helpers -----------------------------------------------------------------------------------
def coco(images, boxes, classes=("helmet", "dozer")):
    return {"images": [{"id": i, "file_name": f"f{i:03d}.jpg", "width": 100, "height": 100} for i in range(images)],
            "annotations": [{"id": k, "image_id": i, "category_id": c, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0}
                            for k, (i, c) in enumerate(boxes)],
            "categories": [{"id": i, "name": n, "supercategory": n} for i, n in enumerate(classes)]}


def test_a_dataset_without_valid_gets_one_from_train(tmp_path):
    (tmp_path / "train").mkdir()
    for i in range(10):
        (tmp_path / "train" / f"f{i:03d}.jpg").write_bytes(b"jpg")
    (tmp_path / "train" / "_annotations.coco.json").write_text(json.dumps(coco(10, [(i, i % 2) for i in range(10)])))
    lines = []
    assert train_job.ensure_valid(tmp_path, lines.append) is True
    train, valid = (train_job.load_coco(tmp_path / s) for s in ("train", "valid"))
    assert len(train["images"]) == 8 and len(valid["images"]) == 2
    assert {a["image_id"] for a in valid["annotations"]} == {i["id"] for i in valid["images"]}
    assert sorted(p.name for p in (tmp_path / "valid").glob("*.jpg")) == ["f004.jpg", "f009.jpg"]
    assert valid["categories"] == train["categories"] and "held back" in lines[0]
    assert train_job.ensure_valid(tmp_path, lines.append) is False  # it has one now


def test_a_class_missing_from_valid_gets_valid_images(tmp_path):
    for side in ("train", "valid"):
        (tmp_path / side).mkdir()
    for i in range(10):
        (tmp_path / "train" / f"f{i:03d}.jpg").write_bytes(b"jpg")
    (tmp_path / "valid" / "f000.jpg").write_bytes(b"valid jpg")
    classes = ("truck", "dozer", "stockpile", "fire")
    # truck everywhere, stockpile in 6 training images only, dozer in one; valid has a truck only
    train = coco(10, [(i, 0) for i in range(10)] + [(i, 2) for i in range(4, 10)] + [(3, 1)], classes)
    valid = coco(1, [(0, 0)], classes)
    valid["images"][0]["file_name"] = "v000.jpg"
    (tmp_path / "valid" / "f000.jpg").rename(tmp_path / "valid" / "v000.jpg")
    (tmp_path / "train" / "_annotations.coco.json").write_text(json.dumps(train))
    (tmp_path / "valid" / "_annotations.coco.json").write_text(json.dumps(valid))
    lines = []
    assert train_job.cover_valid(tmp_path, lines.append) == ["stockpile"]  # one dozer image stays for training
    train, valid = (train_job.load_coco(tmp_path / s) for s in ("train", "valid"))
    assert sorted(i["file_name"] for i in valid["images"]) == ["f008.jpg", "v000.jpg"]
    assert sorted(p.name for p in (tmp_path / "valid").glob("*.jpg")) == ["f008.jpg", "v000.jpg"]
    assert len(train["images"]) == 9 and not any(a["image_id"] == 8 for a in train["annotations"])
    ids = {i["id"] for i in valid["images"]}
    assert len(ids) == 2 and {a["image_id"] for a in valid["annotations"]} == ids
    assert len({a["id"] for a in valid["annotations"]}) == len(valid["annotations"]) == 3  # truck, truck, stockpile
    assert "stockpile" in lines[0] and "held back" in lines[0]
    assert train_job.cover_valid(tmp_path, lines.append) == []  # covered now


def test_per_class_scores():
    truth = [("helmet", [0, 0, 10, 10]), ("helmet", [50, 50, 60, 60]), ("dozer", [0, 0, 40, 40])]
    pred = [("helmet", 0.9, [1, 1, 10, 10]), ("helmet", 0.8, [80, 80, 90, 90]), ("helmet", 0.2, [50, 50, 60, 60]),
            ("dozer", 0.7, [0, 0, 40, 40])]
    s = train_job.per_class_scores(["helmet", "dozer", "vest"], [(truth, pred)])
    assert s == {"helmet": {"recall": 50, "precision": 50}, "dozer": {"recall": 100, "precision": 100},
                 "vest": {"recall": 0, "precision": 0}}


def test_epoch_lines_from_rfdetr_metrics_csv(tmp_path):
    (tmp_path / "metrics.csv").write_text("epoch,step,train/loss_epoch,val/loss,val/mAP_50\n"
                                          "0,3,,5.1,0.12\n0,3,6.2,,\n1,7,,4.0,0.31\n1,7,4.8,,\n2,9,,3.0,0.5\n")
    # epoch 3 has its val row but not yet its train loss: not reported yet
    assert train_job.epoch_lines(tmp_path / "metrics.csv", 10) == [(1, 6.2, 0.12), (2, 4.8, 0.31)]
    assert train_job.epoch_lines(tmp_path / "none.csv", 10) == []


def test_only_hub_datasets():
    assert train_job.parse_location("hf://devx/safety-detector-dataset/r14") == ("devx/safety-detector-dataset", "r14")
    for bad in ("pod:/data/datasets/r9", "hf://x", "hf://a/b/c/d"):
        with pytest.raises(ValueError):
            train_job.parse_location(bad)


def test_the_real_trainer_api_without_a_gpu(tmp_path, monkeypatch):
    """The service around train_job.py: one job at a time, hub datasets only; a finished job serves its weights."""
    monkeypatch.setenv("TRAINER_JOBS_DIR", str(tmp_path))
    import importlib

    import services.trainer_rfdetr as svc
    svc = importlib.reload(svc)
    from fastapi.testclient import TestClient
    c = TestClient(svc.app)
    h = {"Authorization": "Bearer fake-trainer-key"}
    assert c.post("/v1/jobs", json={"dataset": "pod:/data/r9", "classes": ["a"]}, headers=h).status_code == 422

    class Done:  # stands in for a train_job.py subprocess that already finished
        returncode = 0

        def poll(self):
            return 0
    monkeypatch.setattr(svc.subprocess, "Popen", lambda *a, **k: Done())
    j = c.post("/v1/jobs", json={"dataset": "hf://o/r/r1", "classes": ["a"], "epochs": 2}, headers=h).json()
    d = tmp_path / j["id"]
    assert json.loads((d / "job.json").read_text())["classes"] == ["a"]
    (d / "status.json").write_text(json.dumps({"status": "running", "stage": "training", "progress": 0.5, "error": None}))
    assert c.post("/v1/jobs", json={"dataset": "hf://o/r/r1", "classes": ["a"]}, headers=h).status_code == 409
    w = tmp_path / "w.pth"
    w.write_bytes(bytes(range(256)) * 100)
    (d / "result.json").write_text(json.dumps({"weights": str(w), "sha256": "x", "bytes": w.stat().st_size,
                                               "args": {"classes": ["a"]}, "metrics": {"split": "valid", "classes": {"a": {"recall": 1, "precision": 2}}}}))
    (d / "status.json").write_text(json.dumps({"status": "done", "stage": "done", "progress": 1, "error": None}))
    assert c.get(f"/v1/jobs/{j['id']}", headers=h).json()["result_model"] == "detector_r2_medium"
    assert c.get(f"/v1/jobs/{j['id']}/metrics", headers=h).json()["current"] is None
    r = c.get(f"/v1/jobs/{j['id']}/weights/file", headers={**h, "Range": "bytes=0-99"})
    assert r.status_code == 206 and r.content == w.read_bytes()[:100]
