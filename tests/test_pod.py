"""The pod server: its (fake) services and the tunnel agent. python -m pytest -q (from runpod/)."""

import asyncio
import base64
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest
from websockets.asyncio.server import serve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server  # noqa: E402
from services.common import KEYS  # noqa: E402


class FakeGateway:
    """A tiny /tunnel endpoint: checks the key, then sends request frames to the agent and collects the replies."""

    def __init__(self, key="k"):
        self.key, self.pods, self.replies = key, [], asyncio.Queue()
        self.loop, self.port, self.ws = None, None, {}  # pod id -> socket

    async def _handler(self, ws):
        self.pods.append((ws.request.headers["X-Pod-Id"], ws.request.headers["X-Pod-Services"]))
        self.ws[ws.request.headers["X-Pod-Id"]] = ws
        async for m in ws:
            await self.replies.put(json.loads(m))

    def _check(self, conn, request):
        if request.headers.get("Authorization") != f"Bearer {self.key}":
            return conn.respond(403, "bad key\n")

    def start(self):
        ready = threading.Event()

        async def main():
            async with serve(self._handler, "127.0.0.1", 0, process_request=self._check) as srv:
                self.port = srv.sockets[0].getsockname()[1]
                ready.set()
                await asyncio.Future()
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=lambda: self.loop.run_until_complete(main()), daemon=True).start()
        ready.wait(5)
        return self

    def ask(self, pod, frame, timeout=10):
        async def go():
            await self.ws[pod].send(json.dumps(frame))
            return await asyncio.wait_for(self.replies.get(), timeout)
        return asyncio.run_coroutine_threadsafe(go(), self.loop).result(timeout + 1)


@pytest.fixture(scope="module")
def gateway():
    return FakeGateway().start()


@pytest.fixture(scope="module")
def pods(gateway):
    import services.trainer
    services.trainer.STEP_SECONDS = 0
    running = server.start(["inference", "training"], f"ws://127.0.0.1:{gateway.port}/tunnel", "k",
                           ports={"detector": 0, "vllm": 0, "trainer": 0})
    assert all(a.wait_connected() for a in running.agents.values())
    yield running
    running.stop()


def bearer(service):
    return {"Authorization": f"Bearer {KEYS[service]}"}


def test_agents_connect_with_the_key(gateway, pods):
    assert sorted(gateway.pods) == [("inference", "detector"), ("training", "trainer,labeller,vllm")]


def test_request_through_the_tunnel(gateway, pods):
    rep = gateway.ask("inference", {"id": "1", "service": "detector", "method": "POST", "path": "/v1/detect",
                       "headers": {**bearer("detector"), "content-type": "image/jpeg", "x-timestamp": "7"},
                       "body_b64": base64.b64encode(b"\xff\xd8jpeg").decode()})
    body = json.loads(base64.b64decode(rep["body_b64"]))
    assert rep["status"] == 200 and body["alerts"][0]["t"] == 7
    rep = gateway.ask("inference", {"id": "2", "service": "nope", "method": "GET", "path": "/", "headers": {}, "body_b64": ""})
    assert "no nope service" in rep["error"]


def test_wrong_key_is_refused(gateway):
    t = server.AgentThread(server.TunnelAgent(f"ws://127.0.0.1:{gateway.port}/tunnel", "wrong", "x", {"vllm": "http://127.0.0.1:9"})).start()
    try:
        for _ in range(100):
            if t.agent.refused:
                break
            threading.Event().wait(0.05)
        assert t.agent.refused == 403 and not t.agent.connected.is_set()
    finally:
        t.stop()


# ---- the fake services themselves (local only; the gateway reaches them through the tunnel) ---------------------
def test_detector(pods):
    u = pods.urls["detector"]
    assert httpx.get(f"{u}/health").json()["status"] == "ok"
    assert httpx.get(f"{u}/v1/alerts").status_code == 401
    assert httpx.post(f"{u}/v1/alerts/prompt/activate", json={"version": 9}, headers=bearer("detector")).status_code == 404


def test_vllm(pods):
    u, h = pods.urls["vllm"], bearer("trainer")
    assert httpx.get(f"{u}/v1/models", headers=h).json()["data"][0]["id"] == "Qwen/Qwen3.5-9B"
    httpx.post(f"{u}/v1/vlm/start", headers=h)  # on demand: the fake vLLM, as a subprocess here
    for _ in range(150):
        if httpx.get(f"{u}/v1/vlm/status", headers=h).json()["state"] == "ready":
            break
        threading.Event().wait(0.1)
    try:
        r = httpx.post(f"{u}/v1/chat/completions", headers=h, json={"model": "Qwen/Qwen3.5-9B", "messages": [
            {"role": "system", "content": "Classes, one per line:\n- helmet: a hard hat\n- dozer\nIf a clearly …"},
            {"role": "user", "content": [{"type": "text", "text": "Return the boxes as JSON"}]}]})
        boxes = json.loads(r.json()["choices"][0]["message"]["content"])["boxes"]
        assert [b["class"] for b in boxes] == ["helmet", "dozer"]  # the classes the prompt lists, nothing built in
    finally:
        httpx.post(f"{u}/v1/vlm/stop", headers=h)


def test_trainer(pods):
    u, h = pods.urls["trainer"], bearer("trainer")
    assert httpx.post(f"{u}/v1/jobs", json={"dataset": "hf://x", "classes": ["helmet"]}).status_code == 401
    assert httpx.post(f"{u}/v1/jobs", json={"dataset": "hf://x"}, headers=h).status_code == 422  # classes are required
    assert httpx.post(f"{u}/v1/jobs", json={"dataset": "hf://x", "classes": []}, headers=h).status_code == 422
    j = httpx.post(f"{u}/v1/jobs", json={"dataset": "hf://x", "pod_id": "p1", "classes": ["helmet", "dozer"]}, headers=h).json()
    assert j["status"] == "done" and j["result_model"] == "detector_r2_medium"
    m = httpx.get(f"{u}/v1/jobs/{j['id']}/metrics", headers=h).json()
    assert set(m["classes"]) == set(m["current"]) == {"helmet", "dozer"}  # scored on the job's classes only


def test_trainer_reports_classes_and_serves_weights_in_ranges(pods, tmp_path):
    import hashlib

    import services.trainer as trainer
    u, h = pods.urls["trainer"], bearer("trainer")
    classes = ["truck", "person", "equipment", "dozer"]
    j = httpx.post(f"{u}/v1/jobs", json={"dataset": "hf://x/r9", "classes": classes}, headers=h).json()
    weights = tmp_path / "w.pth"
    weights.write_bytes(bytes(range(256)) * 4000)  # ~1 MB
    old, trainer.WEIGHTS_PATH = trainer.WEIGHTS_PATH, str(weights)
    try:
        info = httpx.get(f"{u}/v1/jobs/{j['id']}/weights", headers=h).json()
        assert info["args"]["classes"] == classes and info["bytes"] == weights.stat().st_size
        assert info["sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
        part = httpx.get(f"{u}/v1/jobs/{j['id']}/weights/file", headers={**h, "Range": "bytes=1000-1999"})
        assert part.status_code == 206 and part.content == weights.read_bytes()[1000:2000]
    finally:
        trainer.WEIGHTS_PATH = old
    assert "dozer" in " ".join(httpx.get(f"{u}/v1/jobs/{j['id']}/logs", headers=h).json()["lines"])
