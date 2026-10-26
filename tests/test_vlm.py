"""The pod's VLM service: vLLM started on demand (the fake vLLM stands in), passed through when ready, stopped by the
gateway or when idle; it and a training job never hold the GPU together. python -m pytest -q (from runpod/)."""

import importlib
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
H = {"Authorization": "Bearer fake-trainer-key"}


@pytest.fixture()
def vlm(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_LOG", str(tmp_path / "vllm.log"))
    monkeypatch.setenv("VLLM_BIN", str(tmp_path / "no-vllm"))  # not installed here: the fake vLLM runs
    import services.vlm as mod
    mod = importlib.reload(mod)
    yield mod, TestClient(mod.app)
    mod.stop("test done")


def until_state(c, *states, timeout=15):
    t0 = time.time()
    while (st := c.get("/v1/vlm/status", headers=H).json())["state"] not in states:
        assert time.time() - t0 < timeout, st
        time.sleep(0.1)
    return st


def test_start_chat_stop(vlm):
    mod, c = vlm
    assert c.post("/v1/vlm/start").status_code == 401  # the trainer key is needed
    assert c.get("/health").json()["state"] == "stopped" and c.get("/health").json()["engine"] == "fake"
    assert c.get("/v1/models", headers=H).json()["data"][0]["id"] == "Qwen/Qwen3.5-9B"  # listed while stopped
    body = {"model": "Qwen/Qwen3.5-9B", "messages": [{"role": "user", "content": "hi"}]}
    r = c.post("/v1/chat/completions", json=body, headers=H)
    assert r.status_code == 503 and "start it first" in r.text
    assert c.post("/v1/vlm/start", headers=H).json()["state"] in ("starting", "ready")
    until_state(c, "ready")
    assert c.post("/v1/vlm/start", headers=H).json()["state"] == "ready"  # already up: nothing new started
    r = c.post("/v1/chat/completions", json=body, headers=H)
    assert r.status_code == 200 and r.json()["choices"][0]["message"]["content"]
    assert c.post("/v1/vlm/stop", headers=H).json() == {**c.get("/v1/vlm/status", headers=H).json(), "state": "stopped"}
    assert "starting Qwen/Qwen3.5-9B (fake, prelabel)" in Path(mod.LOG).read_text()


def test_a_failed_start_shows_the_log(vlm, monkeypatch):
    mod, c = vlm
    monkeypatch.setenv("VLLM_CMD", "python3 -c \"print('CUDA out of memory'); raise SystemExit(3)\"")
    c.post("/v1/vlm/start", headers=H)
    st = until_state(c, "failed")
    assert "code 3" in st["detail"] and "CUDA out of memory" in st["detail"]


def test_the_gpu_is_training_or_vlm_never_both(vlm, tmp_path, monkeypatch):
    mod, c = vlm
    from services import gpu
    monkeypatch.setitem(gpu._busy, "training", lambda: "the pod is training train-0001")
    r = c.post("/v1/vlm/start", headers=H)
    assert r.status_code == 409 and "train-0001" in r.text
    monkeypatch.setitem(gpu._busy, "training", lambda: None)

    released = []
    monkeypatch.setattr(gpu, "_release", [lambda: released.append(1)])
    c.post("/v1/vlm/start", headers=H)
    assert released == [1]  # the labeller let go of its models first
    import services.trainer as trainer
    t = TestClient(trainer.app)
    r = t.post("/v1/jobs", json={"dataset": "hf://o/r/r1", "classes": ["a"]}, headers=H)
    assert r.status_code == 409 and "vLLM is running" in r.text


def test_idle_stop(vlm, monkeypatch):
    mod, c = vlm
    monkeypatch.setattr(mod, "IDLE_S", 0.5)
    c.post("/v1/vlm/start", headers=H)
    until_state(c, "ready")
    st = until_state(c, "stopped", timeout=10)
    assert "without a request" in st["detail"]


def test_inference_mode_keeps_rfdetr_beside_vllm(vlm, monkeypatch):
    mod, c = vlm
    from services import gpu
    released = []
    monkeypatch.setattr(gpu, "_release", [lambda: released.append(1)])
    st = c.post("/v1/vlm/start", json={"mode": "inference"}, headers=H).json()
    assert st["mode"] == "inference" and released == []  # the labeller keeps its model
    until_state(c, "ready")
    assert gpu.busy("vlm-exclusive") is None and "inference" in gpu.busy("vlm")  # RF-DETR may run, training may not
    assert "(fake, inference)" in Path(mod.LOG).read_text()
    assert c.post("/v1/vlm/start", json={"mode": "inference"}, headers=H).json()["state"] == "ready"  # kept alive
    st = c.post("/v1/vlm/start", headers=H).json()  # pre-label mode: restarted, and the labeller lets go first
    assert st["mode"] == "prelabel" and released == [1] and "pre-labelling" in gpu.busy("vlm-exclusive")


def test_inference_flags_quantize_and_allow_a_clip(vlm, monkeypatch):
    mod, _ = vlm
    monkeypatch.setattr(mod, "real_bin", lambda: Path("/opt/vllm"))
    cmd = mod.command(8000, "inference")
    assert cmd[cmd.index("--quantization") + 1] == "fp8" and cmd[cmd.index("--limit-mm-per-prompt") + 1] == '{"image": 8}'
    assert float(cmd[cmd.index("--gpu-memory-utilization") + 1]) <= 0.75
    assert "--quantization" not in mod.command(8000, "prelabel")
