"""Fake vLLM (the OpenAI-compatible server): what services/vlm.py starts where real vLLM isn't installed. No key: a
private 127.0.0.1 port that only services/vlm.py calls."""

import time
import uuid

from pydantic import BaseModel

from .common import make_app, text_of, vlm_answer

app = make_app("vLLM Qwen3.5-9B")
MODEL = "Qwen/Qwen3.5-9B"


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "vllm"}]}


class Chat(BaseModel):
    model: str
    messages: list[dict]
    max_tokens: int | None = None
    temperature: float | None = None


@app.post("/v1/chat/completions")
def chat(body: Chat):
    text = " ".join(text_of(m.get("content", "")) for m in body.messages)
    images = sum(1 for m in body.messages if isinstance(m.get("content"), list)
                 for part in m["content"] if isinstance(part, dict) and part.get("type") == "image_url")
    return {"id": f"chatcmpl-{uuid.uuid4().hex[:12]}", "object": "chat.completion", "created": int(time.time()),
            "model": body.model, "choices": [{"index": 0, "finish_reason": "stop",
                                              "message": {"role": "assistant", "content": vlm_answer(text, images)}}],
            "usage": {"prompt_tokens": len(text) // 4, "completion_tokens": 20, "total_tokens": len(text) // 4 + 20}}
