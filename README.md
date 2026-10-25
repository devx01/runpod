# RunPod pod server

What runs **on a RunPod GPU pod**: the pod's services, plus a tunnel agent that connects them to the API gateway.
Every call from the gateway to the pod rides **one WebSocket tunnel**, opened by whichever side reaches the other
first:
- **the pod dials** `wss://<gateway>/tunnel` (the gateway is reachable from the internet), or
- **the gateway dials** the agent's port, `--listen 8765` (RunPod: expose `8765/http`, reached as
  `wss://<podId>-8765.proxy.runpod.net/tunnel`), when the gateway is behind NAT, e.g. on your Mac.

The services bind 127.0.0.1; only the tunnel port is public. The services are **fake** (`services/`), except the real
RF-DETR trainer (`--trainer rfdetr`, `services/trainer_rfdetr.py` + `train_job.py`), all with the same APIs:

| Pod | Services | Real later |
|---|---|---|
| `inference` | `detector` (:9091, detector + alert pipeline) | RF-DETR detector `serve` |
| `training` | `trainer` (:9092, jobs, logs, metrics, weights) | **real:** `--trainer rfdetr` fine-tunes RF-DETR on the GPU; the pod is rented per job, or one you set up |
| `training` | `labeller` (:9096, RF-DETR pre-labelling) | **real** everywhere: `services/labeller.py` runs the gateway's own `labelling/detector.py` on the GPU (CPU / MPS locally) |
| `training` | `vllm` (:9000, the VLM: OpenAI-compatible + start / status / stop) | **real** where vLLM is installed (`deploy.sh` with `VLM=1`): `services/vlm.py` starts vLLM with Qwen3.5-9B on demand; elsewhere it starts the fake vLLM (`services/vllm.py`) |

The services bind 127.0.0.1 only; only the agent calls them.

## Run
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python server.py all          # both pods, for local development (or: inference / training)
```
Settings come from the environment (see `.env.example`):
- `GATEWAY_TUNNEL_URL`: where to dial, default `ws://127.0.0.1:8090/tunnel`; `wss://<gateway>/tunnel` for real
- `TUNNEL_KEY`: the gateway's static key (sample `fake-tunnel-key`)
- `POD_ID`: optional; overrides the pod id the gateway shows

The agent retries with backoff, so the pods can start before the gateway. To use another real service instead of a
fake:
```bash
.venv/bin/python server.py inference --service detector=http://127.0.0.1:8091
```
To see the gateway register a new label from a VLM answer, start the pods with an extra class in the fake VLM's
pre-label answer:
```bash
FAKE_VLM_EXTRA_CLASS=vest .venv/bin/python server.py all
```

## A real training pod on RunPod
1. **On the pod** (RunPod console → edit pod): expose HTTP port `8765` (next to `8888`), and set the environment
   variables `TUNNEL_KEY` and `TRAINER_KEY` (long random values, also exported in the gateway's shell) and `HF_TOKEN`
   (a **read-only** Hugging Face token: the pod only downloads the dataset). The pod restarts.
2. **Deploy** from this repo: `make deploy-pod POD_SSH="root@<ip> -p <port>"` (the pod's "SSH over exposed TCP").
   - It copies `runpod/`, installs `requirements-train.txt` into a venv that keeps the image's CUDA torch, and starts
     `server.py training --listen 8765 --no-dial --trainer rfdetr` (log: `~/runpod/pod.log` on the pod).
   - The pod has no volume, so run it again after every pod restart.
3. **The gateway**, from the shell with your keys: `TRAINING_POD=manual`,
   `POD_TUNNEL_URLS=wss://<podId>-8765.proxy.runpod.net/tunnel`, `TUNNEL_KEY`, `TRAINER_KEY`. It dials the pod and the
   training page runs jobs on it.

A job (`train_job.py`, a subprocess, one job at a time):
1. **Download** `hf://<org>/<repo>/<rid>` with `HF_TOKEN`. `pod:/…` locations are refused (422).
2. **Prepare:** the classes come from the COCO file. A dataset without a `valid` split (a short video) gets every
   5th training image as valid; the log says the scores are optimistic then.
3. **Train:** `RFDETR<Size>().train(dataset_dir, epochs, batch_size=4, grad_accum_steps=4)`. Each epoch's loss and
   val mAP50 go to the log (the training page's chart).
4. **Evaluate:** the best checkpoint on valid, recall / precision per class at IoU 0.5, confidence 0.5. There's no
   current model on the pod, so `current` is empty.
5. **Done:** the gateway fetches `checkpoint_best_total.pth` in 4 MB chunks and pushes it to Hugging Face.

**Pre-labelling on the pod** (`services/labeller.py`, key `TRAINER_KEY`): the gateway sends one frame (JPEG) per
request and gets the boxes back, computed with the same code and rule as on the gateway (`detector.prelabel_frame`:
full frame, plus tiles for labels marked small). The weights get there once, cached by SHA-256 in `~/runpod-models`:
a Hub model is fetched by the pod itself (`POST /v1/models/fetch`, its read-only `HF_TOKEN`, pinned to the commit);
the shipped or a local model is uploaded by the gateway in 4 MB chunks (`PUT /v1/models/{sha256}` with
`Content-Range`, resumable, checked against the SHA-256). `deploy.sh` copies `gateway/labelling/detector.py` to
`~/runpod/shared/`.

**The VLM on the pod** (`services/vlm.py`, service `vllm`, key `TRAINER_KEY`): vLLM takes most of a 24 GB GPU's memory,
so it runs only while it is needed. For a VLM pre-label run, the gateway calls `POST /v1/vlm/start`, polls
`GET /v1/vlm/status` until `ready`, sends one frame per chat request, and calls `POST /v1/vlm/stop` at the end (also
when the run fails). The pod refuses to start vLLM while a training job runs, and the trainer refuses a job while vLLM
is up (`services/gpu.py`). Before vLLM starts, the RF-DETR labeller lets go of its cached models. vLLM also stops after
10 minutes without a request, in case the gateway goes away. That only stops vLLM, never the pod.
- **Where it lives:** vLLM is in its own venv at `~/vllm-venv` (its torch differs from rfdetr's), and the model in
  `~/hf`, both on the container disk (50 GB+); after a pod restart, `deploy.sh` with `VLM=1` installs them again.
- **Its log:** `~/runpod/vllm.log`.
- **Settings:** `VLLM_MODEL`, `VLLM_IDLE_S`, `VLLM_BIN`, and `VLLM_CMD` to replace the command.

The pod is **never stopped or terminated automatically**: do that in the RunPod console. The job folders (dataset,
output) stay on the pod until you terminate it.

## The tunnel (`tunnel_agent.py`)
1. **Connect:** the agent opens `GATEWAY_TUNNEL_URL` with these headers:
   - `Authorization: Bearer <TUNNEL_KEY>`
   - `X-Pod-Id`
   - `X-Pod-Services: detector,vllm`
2. **Key check:** the gateway compares the key to its own `TUNNEL_KEY`. A wrong key or an unknown service is refused
   (HTTP 403). The agent logs it and keeps retrying.
3. **Requests:** the gateway sends JSON frames `{id, service, method, path, headers, body_b64}`.
   - The agent calls that local service and answers `{id, status, headers, body_b64}`.
   - If the local service doesn't answer, it sends `{id, error}` and the gateway replies 502.
   - Many requests run at once on the one socket.
   - On a tunnel the gateway opened, the agent first sends `{"hello": {"pod", "services"}}` (on a dial-out the same
     facts are the `X-Pod-Id` / `X-Pod-Services` headers).
   - **One tunnel at a time:** while one is up, the agent refuses the gateway's dial (409) and doesn't dial out; the
     gateway refuses a second tunnel from the same pod. Both sides ping every 20 s (RunPod's proxy closes idle
     connections).
4. **Drops:** if the connection drops, the gateway fails that pod's open requests, and the agent reconnects.

The service keys (`DETECTOR_KEY`, `TRAINER_KEY`) still apply inside the tunnel: the gateway sends them, and the
services check them.

**Training pods** are rented per job (`TRAINING_POD=rent`), or the one you set up (`TRAINING_POD=manual`).
- A rented pod: the gateway rents it through the RunPod API with `ports: ["8765/http"]` and passes
  `GATEWAY_TUNNEL_URL`, `TUNNEL_KEY` and `POD_ID` in its environment; with the real API it also dials
  `wss://<podId>-8765.proxy.runpod.net/tunnel`.
- When the tunnel is up (either way), the gateway starts the training job on it.
- The gateway waits up to `POD_BOOT_TIMEOUT`; after that it marks the job failed. It never stops the pod.
- For a real deployment, keep the key in a RunPod secret rather than plain pod env, and use one key per pod.

**What the trainer must hand back** (the fake does this; a real trainer must too):
- A job receives `classes` (the dataset's class names in class-id order); a real trainer can also read them from the
  dataset's COCO file.
- `GET /v1/jobs/{id}/weights` returns `{name, sha256, bytes, args}`. The gateway saves `args` as the weights'
  `_args.json`: model size, epochs and **`classes`**. This is how the gateway knows a model trained with `dozer`
  detects `dozer`.
- `GET /v1/jobs/{id}/weights/file` must support `Range` requests, because the gateway fetches the file in 4 MB
  chunks through the tunnel.
- The gateway does that as soon as the job is done: it checks the weights, pushes them to the private Hugging Face
  model repo (`HF_MODEL_REPO`), and then stops the pod. The pod itself never needs a write token for the Hub.
- The fake serves `FAKE_TRAINER_WEIGHTS` (`make up-pods` points it at the shipped checkpoint) as its "trained"
  weights. Without it, it serves placeholder bytes, which the gateway rejects ("not an RF-DETR checkpoint").

## Tests
`.venv/bin/python -m pytest -q` (or the gateway's venv). The tests check:
- the agents against a tiny tunnel endpoint: the key check, a request through the tunnel, an unknown service
- the fake services

- the listening side (key, hello, one tunnel at a time) and the real trainer's helpers (valid fallback, per-class
  scores, the epoch log, hub-only datasets, its API with a stand-in subprocess), without a GPU

The gateway's `tests/test_tunnel.py` and `tests/test_dial.py` cover the real round trip, both ways.
