"""The RunPod pod server: start the pod's services and the tunnel agent that connects them to the gateway.

  python server.py inference     detector (:9091), pod id "inference"
  python server.py training      trainer (:9092), RF-DETR pre-labeller (:9096), VLM (:9000, vLLM started on demand),
                                 pod id "training": the GPU pod
  python server.py all           both pods, for local development (two agents, two tunnels)

The services are fake (services/), except the real RF-DETR trainer (--trainer rfdetr, on a GPU pod). To use another
real one, point the agent at it instead of starting the fake:
  python server.py inference --service detector=http://127.0.0.1:8091

The tunnel is opened by whichever side reaches the other first:
  --gateway wss://<gateway>/tunnel   the agent dials the gateway (default GATEWAY_TUNNEL_URL, else local :8090)
  --listen 8765                      the agent also accepts the gateway's tunnel on this port (RunPod: expose 8765/http)
  --no-dial                          only listen (the gateway can't be reached from the pod, e.g. it is behind NAT)
  python server.py training --listen 8765 --no-dial --trainer rfdetr      # a real training pod

Settings (environment or .env.example): TUNNEL_KEY, TRAINER_KEY, HF_TOKEN (the trainer's, read-only), POD_ID
(optional; RUNPOD_POD_ID on RunPod). A command started over SSH doesn't get the pod's RunPod environment, so these
are also read from the container's start environment, then from .env next to this file (install.sh writes it). The services bind 127.0.0.1; only the agent calls them.
"""

import argparse
import asyncio
import importlib
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tunnel_agent import TunnelAgent  # noqa: E402

# the training pod also pre-labels on its GPU: RF-DETR (services/labeller.py, real, needs rfdetr) and the VLM
# (services/vlm.py: real vLLM started on demand where it is installed, else the fake vLLM services/vllm.py)
PODS = {"inference": {"detector": 9091}, "training": {"trainer": 9092, "labeller": 9096, "vllm": 9000}}
MODULES = {"detector": "services.detector", "vllm": "services.vlm", "trainer": "services.trainer",
           "labeller": "services.labeller"}
POD_ENV = ("TUNNEL_KEY", "TRAINER_KEY", "HF_TOKEN", "RUNPOD_POD_ID")


ENV_FILE = Path(__file__).resolve().parent / ".env"  # written by install.sh (mode 600), the keys it was given


def pod_environment():
    """The keys this server uses, first found wins: its own environment; the container's start environment (process 1:
    the RunPod console's env vars, which a command started over SSH doesn't inherit); then .env from install.sh."""
    found = []
    try:
        found += Path("/proc/1/environ").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        pass
    try:
        found += ENV_FILE.read_text().splitlines()
    except OSError:
        pass
    for item in found:
        k, _, v = item.partition("=")
        if k in POD_ENV and v and not os.environ.get(k):
            os.environ[k] = v


def start_services(ports):
    """Start fake services (name -> port, 0 = any free port) in threads; returns name -> (server, thread, url)."""
    out = {}
    for name, port in ports.items():
        app = importlib.import_module(MODULES[name]).app
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
        thread = threading.Thread(target=server.run, name=f"pod-{name}", daemon=True)
        thread.start()
        out[name] = [server, thread, None]
    t0 = time.time()
    while not all(s.started for s, _, _ in out.values()):
        if time.time() - t0 > 10 or not all(t.is_alive() for _, t, _ in out.values()):
            raise RuntimeError("a pod service did not start (port in use?)")
        time.sleep(0.02)
    for v in out.values():
        v[2] = f"http://127.0.0.1:{v[0].servers[0].sockets[0].getsockname()[1]}"
    return out


class AgentThread:
    """A tunnel agent on its own event loop in a background thread; stop() / start() simulate the pod going away."""

    def __init__(self, agent):
        self.agent, self.loop, self.task, self.thread = agent, None, None, None

    def start(self):
        ready = threading.Event()

        def main():
            self.loop = asyncio.new_event_loop()
            self.agent.connected = asyncio.Event()  # bound to this loop
            self.task = self.loop.create_task(self.agent.run())
            ready.set()
            try:
                self.loop.run_until_complete(self.task)
            except asyncio.CancelledError:
                pass
            finally:
                self.loop.close()
        self.thread = threading.Thread(target=main, name=f"tunnel-{self.agent.pod_id}", daemon=True)
        self.thread.start()
        ready.wait()
        return self

    def stop(self):
        if self.loop and self.task and not self.task.done():
            self.loop.call_soon_threadsafe(self.task.cancel)
        if self.thread:
            self.thread.join(10)

    def wait_connected(self, timeout=10):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.agent.connected.is_set():
                return True
            time.sleep(0.05)
        return False


class Pods:
    def __init__(self, services, agents):
        self.services, self.agents = services, agents  # name -> [server, thread, url]; pod id -> AgentThread
        self.urls = {n: v[2] for n, v in services.items()}
        self.modules = {n: importlib.import_module(MODULES[n]) for n in services}

    def stop(self):
        for a in self.agents.values():
            a.stop()
        for server, thread, _ in self.services.values():
            server.should_exit = True
            thread.join(5)


def start(pods, gateway_url, key, ports=None, overrides=None, pod_ids=None, listen=None):
    """Start these pods' fake services (except the overridden ones) and one tunnel agent per pod. gateway_url None:
    no dial-out; listen: pod -> port where its agent also accepts the gateway's tunnel (0 = a free port)."""
    overrides, pod_ids, listen = overrides or {}, pod_ids or {}, listen or {}
    wanted = {s: (ports or {}).get(s, p) for pod in pods for s, p in PODS[pod].items() if s not in overrides}
    services = start_services(wanted)
    agents = {}
    for pod in pods:
        urls = {s: overrides.get(s) or services[s][2] for s in PODS[pod]}
        agent = TunnelAgent(gateway_url, key, pod_ids.get(pod, pod), urls, listen_port=listen.get(pod))
        agents[pod] = AgentThread(agent).start()
    return Pods(services, agents)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pod", choices=[*PODS, "all"])
    p.add_argument("--gateway", default=os.environ.get("GATEWAY_TUNNEL_URL", "ws://127.0.0.1:8090/tunnel"))
    p.add_argument("--service", action="append", default=[], metavar="NAME=URL", help="use a real service instead of the fake")
    p.add_argument("--listen", type=int, metavar="PORT", help="also accept the gateway's tunnel on this port")
    p.add_argument("--no-dial", action="store_true", help="don't dial the gateway (only listen)")
    p.add_argument("--trainer", choices=["fake", "rfdetr"], default="fake", help="rfdetr: the real trainer (GPU pod)")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    pod_environment()
    if args.trainer == "rfdetr":
        MODULES["trainer"] = "services.trainer_rfdetr"
    pods = list(PODS) if args.pod == "all" else [args.pod]
    if args.no_dial and args.listen is None:
        p.error("--no-dial needs --listen: the tunnel has to be opened from one side")
    overrides = dict(s.split("=", 1) for s in args.service)
    pod_id = os.environ.get("POD_ID") or os.environ.get("RUNPOD_POD_ID")
    pod_ids = {pods[0]: pod_id} if pod_id and len(pods) == 1 else {}
    listen = {pods[0]: args.listen} if args.listen is not None and len(pods) == 1 else {}
    if args.listen is not None and len(pods) > 1:
        p.error("--listen needs one pod (inference or training)")
    gateway = None if args.no_dial else args.gateway
    if args.listen is not None and os.environ.get("TUNNEL_KEY", "fake-tunnel-key") == "fake-tunnel-key":
        p.error("--listen opens a public port: set TUNNEL_KEY (a long random value) in the pod's environment first")
    running = start(pods, gateway, os.environ.get("TUNNEL_KEY", "fake-tunnel-key"), overrides=overrides,
                    pod_ids=pod_ids, listen=listen)
    for name, url in running.urls.items():
        print(f"{name:10s} {url} ({MODULES[name]}, local only)")
    print(f"tunnel: {'dial ' + gateway if gateway else 'no dial-out'}"
          f"{f' · listening on :{args.listen}/tunnel' if args.listen is not None else ''} · pods: {', '.join(pods)}")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))  # kill (deploy.sh's restart): exit cleanly, vLLM included
    try:
        while True:
            time.sleep(3600)
    except (KeyboardInterrupt, SystemExit):
        running.stop()


if __name__ == "__main__":
    main()
