"""The pod's tunnel agent: one WebSocket tunnel to the gateway, opened by whichever side reaches the other first.

- dial out:  the agent opens wss://<gateway>/tunnel (when the gateway is reachable from the internet)
- listen:    the agent accepts ws://<pod>:<port>/tunnel (exposed by RunPod as wss://<podId>-<port>.proxy.runpod.net),
             for a gateway that can't be reached (behind NAT on site)

Both carry the static key (Authorization: Bearer <TUNNEL_KEY>), and only one tunnel is up at a time. On a tunnel the
gateway opened, the agent first sends a hello frame {"hello": {"pod", "services"}} (on a dial-out the same facts go
in the X-Pod-Id / X-Pod-Services headers). Then, for each request frame
  {"id", "service", "method", "path", "headers", "body_b64"}
it calls that local service (e.g. http://127.0.0.1:9092) and sends back
  {"id", "status", "headers", "body_b64"}   or   {"id", "error"} when the local service doesn't answer.
Requests run concurrently. A dropped dial-out is retried with backoff; the listener keeps listening.
"""

import asyncio
import base64
import hmac
import json
import logging

import httpx
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.exceptions import InvalidStatus
from websockets.http11 import Response

log = logging.getLogger("runpod.tunnel")
MAX_FRAME = 64 * 1024 * 1024  # a frame carries one whole request or response (base64)
DROP_HEADERS = {"content-length", "content-encoding", "transfer-encoding", "connection", "keep-alive"}
PING_S = 20  # keeps a tunnel open through proxies that close idle connections


class TunnelAgent:
    def __init__(self, gateway_url, key, pod_id, services, max_backoff=10.0, request_timeout=300.0,
                 listen_port=None, listen_host="0.0.0.0"):
        """services: name -> local base URL, e.g. {"trainer": "http://127.0.0.1:9092"}. gateway_url None: no dial-out;
        listen_port: also accept the gateway's tunnel on this port (0 = any free port; see .listen_port after start)."""
        self.gateway_url, self.key, self.pod_id = gateway_url, key, pod_id
        self.services = dict(services)
        self.max_backoff, self.request_timeout = max_backoff, request_timeout
        self.listen_port, self.listen_host = listen_port, listen_host
        self.connected = asyncio.Event()
        self.refused = None  # the last handshake status when the gateway refused us (e.g. 403: wrong key)
        self.side = None  # "pod" (we dialed) or "gateway" (it dialed us) while a tunnel is up
        self.listening = asyncio.Event()

    async def run(self):
        """Listen and / or dial, serve, reconnect, until cancelled."""
        async with httpx.AsyncClient(timeout=self.request_timeout) as http:
            self.http = http
            tasks = []
            if self.listen_port is not None:
                tasks.append(asyncio.create_task(self._listen()))
            if self.gateway_url:
                tasks.append(asyncio.create_task(self._dial()))
            try:
                await asyncio.gather(*tasks)
            finally:
                for t in tasks:
                    t.cancel()

    # ---- dial out ---------------------------------------------------------------------------------------------
    async def _dial(self):
        headers = {"Authorization": f"Bearer {self.key}", "X-Pod-Id": self.pod_id,
                   "X-Pod-Services": ",".join(self.services)}
        backoff = 0.5
        while True:
            if self.side == "gateway":  # the gateway reached us first: no second tunnel
                await asyncio.sleep(1.0)
                continue
            try:
                async with connect(self.gateway_url, additional_headers=headers, max_size=MAX_FRAME,
                                   open_timeout=10, ping_interval=PING_S) as ws:
                    self.refused, backoff = None, 0.5
                    if self.side is not None:  # the gateway's tunnel came up meanwhile
                        await ws.close(code=1000)
                        continue
                    await self._serve(ws, "pod")
            except InvalidStatus as e:
                self.refused = e.response.status_code
                log.warning("tunnel refused by the gateway (HTTP %s): check TUNNEL_KEY", self.refused)
            except Exception as e:  # noqa: BLE001 - any drop (gateway down, network, closed socket): retry
                log.info("tunnel down (%s), retrying in %.1f s", type(e).__name__, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self.max_backoff)

    # ---- listen -----------------------------------------------------------------------------------------------
    def _check(self, connection, request):
        """Before the handshake: the path, the key, and only one tunnel at a time."""
        if request.path.split("?")[0] != "/tunnel":
            return connection.respond(404, "no such path\n")
        given = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(given.encode(), self.key.encode()):
            log.warning("tunnel from the gateway refused: wrong key")
            return connection.respond(403, "wrong key\n")
        if self.side is not None:
            return Response(409, "Conflict", Headers([("Content-Type", "text/plain")]), b"a tunnel is already up\n")
        return None

    async def _listen(self):
        async with serve(self._accepted, self.listen_host, self.listen_port, process_request=self._check,
                         max_size=MAX_FRAME, ping_interval=PING_S) as server:
            self.listen_port = server.sockets[0].getsockname()[1]
            self.listening.set()
            log.info("listening for the gateway's tunnel on %s:%s/tunnel", self.listen_host, self.listen_port)
            await asyncio.Future()  # until cancelled

    async def _accepted(self, ws):
        if self.side is not None:  # lost a race with a dial-out that came up after the check
            await ws.close(code=1013)
            return
        await ws.send(json.dumps({"hello": {"pod": self.pod_id, "services": list(self.services)}}))
        try:
            await self._serve(ws, "gateway")
        except Exception as e:  # noqa: BLE001 - the gateway went away
            log.info("tunnel from the gateway closed (%s)", type(e).__name__)

    # ---- one tunnel, whoever opened it ------------------------------------------------------------------------
    async def _serve(self, ws, side):
        self.side = side
        self.connected.set()
        log.info("tunnel up (%s opened it): %s, %s", side, self.pod_id, ", ".join(self.services))
        tasks = set()
        try:
            async for message in ws:
                t = asyncio.create_task(self._handle(ws, json.loads(message)))
                tasks.add(t)
                t.add_done_callback(tasks.discard)
        finally:
            self.side = None
            self.connected.clear()
            log.info("tunnel closed (%s side)", side)

    async def _handle(self, ws, frame):
        rid, base = frame.get("id"), self.services.get(frame.get("service"))
        try:
            if base is None:
                raise LookupError(f"this pod has no {frame.get('service')} service")
            r = await self.http.request(frame["method"], base + frame["path"], headers=frame.get("headers") or {},
                                        content=base64.b64decode(frame.get("body_b64") or ""))
            reply = {"id": rid, "status": r.status_code, "body_b64": base64.b64encode(r.content).decode(),
                     "headers": {k: v for k, v in r.headers.items() if k.lower() not in DROP_HEADERS}}
        except Exception as e:  # noqa: BLE001 - reported to the gateway, which answers 502
            reply = {"id": rid, "error": f"{type(e).__name__}: {e}"}
        await ws.send(json.dumps(reply))
