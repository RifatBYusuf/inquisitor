"""Keep cameras connected and serve live frames to the control room."""

import atexit
import base64
import json
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
from PIL import Image


class CameraStream:
    def __init__(self, address, capture_factory=None):
        self.address = address
        self.capture_factory = capture_factory
        self.id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.frame = None
        self.jpeg = None
        self.sequence = 0
        self.updated = 0.0
        self.captured = None
        self.status = "Connecting"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _open(self):
        if self.capture_factory:
            return self.capture_factory(self.address)
        if self.address.strip().isdigit():
            return cv2.VideoCapture(int(self.address))
        return cv2.VideoCapture(
            self.address,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ],
        )

    def _run(self):
        while not self.stop.is_set():
            capture = None
            try:
                capture = self._open()
                if not capture.isOpened():
                    raise ValueError("Unavailable")
                last_encoded = 0.0
                while not self.stop.is_set():
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError("Unavailable")
                    current = time.monotonic()
                    if current - last_encoded >= 0.10:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        preview = Image.fromarray(rgb)
                        preview.thumbnail((640, 360))
                        ok, encoded = cv2.imencode(
                            ".jpg",
                            cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 75],
                        )
                        if not ok:
                            raise ValueError("Unavailable")
                        with self.lock:
                            self.frame = rgb
                            self.jpeg = base64.b64encode(encoded).decode("ascii")
                            self.sequence += 1
                            self.updated = current
                            self.captured = (
                                datetime.now()
                                .astimezone()
                                .isoformat(timespec="seconds")
                            )
                            self.status = "Live"
                        self.ready.set()
                        last_encoded = current
                    self.stop.wait(0.005)
            except Exception:
                with self.lock:
                    self.status = "Unavailable - reconnecting"
                self.ready.set()
            finally:
                if capture is not None:
                    capture.release()
            self.stop.wait(2.0)

    def snapshot(self, timeout=6.0):
        if not self.ready.wait(timeout):
            raise ValueError("Camera connection timed out")
        with self.lock:
            if (
                self.frame is None
                or self.status != "Live"
                or time.monotonic() - self.updated > 2.0
            ):
                raise ValueError("Camera unavailable")
            return Image.fromarray(self.frame.copy())

    def state(self, previous=0):
        with self.lock:
            live = self.status == "Live" and time.monotonic() - self.updated <= 2.0
            return {
                "id": self.id,
                "sequence": self.sequence,
                "captured": self.captured,
                "status": (
                    self.status
                    if live or self.status != "Live"
                    else "Waiting for video"
                ),
                "jpeg": self.jpeg if live and self.sequence != previous else None,
                "live": live,
            }


class CameraHub:
    def __init__(self, capture_factory=None):
        self.capture_factory = capture_factory
        self.lock = threading.RLock()
        self.streams = {}
        self.rooms = {}
        self.activity = {}
        self.server = None
        self.phone_service = None

    def attach(self, owner, address):
        with self.lock:
            self.rooms.setdefault(owner, set()).add(address)
            stream = self.streams.get(address)
            if stream is None:
                if address.startswith("phone:"):
                    raise ValueError("Phone connection expired. Add your phone again.")
                stream = self.streams[address] = CameraStream(
                    address, self.capture_factory
                )
            return stream

    def connect(self, owner, address):
        with self.lock:
            existing = address in self.rooms.get(owner, set())
            stream = self.attach(owner, address)
        try:
            return stream.snapshot()
        except Exception:
            if not existing:
                self.disconnect(owner, address)
            raise

    def disconnect(self, owner, address):
        with self.lock:
            self.rooms.get(owner, set()).discard(address)
            self.activity.pop(owner, None)
            if not any(address in room for room in self.rooms.values()):
                stream = self.streams.pop(address, None)
                if stream:
                    stream.stop.set()

    def snapshot(self, address):
        with self.lock:
            stream = self.streams.get(address)
        if stream is None:
            raise ValueError("Camera is not connected")
        return stream.snapshot()

    def set_active(self, task, address=None, query=""):
        with self.lock:
            if address is None:
                self.activity.pop(task, None)
            else:
                self.activity[task] = (address, query)

    def room_state(self, owner, previous=None, visible=None):
        previous = previous or {}
        with self.lock:
            if owner not in self.rooms:
                return None
            states = []
            for address in self.rooms[owner]:
                stream = self.streams.get(address)
                if stream is None:
                    continue
                state = stream.state(previous.get(stream.id, 0))
                if visible is not None and stream.id not in visible:
                    state["jpeg"] = None
                active = [
                    query
                    for active_address, query in self.activity.values()
                    if active_address == address
                ]
                state.update(searching=bool(active), query=active[0] if active else "")
                states.append(state)
            return states

    def descriptors(self, owner, feeds):
        with self.lock:
            allowed = self.rooms.get(owner, set())
            return [
                {
                    "id": self.streams[feed["address"]].id,
                    "name": feed["name"],
                    "location": feed.get("location", ""),
                    "paused": feed.get("paused", False),
                }
                for feed in feeds
                if feed["address"] in allowed and feed["address"] in self.streams
            ]

    def start_server(self):
        with self.lock:
            if self.server:
                return self.server.server_port
            hub = self

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    parsed = urlparse(self.path)
                    params = parse_qs(parsed.query, keep_blank_values=True)
                    if parsed.path != "/frames":
                        self.send_error(404)
                        return
                    try:
                        previous = json.loads(params.get("previous", ["{}"])[0])
                        if not isinstance(previous, dict):
                            raise ValueError("Invalid sequence")
                    except (ValueError, TypeError):
                        self.send_error(400)
                        return
                    visible = params.get("visible", [None])[0]
                    visible = set(visible.split(",")) if visible is not None else None
                    states = hub.room_state(
                        params.get("token", [""])[0], previous, visible
                    )
                    if states is None:
                        self.send_error(403)
                        return
                    payload = json.dumps(states).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    try:
                        self.wfile.write(payload)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                def log_message(self, *args):
                    pass

            self.server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
            self.server.daemon_threads = True
            threading.Thread(target=self.server.serve_forever, daemon=True).start()
            return self.server.server_port

    def close(self):
        if self.phone_service:
            self.phone_service.close()
            self.phone_service = None
        with self.lock:
            streams = list(self.streams.values())
            self.streams.clear()
            self.rooms.clear()
            for stream in streams:
                stream.stop.set()
        for stream in streams:
            stream.thread.join(timeout=6.0)
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None

    def pair_phone(self, owner, host):
        from phone_cameras import PhoneService

        with self.lock:
            if self.phone_service is None:
                self.phone_service = PhoneService(self)
        return self.phone_service.pair(owner, host)


_hub = None
_hub_lock = threading.Lock()


def get_hub():
    global _hub
    with _hub_lock:
        if _hub is None:
            _hub = CameraHub()
            atexit.register(_hub.close)
        return _hub
