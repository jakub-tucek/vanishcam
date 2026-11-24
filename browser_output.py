"""Serve processed frames to the VanishCam browser extension over localhost.

Only the loopback interface is used. A request must name the expected Host (blocks DNS
rebinding), come from an allowed meeting origin, and carry the X-VanishCam header, which
forces a CORS preflight that other websites cannot pass. Frames are JPEG-encoded only
while the extension is actively polling.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

DEFAULT_PORT = 47809
ALLOWED_ORIGINS = frozenset(("https://meet.google.com",))
CLIENT_HEADER = "X-VanishCam"
ACTIVE_SECONDS = 2.0
WAIT_SECONDS = 0.5


class BrowserOutput:
    def __init__(self, port: int = DEFAULT_PORT, quality: int = 85):
        self.quality = quality
        self._condition = threading.Condition()
        self._jpeg: bytes | None = None
        self._frame_id = 0
        self._last_request = float("-inf")
        self._server = ThreadingHTTPServer(("127.0.0.1", port), self._handler_class())
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    @property
    def connected(self) -> bool:
        return time.monotonic() - self._last_request < ACTIVE_SECONDS

    def publish(self, frame: np.ndarray) -> None:
        if not self.connected:
            return
        ok, encoded = cv2.imencode(".jpg", frame, (cv2.IMWRITE_JPEG_QUALITY, self.quality))
        if not ok:
            return
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._frame_id += 1
            self._condition.notify_all()

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def _next_frame(self, after: int) -> tuple[int, bytes] | None:
        self._last_request = time.monotonic()
        with self._condition:
            if after > self._frame_id:
                after = 0  # The app restarted while the extension kept its old frame id.
            self._condition.wait_for(lambda: self._frame_id > after and self._jpeg is not None,
                                     timeout=WAIT_SECONDS)
            if self._frame_id > after and self._jpeg is not None:
                return self._frame_id, self._jpeg
        return None

    def _handler_class(self):
        output = self
        expected_hosts = None

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

            def _origin(self) -> str | None:
                nonlocal expected_hosts
                if expected_hosts is None:
                    expected_hosts = {f"127.0.0.1:{output.port}", f"localhost:{output.port}"}
                origin = self.headers.get("Origin")
                if self.headers.get("Host") not in expected_hosts or origin not in ALLOWED_ORIGINS:
                    return None
                return origin

            def _cors(self, origin: str) -> None:
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")

            def do_OPTIONS(self):
                origin = self._origin()
                if origin is None:
                    self.send_error(403)
                    return
                self.send_response(204)
                self._cors(origin)
                self.send_header("Access-Control-Allow-Methods", "GET")
                self.send_header("Access-Control-Allow-Headers", CLIENT_HEADER)
                self.send_header("Access-Control-Allow-Private-Network", "true")
                self.send_header("Access-Control-Max-Age", "600")
                self.end_headers()

            def do_GET(self):
                origin = self._origin()
                url = urlparse(self.path)
                if origin is None or self.headers.get(CLIENT_HEADER) != "1" or url.path != "/frame":
                    self.send_error(403)
                    return
                try:
                    after = int(parse_qs(url.query).get("after", ["0"])[0])
                except ValueError:
                    after = 0
                frame = output._next_frame(after)
                if frame is None:
                    self.send_response(204)
                    self._cors(origin)
                    self.end_headers()
                    return
                frame_id, jpeg = frame
                self.send_response(200)
                self._cors(origin)
                self.send_header("Access-Control-Expose-Headers", "X-Frame-Id")
                self.send_header("X-Frame-Id", str(frame_id))
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(jpeg)

        return Handler
