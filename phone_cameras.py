"""Receive phone camera frames over a private pairing link."""

import base64
import ipaddress
import secrets
import socket
import ssl
import threading
import time
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs, urlparse

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from PIL import Image, UnidentifiedImageError

from live_cameras import CameraStream


def local_address():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.connect(("192.0.2.1", 80))
            return connection.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class PhoneStream(CameraStream):
    def __init__(self, address):
        self.address = address
        self.id = secrets.token_hex(16)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.frame = None
        self.jpeg = None
        self.sequence = 0
        self.updated = 0.0
        self.captured = None
        self.status = "Waiting for phone"
        self.thread = threading.Thread(target=self.stop.wait, daemon=True)
        self.thread.start()

    def receive(self, payload):
        with Image.open(BytesIO(payload)) as image:
            if image.format != "JPEG" or image.width * image.height > 4_000_000:
                raise ValueError("Send a JPEG frame no larger than four megapixels")
            image = image.convert("RGB")
            image.thumbnail((1280, 720))
        preview = image.copy()
        preview.thumbnail((640, 360))
        output = BytesIO()
        preview.save(output, format="JPEG", quality=75)
        with self.lock:
            if self.stop.is_set():
                raise ValueError("Phone disconnected")
            self.frame = image
            self.jpeg = base64.b64encode(output.getvalue()).decode("ascii")
            self.sequence += 1
            self.updated = time.monotonic()
            self.captured = datetime.now().astimezone().isoformat(timespec="seconds")
            self.status = "Live"
            self.ready.set()

    def snapshot(self, timeout=6.0):
        if not self.ready.wait(timeout):
            raise ValueError("The phone has not started its camera yet")
        with self.lock:
            if self.stop.is_set() or time.monotonic() - self.updated > 2.0:
                raise ValueError("Phone camera unavailable")
            return self.frame.copy()


class PhoneService:
    def __init__(self, hub):
        self.hub = hub
        self.lock = threading.RLock()
        self.pairings = {}
        self.server = None
        self.certificates = TemporaryDirectory(prefix="inquisitor-phone-")

    def start(self, host):
        address = ipaddress.ip_address(host)
        with self.lock:
            if self.server:
                return self.server.server_port
            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            name = x509.Name(
                [x509.NameAttribute(NameOID.COMMON_NAME, "Inquisitor phone camera")]
            )
            now = datetime.now(timezone.utc)
            certificate = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(now - timedelta(minutes=5))
                .not_valid_after(now + timedelta(days=30))
                .add_extension(
                    x509.BasicConstraints(ca=True, path_length=0), critical=True
                )
                .add_extension(
                    x509.SubjectAlternativeName(
                        [
                            x509.IPAddress(address),
                            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                            x509.DNSName("localhost"),
                        ]
                    ),
                    critical=False,
                )
                .sign(key, hashes.SHA256())
            )
            folder = Path(self.certificates.name)
            cert_path, key_path = folder / "certificate.pem", folder / "key.pem"
            cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
            key_path.write_bytes(
                key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.PKCS8,
                    serialization.NoEncryption(),
                )
            )
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.minimum_version = ssl.TLSVersion.TLSv1_2
            context.load_cert_chain(cert_path, key_path)
            service = self

            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    if urlparse(self.path).path != "/":
                        self.send_error(404)
                        return
                    payload = Path(__file__).with_name("phone_camera.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Referrer-Policy", "no-referrer")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def do_POST(self):
                    parsed = urlparse(self.path)
                    if parsed.path != "/frame":
                        self.send_error(404)
                        return
                    token = parse_qs(parsed.query).get("token", [""])[0]
                    with service.lock:
                        pairing = service.pairings.get(token)
                    if not pairing:
                        self.send_error(403, "Pairing link expired or disconnected")
                        return
                    address, expires = pairing
                    with service.hub.lock:
                        stream = service.hub.streams.get(address)
                    if (
                        not stream
                        or stream.stop.is_set()
                        or (not stream.ready.is_set() and time.monotonic() > expires)
                    ):
                        self.send_error(403, "Pairing link expired or disconnected")
                        return
                    try:
                        length = int(self.headers.get("Content-Length", "0"))
                        if not 0 < length <= 1_000_000:
                            self.send_error(413)
                            return
                        if (
                            self.headers.get("Content-Type", "").split(";")[0]
                            != "image/jpeg"
                        ):
                            self.send_error(415)
                            return
                        self.connection.settimeout(10)
                        payload = self.rfile.read(length)
                        if len(payload) != length:
                            raise ValueError("Incomplete frame")
                        stream.receive(payload)
                    except (ValueError, OSError, UnidentifiedImageError):
                        self.send_error(400, "Invalid camera frame")
                        return
                    self.send_response(204)
                    self.send_header("Content-Length", "0")
                    self.end_headers()

                def log_message(self, *args):
                    pass

            server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
            server.daemon_threads = True
            server.socket = context.wrap_socket(server.socket, server_side=True)
            self.server = server
            self.host = host
            self.certificate = certificate.public_bytes(serialization.Encoding.PEM)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            return server.server_port

    def pair(self, owner, host):
        port = self.start(host)
        if host != self.host:
            raise ValueError(
                f"The phone server is already using {self.host}. Restart Inquisitor to change its address."
            )
        token = secrets.token_urlsafe(32)
        address = "phone:" + secrets.token_hex(16)
        with self.hub.lock:
            self.hub.streams[address] = PhoneStream(address)
            self.hub.rooms.setdefault(owner, set()).add(address)
        with self.lock:
            self.pairings[token] = (address, time.monotonic() + 600)
        url_host = f"[{host}]" if ":" in host else host
        return address, f"https://{url_host}:{port}/#{token}"

    def close(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.server = None
        self.pairings.clear()
        self.certificates.cleanup()
