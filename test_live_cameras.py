"""Check camera connections, live frames, session privacy, and search activity."""

import base64
from io import BytesIO
import json
import threading
import time
import unittest
from urllib.error import HTTPError
from urllib.request import urlopen

import numpy as np
from PIL import Image
from live_cameras import CameraHub


class FakeCapture:
    def __init__(self):
        self.released = threading.Event()
        self.available = True
        self.number = 0

    def isOpened(self):
        return self.available

    def read(self):
        self.number += 1
        return (
            (True, np.full((480, 800, 3), self.number % 255, dtype=np.uint8))
            if self.available
            else (False, None)
        )

    def release(self):
        self.released.set()


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.captures = []

        def factory(address):
            capture = FakeCapture()
            self.captures.append(capture)
            return capture

        self.hub = CameraHub(factory)
        self.addCleanup(self.hub.close)

    def wait_for(self, predicate, timeout=4):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.02)
        self.fail("Live camera state did not update")

    def test_continuous_frames_and_shared_search_reader(self):
        self.hub.connect("room", "0")
        stream = self.hub.streams["0"]
        first = stream.state()
        self.wait_for(lambda: stream.state()["sequence"] > first["sequence"])
        for _ in range(5):
            self.assertEqual(self.hub.snapshot("0").size, (800, 480))
        self.assertEqual(len(self.captures), 1)
        jpeg = Image.open(BytesIO(base64.b64decode(stream.state()["jpeg"])))
        self.assertLessEqual(jpeg.width, 640)
        self.assertLessEqual(jpeg.height, 360)
        current = stream.state()
        self.assertIsNone(stream.state(current["sequence"])["jpeg"])

    def test_disconnect_releases_only_unreferenced_camera(self):
        self.hub.connect("room-a", "0")
        self.hub.connect("room-b", "0")
        self.assertEqual(len(self.captures), 1)
        self.hub.disconnect("room-a", "0")
        self.assertFalse(self.captures[0].released.is_set())
        self.hub.disconnect("room-b", "0")
        self.assertTrue(self.captures[0].released.wait(2))
        self.assertNotIn("0", self.hub.streams)

    def test_unavailable_stream_reconnects(self):
        self.hub.connect("room", "0")
        stream = self.hub.streams["0"]
        self.captures[0].available = False
        self.wait_for(lambda: not stream.state()["live"])
        with self.assertRaises(ValueError):
            self.hub.snapshot("0")
        self.wait_for(lambda: len(self.captures) == 2 and stream.state()["live"])
        self.assertTrue(self.captures[0].released.is_set())

    def test_search_highlights_and_room_isolation(self):
        self.hub.connect("room-a", "rtsp://user:secret@camera-a")
        self.hub.connect("room-b", "rtsp://camera-b")
        self.hub.set_active("search", "rtsp://user:secret@camera-a", "red car")
        state_a = self.hub.room_state("room-a")
        state_b = self.hub.room_state("room-b")
        self.assertEqual(len(state_a), 1)
        self.assertTrue(state_a[0]["searching"])
        self.assertFalse(state_b[0]["searching"])
        self.assertNotIn("secret", json.dumps(state_a))
        self.hub.set_active("search")
        self.assertFalse(self.hub.room_state("room-a")[0]["searching"])

    def test_live_endpoint_requires_room_token_and_hides_credentials(self):
        self.hub.connect("room-secret-token", "rtsp://user:password@camera")
        port = self.hub.start_server()
        url = f"http://127.0.0.1:{port}/frames"
        with self.assertRaises(HTTPError) as invalid:
            urlopen(url + "?token=invalid", timeout=3)
        self.assertEqual(invalid.exception.code, 403)
        with urlopen(url + "?token=room-secret-token", timeout=3) as response:
            data = response.read().decode()
        states = json.loads(data)
        self.assertEqual(len(states), 1)
        self.assertTrue(states[0]["live"])
        self.assertNotIn("password", data)
        self.assertNotIn("rtsp://", data)
        with urlopen(url + "?token=room-secret-token&visible=", timeout=3) as response:
            self.assertIsNone(json.load(response)[0]["jpeg"])


if __name__ == "__main__":
    unittest.main()
