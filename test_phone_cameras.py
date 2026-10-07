"""Check phone pairing and frame delivery without a real phone."""

import ssl
import time
import unittest
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

from live_cameras import CameraHub


class PhoneTests(unittest.TestCase):
    def setUp(self):
        self.hub = CameraHub()
        self.addCleanup(self.hub.close)
        self.address, self.link = self.hub.pair_phone("room-a", "127.0.0.1")
        self.service = self.hub.phone_service
        self.token = self.link.split("#")[1]
        self.base = self.link.split("/#")[0]
        self.context = ssl.create_default_context(
            cafile=str(Path(self.service.certificates.name) / "certificate.pem")
        )
        output = BytesIO()
        Image.new("RGB", (800, 450), "red").save(output, format="JPEG")
        self.frame = output.getvalue()

    def upload(self, token=None, payload=None):
        request = Request(
            self.base + "/frame?token=" + (token or self.token),
            data=self.frame if payload is None else payload,
            headers={"Content-Type": "image/jpeg"},
            method="POST",
        )
        return urlopen(request, context=self.context, timeout=3)

    def test_phone_frames_reach_only_the_paired_room(self):
        with urlopen(self.base + "/", context=self.context, timeout=3) as response:
            self.assertIn(b"Connect camera", response.read())
        with self.upload() as response:
            self.assertEqual(response.status, 204)
        state = self.hub.room_state("room-a")[0]
        self.assertTrue(state["live"])
        self.assertIsNotNone(state["captured"])
        self.assertEqual(self.hub.snapshot(self.address).size, (800, 450))
        self.assertIsNone(self.hub.room_state("room-b"))
        self.assertNotIn(self.token, str(state))
        self.hub.set_active("job", self.address, "red car")
        self.assertTrue(self.hub.room_state("room-a")[0]["searching"])

    def test_invalid_expired_and_disconnected_links_cannot_upload(self):
        with self.assertRaises(HTTPError) as error:
            self.upload(token="wrong")
        self.assertEqual(error.exception.code, 403)
        with self.assertRaises(HTTPError) as error:
            self.upload(payload=b"invalid image")
        self.assertEqual(error.exception.code, 400)
        self.service.pairings[self.token] = (self.address, time.monotonic() - 1)
        with self.assertRaises(HTTPError) as error:
            self.upload()
        self.assertEqual(error.exception.code, 403)
        self.service.pairings[self.token] = (self.address, time.monotonic() + 600)
        self.hub.disconnect("room-a", self.address)
        with self.assertRaises(HTTPError) as error:
            self.upload()
        self.assertEqual(error.exception.code, 403)


if __name__ == "__main__":
    unittest.main()
