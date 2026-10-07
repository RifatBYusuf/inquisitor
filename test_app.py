"""Check the web app's workflows using a fake detector."""

from pathlib import Path
from tempfile import TemporaryDirectory
import time
import json
import re
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

from search_jobs import JobManager, TERMINAL
from live_cameras import CameraHub
from test_search_jobs import FakeDetector


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manager = JobManager(self.root / "jobs", lambda segment: FakeDetector())
        self.addCleanup(self.manager.clear)
        self.capture = MagicMock()
        self.capture.isOpened.return_value = True
        self.capture.read.return_value = (True, np.zeros((48, 64, 3), dtype=np.uint8))
        self.hub = CameraHub(lambda address: self.capture)
        self.addCleanup(self.hub.close)
        hub_patcher = patch("live_cameras.get_hub", return_value=self.hub)
        hub_patcher.start()
        self.addCleanup(hub_patcher.stop)
        st.cache_resource.clear()
        patcher = patch("search_jobs.JobManager", return_value=self.manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(st.cache_resource.clear)
        self.app = AppTest.from_file(
            str(Path(__file__).with_name("app.py")), default_timeout=15
        ).run()

    def click(self, label):
        next(b for b in self.app.button if b.label == label).click().run()
        self.assertFalse(self.app.exception, str(self.app.exception))

    def test_workspaces_render(self):
        self.assertFalse(self.app.exception)
        self.assertFalse(
            any(
                item.value == "Monitoring and media search jobs"
                for item in self.app.subheader
            )
        )
        for label in [
            "Connect cameras / webcam",
            "Search history",
            "Search files / folders",
        ]:
            self.click(label)
            self.assertEqual(
                any(
                    item.value == "Monitoring and media search jobs"
                    for item in self.app.subheader
                ),
                label == "Search history",
            )

    def test_search_requires_source_and_description(self):
        button = next(
            item for item in self.app.button if item.label == "Search images and videos"
        )
        self.assertTrue(button.disabled)
        self.app.text_input(key="background_query").set_value("red car").run()
        self.assertTrue(
            next(
                item
                for item in self.app.button
                if item.label == "Search images and videos"
            ).disabled
        )
        self.assertEqual(
            len(
                [item for item in self.app.subheader if item.value == "Search results"]
            ),
            1,
        )
        self.assertTrue(
            any(item.label == "Advanced settings" for item in self.app.expander)
        )

    def test_desktop_picker_selection_and_cancellation(self):
        folder = self.root / "media"
        folder.mkdir()
        image = folder / "photo.jpg"
        image.touch()
        with patch("media_picker.choose_media", return_value=[str(image), str(image)]) as picker:
            self.click("Choose files")
            self.assertEqual(picker.call_args.args[0], "files")
            self.assertIn(".jpg", picker.call_args.args[1])
        self.assertEqual(self.app.session_state["selected_media_paths"], [str(image.resolve())])
        with patch("media_picker.choose_media", return_value=[str(folder)]):
            self.click("Choose folder / drive")
        expected = [str(image.resolve()), str(folder.resolve())]
        self.assertEqual(self.app.session_state["selected_media_paths"], expected)
        with patch("media_picker.choose_media", return_value=[]):
            self.click("Choose folder / drive")
        self.assertEqual(self.app.session_state["selected_media_paths"], expected)

    def test_desktop_picker_failure_shows_message(self):
        self.app.session_state["browse_directory"] = str(self.root)
        with patch("media_picker.choose_media", side_effect=RuntimeError("No desktop")):
            self.click("Choose files")
        self.assertTrue(self.app.warning)
        self.assertFalse(any(item.label == "Browse inside app" for item in self.app.button))
        self.assertFalse(any(item.label == "Folders" for item in self.app.selectbox))

    def test_camera_connection_preview_and_unavailable_feedback(self):
        capture = self.capture
        self.click("Connect cameras / webcam")
        next(
            item for item in self.app.text_input if item.label == "Source name"
        ).set_value("Front gate")
        next(
            item
            for item in self.app.text_input
            if item.label == "Feed address or webcam index"
        ).set_value("0")
        with patch("cv2.VideoCapture", return_value=capture):
            self.click("Connect and test source")
        self.assertEqual(len(self.app.session_state["feeds"]), 1)
        self.assertTrue(
            any(
                "Connected" == item.value
                for item in self.app.markdown
            )
        )
        self.assertIn("0", self.app.session_state["camera_previews"])
        self.click("Pause camera")
        self.assertTrue(self.app.session_state["feeds"][0]["paused"])
        config = json.loads(re.search(r"const config\s*=\s*(.*);", self.app.get("iframe")[0].proto.srcdoc).group(1))
        self.assertTrue(config["feeds"][0]["paused"])
        self.click("Resume camera")
        self.assertFalse(self.app.session_state["feeds"][0]["paused"])
        capture.read.return_value = (False, None)
        deadline = time.monotonic() + 2
        while self.hub.streams["0"].state()["live"] and time.monotonic() < deadline:
            time.sleep(0.01)
        with patch("cv2.VideoCapture", return_value=capture):
            self.click("Refresh preview")
        self.assertTrue(any("Unavailable" in item.value for item in self.app.markdown))
        self.assertNotIn("0", self.app.session_state["camera_previews"])
        self.click("Disconnect")
        self.assertEqual(self.app.session_state["feeds"], [])

    def test_live_control_room_layouts_and_camera_search_highlights(self):
        self.click("Connect cameras / webcam")
        for index in range(2):
            next(
                item for item in self.app.text_input if item.label == "Source name"
            ).set_value(f"Camera {index + 1}")
            next(
                item
                for item in self.app.text_input
                if item.label == "Feed address or webcam index"
            ).set_value(str(index))
            self.click("Connect and test source")
            self.assertFalse(any(item.label == "Control room layout" for item in self.app.selectbox))
            html = self.app.get("iframe")[0].proto.srcdoc
            config = json.loads(re.search(r"const config\s*=\s*(.*);", html).group(1))
            self.assertEqual(config["side"], 1 if index == 0 else 2)
            self.assertEqual(len(config["feeds"]), index + 1)
            self.assertNotIn("address", config["feeds"][0])
        self.app.text_input(key="search_query").set_value("red car").run()
        searched = []
        owner = self.app.session_state["camera_owner"]

        def explore(image, query, **options):
            active = [item for item in self.hub.room_state(owner) if item["searching"]]
            self.assertEqual(len(active), 1)
            searched.append(active[0]["id"])
            stream = self.hub.streams[str(len(searched) - 1)]
            sequence = stream.state()["sequence"]
            time.sleep(0.2)
            self.assertGreater(stream.state()["sequence"], sequence)
            return image.copy(), [
                {"object": query, "confidence": 0.9, "box": [0, 0, 10, 10]}
            ]

        finder = SimpleNamespace(load_models=lambda **options: None, explore=explore)
        with patch.dict(sys.modules, {"detector": finder}):
            self.click("Search connected sources")
        self.assertEqual(searched, [self.hub.streams[str(i)].id for i in range(2)])
        self.assertFalse(any(item["searching"] for item in self.hub.room_state(owner)))
        self.assertEqual(self.app.session_state["scanned"], 2)

    def test_monitoring_inside_camera_workspace(self):
        self.assertFalse(
            any(item.label == "Scheduled monitoring" for item in self.app.button)
        )
        self.app.session_state["feeds"] = [
            {"name": "Test camera", "location": "", "address": "0"}
        ]
        self.click("Connect cameras / webcam")
        self.app.toggle(key="monitor_enabled").set_value(True).run()
        self.assertFalse(self.app.exception, str(self.app.exception))
        self.app.text_input(key="background_query").set_value("red car").run()
        self.click("Start monitoring job")
        jobs = self.manager.list()
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["kind"], "live")
        self.manager.cancel(jobs[0]["id"])

    def test_media_search_submission_and_saved_results(self):
        image = self.root / "car.jpg"
        Image.new("RGB", (64, 48), "red").save(image)
        self.app.session_state["selected_media_paths"] = [str(image)]
        self.app.run()
        self.app.text_input(key="background_query").set_value("red car").run()
        self.click("Search images and videos")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            jobs = self.manager.list()
            if jobs and jobs[0]["status"] in TERMINAL:
                break
            time.sleep(0.02)
        self.app.run()
        self.assertFalse(self.app.exception, str(self.app.exception))
        self.assertEqual(self.manager.list()[0]["scanned"], 1)
        self.assertEqual(len(self.manager.list()[0]["results"]), 1)
        self.assertTrue(
            any("Completed" in element.value for element in self.app.markdown)
        )
        self.assertFalse(any(item.label == "View match" for item in self.app.button))
        self.assertFalse(any("detection confidence" in item.value.lower() for item in self.app.caption))
        self.assertFalse(self.app.exception)
        self.app.run()
        self.assertFalse(
            any(item.label == "Clear all jobs and results" for item in self.app.button)
        )
        self.click("Search history")
        self.assertTrue(
            any(
                item.value == "Monitoring and media search jobs"
                for item in self.app.subheader
            )
        )
        self.click("Clear all jobs and results")
        self.assertEqual(self.manager.list(), [])
        self.click("Undo clear")
        self.assertEqual(len(self.manager.list()), 1)
        self.assertEqual(len(self.manager.list()[0]["results"]), 1)
        self.assertTrue(
            any(
                item.value == "Monitoring and media search jobs"
                for item in self.app.subheader
            )
        )
        self.click("Search files / folders")
        self.assertFalse(
            any(
                item.value == "Monitoring and media search jobs"
                for item in self.app.subheader
            )
        )
        self.assertTrue(any("Completed" in item.value for item in self.app.markdown))


if __name__ == "__main__":
    unittest.main()
