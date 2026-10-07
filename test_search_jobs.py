"""Check video timing and search jobs without downloading AI models."""

from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import time
import threading
import unittest
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from search_jobs import JobManager, TERMINAL, discover, now, video_frames


class FakeDetector:
    def explore(self, image, query, **options):
        return image.copy(), [
            {"object": query, "confidence": 0.9, "box": [0, 0, 10, 10]}
        ]


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root / "media"
        self.media.mkdir()
        Image.new("RGB", (64, 48), "red").save(self.media / "image.jpg")
        self.video = self.media / "clip.avi"
        writer = cv2.VideoWriter(
            str(self.video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48)
        )
        self.assertTrue(writer.isOpened())
        for i in range(40):
            writer.write(np.full((48, 64, 3), i * 5, dtype=np.uint8))
        writer.release()
        self.manager = JobManager(self.root / "jobs", lambda segment: FakeDetector())

    def config(self, **options):
        return dict(
            {
                "kind": "files",
                "query": "red car",
                "paths": [str(self.media)],
                "interval": 1.0,
                "max_frames": 100,
            },
            **options
        )

    def wait(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            job = next(j for j in self.manager.list() if j["id"] == job_id)
            if job["status"] in TERMINAL:
                return job
            time.sleep(0.02)
        self.fail("Job failed to finish")

    def test_discovery_deduplicates_and_limits(self):
        files = list(discover([str(self.media), str(self.video)]))
        self.assertEqual(len(files), 2)
        self.assertEqual(len(list(discover([str(self.media)], limit=1))), 1)

    def test_video_sampling_and_offsets(self):
        frames = list(
            video_frames(self.video, interval=1.0, offset_start=0.5, offset_end=3.0)
        )
        self.assertEqual([round(f[1], 2) for f in frames], [0.5, 1.5, 2.5])
        self.assertTrue(all(f[2] is None for f in frames))

    def test_historical_window_and_recording_times(self):
        recording = now() - timedelta(days=1)
        frames = list(
            video_frames(
                self.video,
                interval=0.5,
                recording_start=recording,
                window_start=recording + timedelta(seconds=1),
                window_end=recording + timedelta(seconds=2),
            )
        )
        self.assertEqual([f[1] for f in frames], [1.0, 1.5])
        self.assertEqual(frames[0][2], recording + timedelta(seconds=1))
        self.assertEqual(
            list(
                video_frames(
                    self.video,
                    recording_start=recording,
                    window_start=recording + timedelta(hours=1),
                    window_end=recording + timedelta(hours=2),
                )
            ),
            [],
        )

    def test_calendar_search_requires_recording_date(self):
        with self.assertRaisesRegex(ValueError, "Recording start"):
            list(video_frames(self.video, window_start=now()))

    def test_files_results_persist_and_respect_budget(self):
        job_id = self.manager.submit(self.config(max_frames=2))
        job = self.wait(job_id)
        self.assertEqual(job["scanned"], 2)
        self.assertEqual(job["status"], "limit reached")
        self.assertEqual(len(job["results"]), 2)
        for item in job["results"]:
            self.assertTrue((self.manager.root / job_id / item["image_file"]).exists())
        recovered = JobManager(self.manager.root).list()[0]
        self.assertEqual(recovered["results"], job["results"])

    def test_clear_saved_results_and_preserve_other_jobs(self):
        first = self.manager.submit(self.config(max_frames=1))
        self.wait(first)
        second = self.manager.submit(self.config(max_frames=1))
        self.wait(second)
        self.manager.clear(first)
        self.assertFalse((self.manager.root / first).exists())
        self.assertEqual([job["id"] for job in self.manager.list()], [second])
        self.assertTrue((self.media / "image.jpg").exists())
        self.manager.clear()
        self.assertEqual(JobManager(self.manager.root).list(), [])
        self.assertFalse((self.manager.root / second).exists())

    def test_clear_during_inference_does_not_recreate_results(self):
        entered, release = threading.Event(), threading.Event()

        class SlowDetector(FakeDetector):
            def explore(self, image, query, **options):
                entered.set()
                release.wait(5)
                return super().explore(image, query, **options)

        self.manager.engine_factory = lambda segment: SlowDetector()
        job_id = self.manager.submit(self.config(max_frames=1))
        try:
            self.assertTrue(entered.wait(5))
            self.manager.clear(job_id)
            self.assertEqual(self.manager.list(), [])
            self.assertFalse((self.manager.root / job_id).exists())
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while job_id in self.manager.events and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertNotIn(job_id, self.manager.events)
        self.assertFalse((self.manager.root / job_id).exists())
        self.assertEqual(JobManager(self.manager.root).list(), [])

    def test_undo_clear_restores_reports_and_matching_images(self):
        job_id = self.manager.submit(self.config(max_frames=1))
        saved = self.wait(job_id)
        self.manager.clear(job_id, undoable=True)
        self.assertEqual(self.manager.list(), [])
        self.assertEqual(self.manager.undo_clear(), [job_id])
        self.assertEqual(self.manager.list()[0]["results"], saved["results"])
        self.assertTrue(
            (self.manager.root / job_id / saved["results"][0]["image_file"]).exists()
        )
        self.assertEqual(JobManager(self.manager.root).list()[0]["id"], job_id)

    def test_undo_during_inference_keeps_restored_search_stopped(self):
        entered, release = threading.Event(), threading.Event()

        class SlowDetector(FakeDetector):
            def explore(self, image, query, **options):
                entered.set()
                release.wait(5)
                return super().explore(image, query, **options)

        self.manager.engine_factory = lambda segment: SlowDetector()
        job_id = self.manager.submit(self.config(max_frames=1))
        try:
            self.assertTrue(entered.wait(5))
            self.manager.clear(job_id, undoable=True)
            self.manager.undo_clear()
            self.assertEqual(self.manager.list()[0]["status"], "cancelled")
        finally:
            release.set()
        deadline = time.monotonic() + 5
        while job_id in self.manager.events and time.monotonic() < deadline:
            time.sleep(0.02)
        restored = self.manager.list()[0]
        self.assertEqual(restored["status"], "cancelled")
        self.assertEqual(restored["results"], [])
        self.assertEqual(JobManager(self.manager.root).list()[0]["status"], "cancelled")

    def test_per_file_historical_search(self):
        recording = now() - timedelta(days=1)
        config = self.config(
            paths=[str(self.video)],
            recording_times={str(self.video.resolve()): recording.isoformat()},
            window_start=(recording + timedelta(seconds=1)).isoformat(),
            window_end=(recording + timedelta(seconds=3)).isoformat(),
        )
        job = self.wait(self.manager.submit(config))
        self.assertEqual(job["status"], "completed")
        self.assertEqual([r["video_seconds"] for r in job["results"]], [1.0, 2.0])
        self.assertEqual(
            datetime.fromisoformat(job["results"][0]["captured"]),
            recording + timedelta(seconds=1),
        )

    def test_scheduled_cancel_and_secret_redaction(self):
        start = now() + timedelta(seconds=10)
        config = self.config(
            kind="live",
            feeds=[{"name": "Gate", "address": "rtsp://user:secret@camera"}],
            window_start=start.isoformat(),
            window_end=(start + timedelta(seconds=10)).isoformat(),
        )
        job_id = self.manager.submit(config)
        self.manager.cancel(job_id)
        self.assertEqual(self.wait(job_id)["status"], "cancelled")
        report = (self.manager.root / job_id / "job.json").read_text()
        self.assertNotIn("secret", report)
        self.assertNotIn("feeds", self.manager.list()[0])

    def test_live_window(self):
        start = now()
        config = self.config(
            kind="live",
            feeds=[{"name": "Gate", "address": "0", "location": "Entrance"}],
            window_start=start.isoformat(),
            window_end=(start + timedelta(seconds=0.4)).isoformat(),
            interval=0.1,
        )
        with patch("search_jobs.snapshot", return_value=Image.new("RGB", (64, 48))):
            job = self.wait(self.manager.submit(config))
        self.assertEqual(job["status"], "completed")
        self.assertGreater(job["scanned"], 0)
        self.assertTrue(all(r["location"] == "Entrance" for r in job["results"]))

    def test_invalid_video_surfaces_error(self):
        broken = self.media / "broken.mp4"
        broken.write_bytes(b"not video")
        job = self.wait(self.manager.submit(self.config(paths=[str(broken)])))
        self.assertEqual(job["status"], "failed")
        self.assertTrue(job["errors"])

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.manager.submit(self.config(interval=0))
        with self.assertRaises(ValueError):
            self.manager.submit(
                self.config(
                    window_start=now().isoformat(),
                    window_end=(now() - timedelta(days=1)).isoformat(),
                )
            )

    def test_restart_marks_incomplete_job_interrupted(self):
        folder = self.manager.root / "test"
        folder.mkdir()
        (folder / "job.json").write_text(
            json.dumps({"id": "test", "status": "running"})
        )
        recovered = JobManager(self.manager.root).list()[0]
        self.assertEqual(recovered["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
