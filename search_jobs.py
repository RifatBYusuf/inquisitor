"""Run searches in the background without touching Streamlit's session state."""

import json
import math
import shutil
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import cv2
from PIL import Image, ImageOps

IMAGES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEOS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm", ".mts", ".m2ts"}
TERMINAL = {"completed", "cancelled", "failed", "interrupted", "limit reached"}


def now():
    return datetime.now().astimezone()


def timestamp(value):
    return datetime.fromisoformat(value).astimezone() if value else None


def media_extensions(media_type="both"):
    if media_type == "images":
        return IMAGES
    if media_type == "videos":
        return VIDEOS
    if media_type == "both":
        return IMAGES | VIDEOS
    raise ValueError("Choose images, videos, or both")


def discover(
    paths, recursive=True, limit=200, cancelled=lambda: False, media_type="both"
):
    """Collect files from the selected paths without including the same file twice."""
    seen = set()
    count = 0
    extensions = media_extensions(media_type)
    for raw in paths:
        root = Path(raw.strip().strip('"')).expanduser()
        if not root.exists():
            raise ValueError(f"Path does not exist: {root}")
        candidates = (
            [root]
            if root.is_file()
            else (root.rglob("*") if recursive else root.iterdir())
        )
        for path in candidates:
            if cancelled():
                return
            if not path.is_file() or path.suffix.lower() not in extensions:
                continue
            resolved = str(path.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path.resolve()
            count += 1
            if count >= limit:
                return


def open_camera(address):
    if str(address).strip().isdigit():
        capture = cv2.VideoCapture(int(address))
    else:
        capture = cv2.VideoCapture(
            address,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                5000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                5000,
            ],
        )
    return capture


def snapshot(address):
    capture = open_camera(address)
    try:
        ok, frame = capture.read() if capture.isOpened() else (False, None)
        if not ok:
            raise ValueError("Camera unavailable or no frame received")
        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()


def video_frames(
    path,
    interval=2.0,
    offset_start=0.0,
    offset_end=None,
    recording_start=None,
    window_start=None,
    window_end=None,
    cancelled=lambda: False,
):
    """Sample video frames and include their video position and known recording time."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("Unable to open video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or not math.isfinite(count) or count <= 0:
            raise ValueError("Video has no usable duration/frame-rate metadata")
        duration = count / fps
        begin = max(0.0, offset_start)
        end = min(duration, offset_end if offset_end is not None else duration)
        if window_start or window_end:
            if recording_start is None:
                raise ValueError(
                    "Recording start time required for calendar-time video search"
                )
            if window_start:
                begin = max(begin, (window_start - recording_start).total_seconds())
            if window_end:
                end = min(end, (window_end - recording_start).total_seconds())
        if interval <= 0 or end <= begin:
            return
        sample = 0
        last_index = -1
        while not cancelled():
            offset = begin + sample * interval
            if offset >= end:
                break
            frame_index = min(int(math.ceil(offset * fps)), int(count) - 1)
            sample += 1
            if frame_index == last_index:
                continue
            actual = frame_index / fps
            if actual >= end:
                break
            last_index = frame_index
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
                raise ValueError("Video codec does not support seeking")
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Unable to decode frame at {actual:.2f}s")
            yield Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), actual, (
                recording_start + timedelta(seconds=actual) if recording_start else None
            )
    finally:
        capture.release()


def image_time(path):
    with Image.open(path) as image:
        value = image.getexif().get(36867) or image.getexif().get(306)
        if value:
            try:
                return (
                    datetime.strptime(value, "%Y:%m:%d %H:%M:%S").astimezone(),
                    "EXIF capture time (host timezone)",
                )
            except (TypeError, ValueError):
                pass
    return (
        datetime.fromtimestamp(path.stat().st_mtime).astimezone(),
        "File modification time",
    )


class JobManager:
    def __init__(self, root, engine_factory=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.jobs = {}
        self.events = {}
        self.removed = set()
        self.undo_batches = []
        self.engine_factory = engine_factory
        self.camera_reader = None
        self.camera_activity = None
        for file in self.root.glob("*/job.json"):
            try:
                job = json.loads(file.read_text(encoding="utf-8"))
                self.jobs[job["id"]] = job
                if job["status"] not in TERMINAL:
                    job.update(
                        status="interrupted",
                        message="Application stopped; restart this search manually.",
                    )
                    self._save(job)
            except (ValueError, OSError, KeyError):
                continue

    def _save(self, job):
        if self.jobs.get(job["id"]) is not job:
            return
        # Camera addresses can include passwords, so keep them out of saved reports.
        safe = {key: value for key, value in job.items() if key != "feeds"}
        folder = self.root / job["id"]
        folder.mkdir(exist_ok=True)
        temp = folder / "job.tmp"
        temp.write_text(json.dumps(safe, indent=2), encoding="utf-8")
        temp.replace(folder / "job.json")

    def _update(self, job, **values):
        with self.lock:
            job.update(values)
            self._save(job)

    def list(self):
        with self.lock:
            return [
                json.loads(json.dumps({k: v for k, v in j.items() if k != "feeds"}))
                for j in reversed(list(self.jobs.values()))
            ]

    def submit(self, config):
        if config.get("kind") == "files":
            media_extensions(config.get("media_type", "both"))
        if not config.get("query", "").strip():
            raise ValueError("Enter an object description")
        if (
            not math.isfinite(config.get("interval", 2.0))
            or config.get("interval", 2.0) <= 0
        ):
            raise ValueError("Sampling interval must be positive")
        start, end = timestamp(config.get("window_start")), timestamp(
            config.get("window_end")
        )
        if start and end and end <= start:
            raise ValueError("End time must be after start time")
        if config["kind"] == "live":
            if not config.get("feeds") or not start or not end or end <= now():
                raise ValueError("Select camera sources and a future end time")
        elif not config.get("paths"):
            raise ValueError("Enter at least one file or folder path")
        if config.get("offset_end") is not None and config["offset_end"] <= config.get(
            "offset_start", 0
        ):
            raise ValueError("Video end offset must be after start offset")
        for value in config.get("recording_times", {}).values():
            timestamp(value)
        if config.get("recording_start"):
            timestamp(config["recording_start"])
        with self.lock:
            if sum(j["status"] not in TERMINAL for j in self.jobs.values()) >= 4:
                raise ValueError(
                    "Stop or finish an active job before starting another (maximum 4)"
                )
            job = dict(
                config,
                id=uuid.uuid4().hex,
                created=now().isoformat(),
                status="queued",
                scanned=0,
                results=[],
                errors=[],
                message="Waiting to start",
            )
            self.jobs[job["id"]] = job
            self.events[job["id"]] = threading.Event()
            self._save(job)
            threading.Thread(target=self._run, args=(job,), daemon=True).start()
            return job["id"]

    def cancel(self, job_id):
        with self.lock:
            if job_id in self.events and self.jobs[job_id]["status"] not in TERMINAL:
                self.events[job_id].set()
                self._update(
                    self.jobs[job_id], message="Stopping after the current operation"
                )

    def clear(self, job_id=None, undoable=False):
        """Stop the selected jobs and remove their reports and saved matches."""
        with self.lock:
            ids = [job_id] if job_id is not None else list(self.jobs)
            batch = []
            if undoable:
                self.undo_batches.append(batch)
            for selected in ids:
                if selected not in self.jobs:
                    continue
                folder = self.root / selected
                if folder.resolve().parent != self.root.resolve():
                    raise ValueError("Job folder is outside the search data directory")
                event = self.events.get(selected)
                if event is not None:
                    event.set()
                if folder.exists():
                    if undoable:
                        archive = self.root / ".cleared" / uuid.uuid4().hex
                        if archive.resolve().parent.parent != self.root.resolve():
                            raise ValueError(
                                "Archive folder is outside the search data directory"
                            )
                        archive.parent.mkdir(exist_ok=True)
                        folder.rename(archive)
                        batch.append((selected, archive))
                    else:
                        shutil.rmtree(folder)
                if event is not None:
                    self.removed.add(selected)
                del self.jobs[selected]

    def undo_clear(self):
        """Bring back saved results without restarting cancelled jobs."""
        with self.lock:
            while self.undo_batches and not self.undo_batches[-1]:
                self.undo_batches.pop()
            if not self.undo_batches:
                return []
            batch = self.undo_batches[-1]
            restored = []
            while batch:
                selected, archive = batch[-1]
                folder = self.root / selected
                if (
                    folder.resolve().parent != self.root.resolve()
                    or archive.resolve().parent != (self.root / ".cleared").resolve()
                ):
                    raise ValueError("Job folder is outside the search data directory")
                job = json.loads((archive / "job.json").read_text(encoding="utf-8"))
                if job["id"] != selected or selected in self.jobs:
                    raise ValueError("Cannot restore this job")
                if job["status"] not in TERMINAL:
                    job.update(
                        status="cancelled",
                        message="Stopped when cleared. Start a new search to run again.",
                    )
                archive.rename(folder)
                self.jobs[selected] = job
                self._save(job)
                batch.pop()
                restored.append(selected)
            self.undo_batches.pop()
            return restored

    def _error(self, job, source, error):
        with self.lock:
            job["errors"].append({"source": str(source), "error": str(error)})
            # Keep repeated camera errors from filling up the report.
            job["errors"] = job["errors"][-100:]
            self._save(job)

    def _run(self, job):
        stop = self.events[job["id"]]
        try:
            if job["kind"] == "live":
                start = timestamp(job["window_start"])
                self._update(
                    job, status="scheduled", message=f"Starts at {start.isoformat()}"
                )
                while now() < start:
                    if stop.wait(min(1.0, (start - now()).total_seconds())):
                        break
                if now() >= timestamp(job["window_end"]):
                    self._update(
                        job, status="completed", message="Monitoring window has ended"
                    )
                    return
            if stop.is_set():
                self._update(job, status="cancelled", message="Stopped by user")
                return
            self._update(job, status="loading", message="Loading detection engine")
            if self.engine_factory:
                finder = self.engine_factory(job.get("segment", False))
            else:
                import detector

                detector.load_models(segment=job.get("segment", False))
                finder = detector
            self._update(job, status="running", message="Searching")
            if job["kind"] == "live":
                self._live(job, finder, stop)
            else:
                self._files(job, finder, stop)
            status = (
                "cancelled"
                if stop.is_set()
                else (
                    "limit reached"
                    if job["scanned"] >= job.get("max_frames", 10000)
                    else "completed"
                )
            )
            self._update(
                job, status=status, message=f"{job['scanned']} frames/images analyzed"
            )
        except Exception as error:
            self._update(job, status="failed", message=str(error))
        finally:
            with self.lock:
                self.events.pop(job["id"], None)
                self.removed.discard(job["id"])

    def _analyze(
        self,
        job,
        finder,
        stop,
        image,
        source,
        captured=None,
        offset=None,
        time_basis="",
        location="",
    ):
        if stop.is_set() or job["scanned"] >= job.get("max_frames", 10000):
            return
        if job["kind"] == "live" and now() >= timestamp(job["window_end"]):
            return
        annotated, matches = finder.explore(
            image,
            job["query"],
            box_threshold=job.get("confidence", 0.25),
            tiling=job.get("tiling", False),
            segment=job.get("segment", False),
        )
        with self.lock:
            if self.jobs.get(job["id"]) is not job:
                return
            job["scanned"] += 1
            if matches:
                name = f"match-{job['scanned']:06d}.jpg"
                annotated.thumbnail((1400, 1000))
                annotated.convert("RGB").save(self.root / job["id"] / name, quality=90)
                job["results"].append(
                    {
                        "source": str(source),
                        "location": location,
                        "captured": captured.isoformat() if captured else None,
                        "video_seconds": offset,
                        "time_basis": time_basis,
                        "matches": matches,
                        "score": max(m["confidence"] for m in matches),
                        "image_file": name,
                    }
                )
            job["message"] = (
                f"{job['scanned']} analyzed; {len(job['results'])} matching frames/images"
            )
            self._save(job)

    def _files(self, job, finder, stop):
        start, end = timestamp(job.get("window_start")), timestamp(
            job.get("window_end")
        )
        files_seen = 0
        for path in discover(
            job["paths"],
            job.get("recursive", True),
            job.get("max_files", 200),
            stop.is_set,
            media_type=job.get("media_type", "both"),
        ):
            if stop.is_set() or job["scanned"] >= job.get("max_frames", 10000):
                break
            files_seen += 1
            try:
                if path.suffix.lower() in VIDEOS:
                    mapping = job.get("recording_times", {})
                    recording = timestamp(
                        mapping.get(str(path), job.get("recording_start"))
                    )
                    for image, offset, captured in video_frames(
                        path,
                        job["interval"],
                        job.get("offset_start", 0.0),
                        job.get("offset_end"),
                        recording,
                        start,
                        end,
                        stop.is_set,
                    ):
                        if stop.is_set() or job["scanned"] >= job.get(
                            "max_frames", 10000
                        ):
                            break
                        self._analyze(
                            job,
                            finder,
                            stop,
                            image,
                            path,
                            captured,
                            offset,
                            (
                                "Recording start + video offset"
                                if captured
                                else "Video offset"
                            ),
                        )
                else:
                    captured, basis = image_time(path)
                    if (start and captured < start) or (end and captured >= end):
                        continue
                    with Image.open(path) as image:
                        rgb = ImageOps.exif_transpose(image).convert("RGB")
                    self._analyze(
                        job, finder, stop, rgb, path, captured, time_basis=basis
                    )
            except Exception as error:
                self._error(job, path, error)
        if not files_seen and not stop.is_set():
            selected_type = {
                "images": "images",
                "videos": "videos",
                "both": "images or videos",
            }[job.get("media_type", "both")]
            raise ValueError(f"No supported {selected_type} found")
        if job["scanned"] == 0 and job["errors"] and not stop.is_set():
            raise ValueError("No media could be analyzed; review source errors")

    def _live(self, job, finder, stop):
        end = timestamp(job["window_end"])
        while (
            not stop.is_set()
            and now() < end
            and job["scanned"] < job.get("max_frames", 10000)
        ):
            cycle = now()
            for feed in job["feeds"]:
                if (
                    stop.is_set()
                    or now() >= end
                    or job["scanned"] >= job.get("max_frames", 10000)
                ):
                    break
                try:
                    if self.camera_activity:
                        self.camera_activity(job["id"], feed["address"], job["query"])
                    self._update(job, message=f"Searching camera: {feed['name']}")
                    image = (self.camera_reader or snapshot)(feed["address"])
                    self._analyze(
                        job,
                        finder,
                        stop,
                        image,
                        feed["name"],
                        now(),
                        time_basis="Live capture time",
                        location=feed.get("location", ""),
                    )
                except Exception:
                    self._error(
                        job, feed["name"], "Camera unavailable or analysis failed"
                    )
                finally:
                    if self.camera_activity:
                        self.camera_activity(job["id"])
            delay = min(
                job["interval"] - (now() - cycle).total_seconds(),
                (end - now()).total_seconds(),
            )
            if delay > 0:
                stop.wait(delay)
