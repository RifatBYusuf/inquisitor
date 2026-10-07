"""Search images and camera feeds with Inquisitor's local models."""

import json
import math
import re
import uuid
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageOps

from live_cameras import get_hub
from media_picker import choose_media
from search_jobs import IMAGES, TERMINAL, VIDEOS, JobManager, media_extensions

st.set_page_config(

    page_title="Inquisitor | Visual Intelligence",
    page_icon=":material/visibility:",
    layout="wide",
)

styles = Path(__file__).with_name("app.css").read_text(encoding="utf-8")
st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)

for key, default in {

    "mode": "Search files / folders",
    "feeds": [],
    "results": [],
    "last_query": "",
    "scanned": 0,
    "errors": [],
    "searched": False,

}.items():
    
    if key not in st.session_state:

        st.session_state[key] = default


if st.session_state.mode in ("Search folders", "Search recorded footage", "Analyze image"):

    st.session_state.mode = "Search files / folders"

if st.session_state.mode == "Scheduled monitoring":

    st.session_state.mode = "Connect cameras / webcam"
    st.session_state.monitor_enabled = True


@st.cache_resource(show_spinner=False)

def engine():
    import detector

    detector.load_models()
    return detector


@st.cache_resource

def job_manager():
    manager = JobManager(Path(__file__).parent / "search_data")
    manager.camera_reader = get_hub().snapshot
    manager.camera_activity = get_hub().set_active
    return manager


def time_window(prefix, start_default=None, end_default=None):

    start_default = start_default or datetime.now()
    end_default = end_default or start_default + timedelta(hours=1)
    a, b = st.columns(2)

    day_start = a.date_input("Start date", start_default.date(), key=prefix + "_sd")
    clock_start = a.time_input(
        "Start time", start_default.time().replace(microsecond=0), key=prefix + "_st"
    )

    day_end = b.date_input("End date", end_default.date(), key=prefix + "_ed")
    clock_end = b.time_input(
        "End time", end_default.time().replace(microsecond=0), key=prefix + "_et"
    )

    st.caption(
        "Dates and times use this computer's local timezone. The end time is exclusive."
    )
    return (
        datetime.combine(day_start, clock_start).astimezone().isoformat(),
        datetime.combine(day_end, clock_end).astimezone().isoformat(),
    )


def navigate(mode):
    st.session_state.mode = mode


def read_picture(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def snapshot(source):
    return get_hub().snapshot(source)


def control_room():

    st.subheader("Camera control room")
    hub = get_hub()

    feeds = hub.descriptors(st.session_state.camera_owner, st.session_state.feeds)

    if not feeds:
        st.info("Connect a webcam or CCTV source to start live playback.")
        return

    side = min(9, math.ceil(math.sqrt(len(feeds))))
    
    config = {
        "side": side,
        "port": hub.start_server(),
        "token": st.session_state.camera_owner,
        "feeds": feeds,
    }

    config_json = json.dumps(config).replace("<", "\u003c")
    html = (
        Path(__file__)
        .with_name("control_room.html")
        .read_text(encoding="utf-8")
        .replace("__CONFIG__", config_json)
    )

    rows = math.ceil(min(len(feeds), side * side) / side)
    panel_height = min(800, 100 + rows * (200 if side <= 2 else 150))
    if hasattr(st, "iframe"):
        st.iframe(html, height=panel_height)
    else:
        components.html(html, height=panel_height, scrolling=True)


def description(query):
    query = re.sub(
        r"^(?:please\s+)?(?:find|show|search for)\s+(?:me\s+)?(?:a\s+)?(?:photo|picture|image)s?\s+(?:with|of|containing)\s+",
        "",
        query.strip(),
        flags=re.I,
    )

    return re.sub(r"\s+in (?:it|them)[.!]?$", "", query, flags=re.I).strip()


with st.sidebar:

    st.markdown(
        '<div class="brand">INQUISITOR<small>VISUAL INTELLIGENCE</small></div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.caption("WORKSPACE")
    for label in [
        "Search files / folders",
        "Connect cameras / webcam",
        "Search history",
    ]:
        
        st.button(
            label,
            on_click=navigate,
            args=(label,),
            use_container_width=True,
            type="primary" if st.session_state.mode == label else "secondary",
        )

    st.divider()

    with st.expander("Advanced detection settings"):
        confidence = st.slider("Detection confidence", 0.05, 0.90, 0.55, 0.05)
        detailed = st.checkbox("Search small objects in tiles", value=False)
        segment = st.checkbox("Generate segmentation masks", value=False)
    st.divider()
    st.caption(f"{len(st.session_state.feeds)} connected source(s)")
    st.caption("Folder paths and GPU compute use this host computer.")

if "camera_owner" not in st.session_state:
    st.session_state.camera_owner = uuid.uuid4().hex

mode = st.session_state.mode
st.markdown(
    '<div class="hero"><div class="orb"></div><div class="eyebrow">INQUISITOR / VISUAL SEARCH CONSOLE</div></div>',
    unsafe_allow_html=True,
)

stats = st.columns(3)
stats[0].metric("Connected sources", len(st.session_state.feeds))
stats[1].metric(
    "Selected files / folders", len(st.session_state.get("selected_media_paths", []))
)

stats[2].metric("Workspace", mode)
st.divider()

camera_wall = st.container()
search_panel, results_panel = st.columns([1, 1.4], gap="large")

with search_panel:

    sources = []
    job_options = None
    manifest_text = ""

    with st.container(border=True):

        st.subheader(mode)
        if mode == "Search files / folders":
            search_type = st.radio(
                "Search type",
                ["Images only", "Videos only", "Images and videos"],
                horizontal=True,
                key="media_search_type",
            )

            media_type = {
                "Images only": "images",
                "Videos only": "videos",
                "Images and videos": "both",
            }[search_type]

            include_videos = media_type != "images"

            files_button, folder_button = st.columns(2)
            choose_files = files_button.button("Choose files", use_container_width=True)
            choose_folder = folder_button.button("Choose folder / drive", use_container_width=True)
            if choose_files or choose_folder:
                try:
                    chosen = choose_media(
                        "files" if choose_files else "folder",
                        media_extensions(media_type),
                        st.session_state.get("browse_directory"),
                    )
                    selected = st.session_state.setdefault("selected_media_paths", [])
                    for entry in chosen:
                        resolved = str(Path(entry).resolve())
                        if resolved not in selected:
                            selected.append(resolved)
                    if chosen:
                        last = Path(chosen[-1]).resolve()
                        st.session_state.browse_directory = str(last.parent if choose_files else last)
                        st.rerun()
                except (RuntimeError, OSError, ValueError):
                    st.warning("The file picker could not open. Run Inquisitor on a computer with a desktop and try again.")

            paths = st.session_state.get("selected_media_paths", [])

            for index, selected_path in enumerate(paths):

                name_column, remove_column = st.columns([5, 1])
                name_column.write(Path(selected_path).name or selected_path)
                name_column.caption(selected_path)
                if remove_column.button("Remove", key=f"remove_media_{index}"):
                    st.session_state.selected_media_paths.pop(index)
                    st.rerun()

            with st.expander("Advanced settings"):

                a, b = st.columns(2)
                recursive = a.checkbox("Include subfolders", value=True)
                limit = b.number_input(
                    "Maximum media files",
                    min_value=1,
                    max_value=10000,
                    value=200,
                    step=50,
                )

                job_options = {

                    "kind": "files",
                    "media_type": media_type,
                    "paths": list(paths),
                    "recursive": recursive,
                    "max_files": int(limit),
                }

                a, b = st.columns(2)

                if include_videos:
                    job_options["interval"] = a.number_input(
                        "Video sampling interval (seconds)",
                        min_value=0.1,
                        max_value=3600.0,
                        value=2.0,
                        step=0.5,
                    )

                job_options["max_frames"] = int(
                    b.number_input(
                        "Maximum images / frames analyzed",
                        min_value=1,
                        max_value=100000,
                        value=10000,
                        step=100,
                    )
                )

                if include_videos:

                    a, b = st.columns(2)
                    job_options["offset_start"] = a.number_input(
                        "Video start offset (seconds)", min_value=0.0, value=0.0
                    )

                    end_offset = b.number_input(
                        "Video end offset (seconds; 0 = end of file)",
                        min_value=0.0,
                        value=0.0,
                    )

                    job_options["offset_end"] = end_offset or None
                if st.checkbox(

                    "Restrict search to a calendar timeframe", key="archive_window"
                ):
                    job_options["window_start"], job_options["window_end"] = (
                        time_window(
                            "archive",
                            datetime.now() - timedelta(days=1),
                            datetime.now(),
                        )
                    )

                if include_videos:

                    st.caption(
                        "Video files do not reliably contain their recording date. Provide the actual time at offset 0. For multiple clips, supply a JSON mapping of full file paths to ISO dates/times. Images use EXIF capture time, falling back to file modification time."
                    )
                    
                    if st.checkbox(
                        "Set a recording start time for the selected video(s)"
                    ):
                        d = st.date_input("Recording date")
                        t = st.time_input("Recording time", datetime.min.time())
                        job_options["recording_start"] = (
                            datetime.combine(d, t).astimezone().isoformat()
                        )
                    manifest_text = st.text_area(
                        "Per-file recording start times (JSON)",
                        placeholder='{"D:\\\\CCTV\\\\camera01.mp4": "2026-10-01T09:00:00+10:00"}',
                    )
                if include_videos:
                    st.caption(
                        "Sampling can miss events between frames. Shorter intervals search more frames and require more processing."
                    )

        elif mode == "Connect cameras / webcam":
            for feed in st.session_state.feeds:

                get_hub().attach(st.session_state.camera_owner, feed["address"])

            st.caption(
                "Connect a webcam on this computer using index 0 (or 1 for another webcam), or enter an RTSP / HTTP camera feed. Search all connected sources together."
            )

            with st.expander("Add your phone", expanded=False):

                from phone_cameras import local_address
                import qrcode

                st.write(
                    "Scan a QR code to use your phone as a live camera. No phone app needed."
                )

                phone_name = st.text_input("Phone camera name", value="Phone camera")
                phone_host = st.text_input(
                    "This computer's Wi-Fi IP address", value=local_address()
                )

                st.caption(
                    "Connect both devices to the same Wi-Fi. Keep the phone's camera page open."
                )

                if st.button("Create phone QR code", use_container_width=True):

                    try:
                        phone_address, phone_url = get_hub().pair_phone(
                            st.session_state.camera_owner, phone_host.strip()
                        )

                        st.session_state.feeds.append(

                            {
                                "name": phone_name.strip() or "Phone camera",
                                "location": "Phone",
                                "address": phone_address,
                            }
                        )

                        st.session_state.phone_pairing = {
                            "address": phone_address,
                            "url": phone_url,
                        }
                    except (ValueError, OSError) as error:
                        st.error(f"Could not create phone connection: {error}")
                pairing = st.session_state.get("phone_pairing")

                if pairing and any(
                    feed["address"] == pairing["address"]
                    for feed in st.session_state.feeds

                ):
                    qr_image = qrcode.make(pairing["url"])
                    qr_output = BytesIO()
                    qr_image.save(qr_output, format="PNG")
                    st.image(qr_output.getvalue(), width=240)
                    st.link_button("Open phone camera page", pairing["url"])
                    st.info(
                        "Scan within 10 minutes, then tap Connect camera and allow camera access. The camera will appear in the control room automatically."
                    )

                    st.caption(
                        "The local HTTPS connection uses an Inquisitor certificate. Your phone may ask you to trust it first; some phones require installing a trusted certificate. If the page cannot open, check the IP address and allow the phone camera port through your computer's firewall."
                    )

                    st.download_button(
                        "Download phone connection certificate",
                        get_hub().phone_service.certificate,
                        "inquisitor-phone.crt",
                        mime="application/x-pem-file",
                    )

                    with st.expander("Help with phone camera access"):
                        st.write(
                            "Open the link in Safari or Chrome. If camera access is blocked because the connection is not trusted, install the downloaded certificate on your phone. On iPhone, install it under Settings > General > VPN & Device Management, then enable trust under General > About > Certificate Trust Settings. On Android, use Settings > Security > Encryption & credentials > Install a certificate (names vary by device). Only trust a certificate from your own Inquisitor computer. A fresh certificate is created when Inquisitor restarts."
                        )

            with st.form("connect_source", clear_on_submit=True):
                a, b = st.columns(2)

                name = a.text_input("Source name", placeholder="North gate / Camera 01")
                location = b.text_input("Location", placeholder="Engineering building")

                address = st.text_input(
                    "Feed address or webcam index",
                    type="password",
                    placeholder="rtsp://… or 0",
                )
                connect = st.form_submit_button(
                    "Connect and test source", type="primary"
                )

            if connect:

                if not name.strip() or not address.strip():
                    st.warning("Enter a source name and feed address.")
                else:
                    try:
                        with st.spinner("Connecting to source…"):
                            frame = get_hub().connect(
                                st.session_state.camera_owner, address.strip()
                            )
                        if not any(
                            feed["address"] == address.strip()
                            for feed in st.session_state.feeds
                        ):
                            st.session_state.feeds.append(
                                {
                                    "name": name.strip(),
                                    "location": location.strip(),
                                    "address": address.strip(),
                                }
                            )

                        else:

                            st.info("This source is already connected.")
                        st.session_state.setdefault("camera_previews", {})[
                            address.strip()
                        ] = frame

                        st.session_state.setdefault("camera_status", {})[
                            address.strip()
                        ] = "Connected"
                        st.success(f"Connected: {name}")
                    except Exception:

                        st.error(
                            "Connection failed. Check the feed address, network, and credentials."
                        )

            with camera_wall:
                control_room()
                if st.session_state.feeds:
                    with st.expander("Connected cameras", expanded=False):
                        headings = st.columns([2, 2, 1.4, 1.4])
                        for column, title in zip(headings, ["Camera / source", "Connection status", "Disconnect", "Preview"]):
                            column.markdown(f"**{title}**")
                        states = {state["id"]: state for state in (get_hub().room_state(st.session_state.camera_owner) or [])}
                        descriptors = {feed["name"]: feed for feed in get_hub().descriptors(st.session_state.camera_owner, st.session_state.feeds)}
                        for i, feed in enumerate(st.session_state.feeds):
                            name_cell, status_cell, disconnect_cell, refresh_cell = st.columns([2, 2, 1.4, 1.4])
                            name_cell.write(f"{i + 1}. {feed['name']}")
                            descriptor = descriptors.get(feed["name"], {})
                            state = states.get(descriptor.get("id"), {})
                            status = st.session_state.get("camera_status", {}).get(feed["address"], state.get("status", "Connecting"))
                            if state.get("live"):
                                status = "Connected"
                            elif state.get("status"):
                                status = state["status"]
                            status_cell.write(status)
                            if disconnect_cell.button("Disconnect", key=f"disconnect_{i}"):
                                get_hub().disconnect(st.session_state.camera_owner, feed["address"])
                                st.session_state.feeds.pop(i)
                                st.rerun()
                            if refresh_cell.button("Refresh preview", key=f"preview_{i}"):
                                try:
                                    snapshot(feed["address"])
                                    feed["paused"] = False
                                    st.session_state.setdefault("camera_status", {})[feed["address"]] = "Connected"
                                except Exception:
                                    st.session_state.setdefault("camera_status", {})[feed["address"]] = "Unavailable - check the connection and try again"
                                    st.session_state.setdefault("camera_previews", {}).pop(feed["address"], None)
                                st.rerun()

            sources = list(st.session_state.feeds)
            st.caption(
                "Search connected cameras now, or enable Scheduled monitoring to choose a timeframe."
            )

            if st.toggle("Scheduled monitoring", key="monitor_enabled"):
                st.caption(
                    "Detect objects on connected cameras during a chosen timeframe. Monitoring runs in the background while this application stays running."
                )
            
                feeds = list(st.session_state.feeds)
                selected_feeds = st.multiselect(
                    "Camera sources",
                    range(len(feeds)),
                    default=list(range(len(feeds))),
                    format_func=lambda i: feeds[i]["name"],

                )

                if not feeds:
                    st.info(
                        "Connect a webcam or camera feed using the camera workspace first."
                    )

                start, end = time_window("monitor")
                with st.expander("Advanced monitoring settings"):

                    a, b = st.columns(2)
                    interval = a.number_input(
                        "Check each camera every (seconds)",
                        min_value=1.0,
                        max_value=3600.0,
                        value=5.0,
                    )

                    budget = b.number_input(
                        "Maximum frames analyzed",
                        min_value=1,
                        max_value=100000,
                        value=10000,
                        step=100,
                    )

                job_options = {
                    "kind": "live",
                    "feeds": [feeds[i].copy() for i in selected_feeds],
                    "window_start": start,
                    "window_end": end,
                    "interval": interval,
                    "max_frames": int(budget),
                }

                st.caption(
                    "Matching snapshots and reports are saved locally. This does not record continuous video; search your CCTV recordings separately for past events."
                )

        else:

            st.caption(
                "Image and instant camera searches from this session. Saved media searches and monitoring jobs appear alongside."
            )

            if st.session_state.get("history"):
                st.dataframe(
                    st.session_state.history, use_container_width=True, hide_index=True
                )

            else:
                st.info("Completed searches will appear here.")

with results_panel:

    if job_options is not None:
        query = st.text_input(
            "Describe what you want to find",
            placeholder="red car, person with a backpack, bicycle",
            key="background_query",
        )

        ready = bool(
            job_options.get("feeds")
            if job_options["kind"] == "live"
            else job_options.get("paths")
        )

        launch = st.button(
            (
                "Start monitoring job"
                if job_options["kind"] == "live"
                else "Search images and videos"
            ),
            type="primary",
            use_container_width=True,
            disabled=not ready or not description(query),
        )

        if not ready:
            st.caption(
                "Choose camera sources first."
                if job_options["kind"] == "live"
                else "Select files or a folder first."
            )

        if launch:
            try:
                if manifest_text.strip():
                    mapping = json.loads(manifest_text)
                    if not isinstance(mapping, dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in mapping.items()
                    ):
                        
                        raise ValueError(
                            "Recording times must be a JSON object mapping file paths to ISO timestamps."
                        )
                    job_options["recording_times"] = {
                        str(Path(k).expanduser().resolve()): v
                        for k, v in mapping.items()
                    }

                job_options.update(
                    query=description(query),
                    confidence=confidence,
                    tiling=detailed,
                    segment=segment,
                )

                st.session_state["selected_job"] = job_manager().submit(job_options)
                st.session_state["job_view"] = st.session_state["selected_job"]
                st.session_state.setdefault("current_jobs", {})[mode] = (
                    st.session_state["selected_job"]
                )

                st.rerun()
            except Exception as error:
                st.error(str(error))

    elif mode != "Search history":

        query = st.text_input(
            "Describe what you want to find",
            placeholder="red car, laptop, backpack, bicycle",
            key="search_query",
        )

        ready = bool(sources)
        run = st.button(
            "Search connected sources" if sources else "Run visual search",
            type="primary",
            use_container_width=True,
            disabled=not ready or not description(query),
        )

        if not ready:
            st.caption("Connect a camera first.")

        if run:

            try:

                target, items = description(query), []
                if not target:
                    raise ValueError("Enter an object description.")
                
                items = [(feed["name"], feed["location"], feed) for feed in sources]
                if not items:

                    raise ValueError("No images or camera sources available to search.")
                if sources:

                    get_hub().set_active(
                        st.session_state.camera_owner, sources[0]["address"], target
                    )

                with st.spinner("Loading the local GPU search engine…"):
                    finder = engine()
                st.session_state["snapshot_mode"] = mode
                st.session_state.setdefault("current_jobs", {})[mode] = "snapshot"
                st.session_state["selected_job"] = "snapshot"
                st.session_state["job_view"] = "snapshot"
                st.session_state.update(

                    results=[], errors=[], scanned=0, searched=True, last_query=target
                )

                progress = st.progress(0.0, text="Starting visual search")
                for index, (name, location, source) in enumerate(items):

                    progress.progress(
                        index / len(items), text=f"Scanning {index + 1} of {len(items)}"
                    )

                    if isinstance(source, dict):
                        get_hub().set_active(
                            st.session_state.camera_owner, source["address"], target
                        )

                    try:
                        image = (
                            snapshot(source["address"])
                            if isinstance(source, dict)
                            else (
                                source
                                if isinstance(source, Image.Image)
                                else read_picture(source)
                            )
                        )

                        captured = datetime.now().isoformat(timespec="seconds")
                        finder.load_models(segment=segment)
                        annotated, found = finder.explore(

                            image,
                            target,
                            box_threshold=confidence,
                            tiling=detailed,
                            segment=segment,
                        )

                        st.session_state.scanned += 1
                        if found:
                            annotated.thumbnail((1400, 1000))
                            st.session_state.results.append(
                                {
                                    "source": name,
                                    "location": location,
                                    "captured": captured,
                                    "matches": found,
                                    "score": max(item["confidence"] for item in found),
                                    "image": annotated,
                                }
                            )

                    except Exception as error:
                        message = (
                            "Source unavailable or analysis failed"
                            if isinstance(source, dict)
                            else str(error)
                        )

                        st.session_state.errors.append(
                            {"source": name, "error": message}
                        )

                get_hub().set_active(st.session_state.camera_owner)
                progress.empty()

                st.session_state.results.sort(
                    key=lambda item: item["score"], reverse=True
                )

                history = st.session_state.setdefault("history", [])
                history.insert(
                    0,

                    {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "query": target,
                        "scope": mode,
                        "scanned": st.session_state.scanned,
                        "matching_images": len(st.session_state.results),
                    },
                )

                del history[50:]
                st.rerun()

            except Exception as error:
                st.error(str(error))

            finally:
                get_hub().set_active(st.session_state.camera_owner)


def snapshot_results():

    results = st.session_state.results

    if st.session_state.searched:
        status = (
            "Failed"
            if not st.session_state.scanned and st.session_state.errors
            else "Completed"
        )
        st.write(
            f"**{status} | {st.session_state.scanned} images checked | {len(results)} matches**"
        )

    if results:
        st.caption(
            f"{len(results)} matches"
        )
        report = [
            {key: value for key, value in item.items() if key != "image"}
            for item in results
        ]

        st.download_button(
            "Export search report",
            json.dumps(
                {"query": st.session_state.last_query, "results": report}, indent=2
            ),
            "inquisitor-search.json",
            "application/json",
        )

        columns = st.columns(2)

        for index, item in enumerate(results):
            with columns[index % 2], st.container(border=True):
                preview = item["image"].copy()
                preview.thumbnail((320, 200))
                st.image(preview, width=preview.width)
                st.caption(f"**{item['source']}** · {item['captured']}")
    elif st.session_state.errors and not st.session_state.scanned:
        st.error("No sources could be searched. Check the source errors and try again.")

    elif st.session_state.searched:

        st.info(
            f"No matches found for “{st.session_state.last_query}”. Try a shorter description or lower detection confidence."
        )

    else:
        st.markdown(
            '<div class="empty"><div class="eyebrow"><span class="signal"></span>AWAITING YOUR FIRST QUERY</div><p>Connect camera feeds or select files or a folder to start a visual search.</p></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.errors:
        with st.expander(
            f"{len(st.session_state.errors)} source(s) could not be scanned"
        ):
            st.dataframe(
                st.session_state.errors, use_container_width=True, hide_index=True
            )


@st.fragment(run_every="3s")

def background_results(history=False):
    manager = job_manager()
    jobs = manager.list()

    if history:

        st.subheader("Monitoring and media search jobs")
        if any(manager.undo_batches):
            st.caption("Cleared jobs can be restored. Running searches stay stopped.")
            if st.button("Undo clear", key="undo_clear"):
                try:
                    restored = manager.undo_clear()
                    if restored:
                        st.session_state["selected_job"] = restored[0]
                        st.session_state.pop("job_view", None)
                    st.rerun()
                except (OSError, ValueError) as error:
                    st.error(f"Unable to restore jobs: {error}")

        if jobs and st.button("Clear all jobs and results", key="clear_all_jobs"):
            try:
                manager.clear(undoable=True)
                for key in ("selected_job", "job_view"):
                    st.session_state.pop(key, None)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"Unable to clear jobs: {error}")

        if not jobs:
            st.caption(
                "No jobs yet. Start a media search or schedule camera monitoring."
            )
            return
        
        st.dataframe(
            [
                {
                    "id": j["id"][:8],
                    "query": j["query"],
                    "status": j["status"],
                    "analyzed": j["scanned"],
                    "matches": len(j["results"]),
                    "created": j["created"],
                }
                for j in jobs
            ],
            use_container_width=True,
            hide_index=True,
        )

        ids = [j["id"] for j in jobs]
        preferred = st.session_state.get("selected_job", ids[0])
        selection = st.selectbox(

            "View search",
            ids,
            index=ids.index(preferred) if preferred in ids else 0,
            format_func=lambda value: next(
                f"{j['query']} | {j['status']} | {value[:8]}"
                for j in jobs
                if j["id"] == value
            ),
            key="job_view",
        )

    else:
        st.subheader("Search results")
        selection = st.session_state.get("current_jobs", {}).get(st.session_state.mode)

        if (
            selection == "snapshot"
            and st.session_state.get("snapshot_mode") == st.session_state.mode
        ):
            snapshot_results()
            return
        
        if selection not in [j["id"] for j in jobs]:
            st.info(
                "Start a search to see progress and matches here. Previous searches are available in Search history."
            )
            return
        
    job = next(j for j in jobs if j["id"] == selection)

    status_names = {
        "queued": "Queued",
        "scheduled": "Scheduled",
        "loading": "Loading search engine",
        "running": "Searching",
        "completed": "Completed",
        "cancelled": "Stopped",
        "failed": "Failed",
        "interrupted": "Interrupted",
        "limit reached": "Frame limit reached",
    }

    label = status_names.get(job["status"], job["status"].capitalize())
    st.write(
        f"**{label} | {job['scanned']} images / frames checked | {len(job['results'])} matches**"
    )

    st.caption(f"Search: {job['query']}")

    if job["status"] == "completed":
        st.success("Search completed.")
    elif job["status"] in ("failed", "interrupted"):

        st.error(job["message"])
    else:
        st.caption(job["message"])

    if job["status"] == "scheduled":
        st.info(
            f"Monitoring starts at {job['window_start']} and ends at {job['window_end']}."
        )

    if history:

        if st.button("Clear this job and results", key="clear_" + selection):
            try:
                manager.clear(selection, undoable=True)
                for key in ("selected_job", "job_view"):
                    st.session_state.pop(key, None)
                st.rerun()
            except (OSError, ValueError) as error:
                st.error(f"Unable to clear job: {error}")

    if job["status"] not in TERMINAL:
        if st.button("Stop this job", key="cancel_" + selection):
            manager.cancel(selection)
            st.rerun()

    st.download_button(
        "Export job report",
        json.dumps(job, indent=2),
        f"inquisitor-{selection[:8]}.json",
        "application/json",
        key="report_" + selection,
    )

    if job["errors"]:
        with st.expander("Source errors"):
            st.dataframe(job["errors"], use_container_width=True, hide_index=True)
    ordered = sorted(job["results"], key=lambda r: r["score"], reverse=True)

    if not ordered:

        if job["status"] not in TERMINAL:
            st.info(
                "No matches yet. Results will appear here as the search progresses."
            )

        elif job["status"] in ("failed", "interrupted"):
            st.info(
                "Check the source errors or reconnect your sources, then start a new search."
            )

        elif job["status"] == "cancelled":
            st.info("Search stopped. No matching frames were saved.")

        else:
            st.info(
                "No matches found. Try a shorter description or lower detection confidence in Advanced detection settings."
            )

        return
    page = st.number_input(

        "Results page (24 per page)",
        min_value=1,
        max_value=max(1, (len(ordered) + 23) // 24),
        value=1,
        key="page_" + selection,
    )

    columns = st.columns(2)

    for index, item in enumerate(ordered[(page - 1) * 24 : page * 24]):

        with columns[index % 2], st.container(border=True):
            file = manager.root / selection / item["image_file"]

            if file.exists():
                preview = read_picture(file)
                preview.thumbnail((320, 200))
                st.image(preview, width=preview.width)
            timeframe = ""
            if item.get("captured"):
                timeframe = item["captured"]
            elif item.get("video_seconds") is not None:
                timeframe = f"Video time: {timedelta(seconds=round(item['video_seconds']))}"
            st.caption(f"**{item['source']}**" + (f" · {timeframe}" if timeframe else ""))


with results_panel:
    background_results(history=mode == "Search history")
