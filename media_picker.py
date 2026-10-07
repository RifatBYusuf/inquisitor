"""Open desktop file dialogs outside Streamlit's worker thread."""

import json
from pathlib import Path
import subprocess
import sys


def choose_media(kind, extensions, initial_directory=None):
    options = {
        "kind": kind,
        "extensions": sorted(extensions),
        "directory": initial_directory or str(Path.home()),
    }
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), json.dumps(options)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )
    if result.returncode:
        raise RuntimeError("The desktop file picker could not open.")
    return json.loads(result.stdout)


def _show_dialog(options):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        directory = options["directory"]
        if not Path(directory).is_dir():
            directory = str(Path.home())
        if options["kind"] == "files":
            paths = filedialog.askopenfilenames(
                parent=root,
                title="Choose images or videos to search",
                initialdir=directory,
                filetypes=[
                    ("Supported media", " ".join("*" + ext for ext in options["extensions"])),
                    ("All files", "*.*"),
                ],
            )
        else:
            folder = filedialog.askdirectory(
                parent=root,
                title="Choose a folder or drive to search",
                initialdir=directory,
                mustexist=True,
            )
            paths = [folder] if folder else []
        return list(paths)
    finally:
        root.destroy()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(_show_dialog(json.loads(sys.argv[1]))))
