# Inquisitor

Search photos, recorded videos, and live cameras by describing what you want to find. Try something like `red car`, `backpack`, or `laptop, bicycle`.

Inquisitor runs the detection models on your computer and shows matching images or frames in a Streamlit web interface.

Built by **Md Rifat Bin Yusuf**.

## What it can do

- Search individual files, folders, mounted drives, and network folders your computer can access.
- Search images, videos, or both, including subfolders.
- Connect webcams, CCTV feeds, and phone cameras.
- Monitor connected cameras during a chosen timeframe.
- Show live camera previews, matching snapshots, and search history.
- Export search reports as JSON.
- Add segmentation masks and search smaller image tiles when needed.

## Install on Windows

These instructions are for a Windows 10 or 11 PC with 64-bit Python 3.12, the Python version used during development. You need internet access for installation and the first model downloads. An NVIDIA GPU is helpful but optional.

### 1. Install Python

Install Python 3.12 using the [Windows downloads page](https://www.python.org/downloads/windows/). If using the traditional installer, select **Add python.exe to PATH** and include the Python launcher and Tcl/Tk components (the file picker uses Tk).

Open PowerShell from the Start menu and check:

```powershell
py -3.12 --version
```

It should show `Python 3.12.x`. If the command is not found, reopen PowerShell after installing Python and check that Python 3.12 and the launcher are installed.

### 2. Download Inquisitor

The easiest option is to open [this repository](https://github.com/RifatBYusuf/inquisitor), click **Code > Download ZIP**, then right-click the downloaded ZIP and choose **Extract All**. Open the extracted folder that contains `app.py` and `README.md`.

To open PowerShell in that folder, type `powershell` into File Explorer's address bar and press Enter. Run all remaining setup commands in that window. Extract the ZIP first; do not run the app from inside the ZIP.

Alternatively, install [Git for Windows](https://git-scm.com/install/windows), reopen PowerShell, and run these commands from the folder where you want to keep the project:

```powershell
git clone https://github.com/RifatBYusuf/inquisitor.git
cd inquisitor
```

Use either the ZIP method or Git; you do not need both.

### 3. Create the virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

The `.venv` folder keeps the app's packages separate from other Python programs. These commands use its Python directly, so you do not need to activate it or change PowerShell's execution policy.

### 4. Install the packages

Install PyTorch and TorchVision:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision
```

For an NVIDIA GPU, use the matching CUDA installation command from the [PyTorch installation page](https://pytorch.org/get-started/locally/), using `.\.venv\Scripts\python.exe -m pip` in place of `pip`. The app uses CUDA when PyTorch can access it; otherwise it uses the CPU. CPU searches can be slow.

Install the remaining packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m pip install transformers huggingface_hub numpy pillow sam2==1.1.0
```

`requirements-web.txt` only covers the web interface and camera packages. The second command installs the AI packages too. The `sam2==1.1.0` package matches the package installed in the current development environment and is a community distribution of SAM 2, credited below.

Internet access is needed to install packages and download the models on first use. Model downloads can take a while and need disk space. SAM 2 is loaded when segmentation is enabled.

Check the installed packages:

```powershell
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import streamlit, torch, torchvision, transformers, sam2, cv2; print('Imports OK'); print('CUDA available:', torch.cuda.is_available())"
```

`Imports OK` means the main packages can load. `CUDA available: False` means the app will use the CPU. If a package import fails, complete the installation commands above before starting the app.

## Start the app

After installation, double-click **Run Inquisitor.bat** in the project folder, or run this from PowerShell in that folder:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) if the browser does not open automatically. Keep the terminal open while using the app. Press `Ctrl+C` in the terminal to stop it.

The batch launcher makes the app reachable on your local network. Folder selection and webcam indexes always refer to the computer running Inquisitor.

On later runs, just open **Run Inquisitor.bat** again. You do not need to reinstall the packages. If port 8501 is already in use, stop the other instance or add `--server.port 8502` to the terminal launch command, then open `http://localhost:8502`.

## Try your first search

1. Put a few JPG or PNG images in a folder. Include an image with an obvious object, such as a car or backpack.
2. Start Inquisitor and open **Search files / folders**.
3. Select **Images only**, click **Choose folder / drive**, and choose that folder.
4. Enter `car` or another object present in your images, then start the search.
5. Wait for the first model download and loading to finish. This first search takes longer than later searches.
6. Review the matching images in the results panel. If there are no matches, try a simpler description or lower the confidence in the sidebar.

Once this works, try a larger folder, recorded videos, or the camera options below.

## Search files and folders

1. Open **Search files / folders**.
2. Choose **Images only**, **Videos only**, or **Images and videos**.
3. Use **Choose files** or **Choose folder / drive** to add your media.
4. Enter a short description and start the search.

Use **Advanced settings** to change the file limit, include subfolders, set video offsets, or adjust the sampling interval. The default file limit is 200. Videos are sampled every 2 seconds by default, rather than checking every frame.

For a calendar timeframe search, images use their EXIF capture time, falling back to file modification time. For videos, supply the recording start time so the app can map video offsets to actual dates and times. You can supply separate recording times for multiple clips using the JSON field.

Supported image extensions: JPG, JPEG, PNG, WEBP, BMP, TIF, TIFF.

Supported video extensions: MP4, AVI, MOV, MKV, M4V, WEBM, MTS, M2TS. Playback and decoding depend on the file's codec.

## Connect cameras

Open **Connect cameras / webcam**, enter a source name and location, then enter its address:

- `0` for the first webcam, or `1` for another webcam.
- An RTSP or HTTP stream address for a network camera.

Click **Connect and test source**. Connected cameras appear in the control room. Expand **Connected cameras** to see their status, refresh previews, or disconnect them.

Enter a description to search the current camera frames. To keep checking cameras over time, enable **Scheduled monitoring**, choose a timeframe and sampling interval, then start the job. Leave the app and computer running for monitoring to continue.

### Use your phone

1. Connect your phone and computer to the same Wi-Fi.
2. Expand **Add your phone** and check the computer's Wi-Fi IP address.
3. Create a QR code and scan it within 10 minutes.
4. Open the link, tap **Connect camera**, and allow camera access.

Keep the phone's camera page open. The connection uses a local HTTPS certificate. If your phone blocks camera access, use the certificate download and help in the app. Only trust a certificate from your own computer. A new certificate is created when Inquisitor restarts.

## Results and settings

Matching images and frames appear in result cards with the source name and timestamp or video offset. Open **Search history** to review saved jobs and export their reports.

Reports and matching snapshots are stored in `search_data/` inside the project folder. Keep that folder if you want to keep your history when moving the app to another folder. It is excluded from Git uploads.

Under **Advanced detection settings**:

- **Detection confidence** defaults to 55%. Lower it to include more possible matches, or raise it to filter out weaker detections. This score is not a guarantee that a match is correct.
- **Search small objects in tiles** checks smaller patches of larger images. It takes more processing time.
- **Generate segmentation masks** uses SAM 2 to highlight the detected objects.

## How it works

Grounding DINO takes your description and an image, then predicts object boxes and confidence scores. Inquisitor filters the boxes by the selected confidence and removes overlapping detections of the same class.

If tiling is enabled, it also searches image patches. If segmentation is enabled, the detected boxes are passed to SAM 2 to produce masks. The app draws the results and saves matching frames and job reports in `search_data/`.

Recorded videos are checked at the selected sampling interval. Live monitoring checks snapshots from connected feeds at its own interval. It does not record continuous video or use SAM 2's video tracking features.

Detection runs locally. The app does not send your media to a hosted AI inference service. The models are downloaded from Hugging Face and cached locally.

## A few things to know

- Short, concrete descriptions usually work better than long requests.
- Color descriptions can produce wrong matches. Review the results yourself; this is not an identity verification tool.
- Sampling can miss events between frames. A shorter interval checks more frames and takes longer.
- Closing the app stops active searches and monitoring.
- The file picker needs a desktop session on the computer running the app.
- If a camera fails to connect, check its address, credentials, network access, and whether another program is using the webcam.
- If the phone cannot connect, check the Wi-Fi IP address, certificate, and firewall access for the phone camera port.

## Update the app

Stop Inquisitor first. If you downloaded it with Git, open PowerShell in the project folder and run:

```powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

Check this README for any new AI package requirements, then start the app again. If Git reports local changes, save those changes before updating.

If you used Download ZIP, download and extract the latest ZIP into a new folder, then follow the installation steps there. To keep saved results, copy your old `search_data/` folder into the new project folder before starting it.

## Credits and licenses

The detection and segmentation models come from these projects. Thanks to their authors and contributors.

| Project | Used here | License |
| --- | --- | --- |
| [Grounding DINO — IDEA Research](https://github.com/IDEA-Research/GroundingDINO) | Text-guided object detection with [grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | [Apache 2.0](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) |
| [SAM 2 — Meta](https://github.com/facebookresearch/sam2), using [Jorge Insua's community distribution](https://github.com/JinsuaFeito-dev/segment-anything-2) | Segmentation with [SAM 2.1 Hiera Large](https://huggingface.co/facebook/sam2.1-hiera-large), with [Small](https://huggingface.co/facebook/sam2.1-hiera-small) as a fallback; installed through [sam2 1.1.0](https://pypi.org/project/sam2/1.1.0/) | Apache 2.0 ([original](https://github.com/facebookresearch/sam2/blob/main/LICENSE), [distribution](https://github.com/JinsuaFeito-dev/segment-anything-2/blob/main/LICENSE)) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Grounding DINO model loading and inference | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |

Grounding DINO copyright: 2023–present, IDEA Research. SAM 2 copyright: Meta Platforms, Inc. and affiliates. Transformers copyright: the Hugging Face team.

SAM 2 also contains optional connected-components code adapted from `cc_torch`, covered by its [BSD 3-Clause license](https://github.com/facebookresearch/sam2/blob/main/LICENSE_cctorch).

Papers:

- Shilong Liu et al., [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023.
- Nikhila Ravi et al., [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024.

Other dependencies include Streamlit, PyTorch, TorchVision, OpenCV, NumPy, Pillow, cryptography, and qrcode. Each keeps its own license. These credits do not replace the license copies and notices required when redistributing third-party code or model weights.
