# Inquisitor

Search photos, recorded videos, and live cameras by describing what you want to find. Try something like `red car`, `backpack`, or `laptop, bicycle`.

Inquisitor runs the detection models on your computer and shows matching images or frames in a Streamlit web interface.

## What it can do

- Search individual files, folders, mounted drives, and network folders your computer can access.
- Search images, videos, or both, including subfolders.
- Connect webcams, CCTV feeds, and phone cameras.
- Monitor connected cameras during a chosen timeframe.
- Show live camera previews, matching snapshots, and search history.
- Export search reports as JSON.
- Add segmentation masks and search smaller image tiles when needed.

## Install

The steps below are for Windows PowerShell. Python 3.12 is the version used in the current development environment. Install [Python](https://www.python.org/downloads/) first, then download this repository and open a terminal in its folder.

Create a virtual environment:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

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

## Start the app

Double-click **Run Inquisitor.bat**, or run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open [localhost:8501](http://localhost:8501) if the browser does not open automatically. Keep the terminal open while using the app. Press `Ctrl+C` in the terminal to stop it.

The batch launcher makes the app reachable on your local network. Folder selection and webcam indexes always refer to the computer running Inquisitor.

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

## Credits and licenses

The detection and segmentation models come from these projects. Thanks to their authors and contributors.

| Project | Used here | License |
| --- | --- | --- |
| [Grounding DINO — IDEA Research](https://github.com/IDEA-Research/GroundingDINO) | Text-guided object detection with [grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base) | [Apache 2.0](https://github.com/IDEA-Research/GroundingDINO/blob/main/LICENSE) |
| [SAM 2 — Meta](https://github.com/facebookresearch/sam2) | Segmentation with [SAM 2.1 Hiera Large](https://huggingface.co/facebook/sam2.1-hiera-large), with [Small](https://huggingface.co/facebook/sam2.1-hiera-small) as a fallback | [Apache 2.0](https://github.com/facebookresearch/sam2/blob/main/LICENSE) |
| [Hugging Face Transformers](https://github.com/huggingface/transformers) | Grounding DINO model loading and inference | [Apache 2.0](https://github.com/huggingface/transformers/blob/main/LICENSE) |
| [SAM 2 community distribution — Jorge Insua](https://github.com/JinsuaFeito-dev/segment-anything-2) | The [sam2 1.1.0](https://pypi.org/project/sam2/1.1.0/) package used in the setup above | [Apache 2.0](https://github.com/JinsuaFeito-dev/segment-anything-2/blob/main/LICENSE) |

Grounding DINO copyright: 2023–present, IDEA Research. SAM 2 copyright: Meta Platforms, Inc. and affiliates. Transformers copyright: the Hugging Face team.

SAM 2 also contains optional connected-components code adapted from `cc_torch`, covered by its [BSD 3-Clause license](https://github.com/facebookresearch/sam2/blob/main/LICENSE_cctorch).

Papers:

- Shilong Liu et al., [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499), 2023.
- Nikhila Ravi et al., [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714), 2024.

Other dependencies include Streamlit, PyTorch, TorchVision, OpenCV, NumPy, Pillow, cryptography, and qrcode. Each keeps its own license. These credits do not replace the license copies and notices required when redistributing third-party code or model weights.
