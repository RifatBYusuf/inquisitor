"""Run Inquisitor's detection models, one search at a time."""

import threading

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from sam2.sam2_image_predictor import SAM2ImagePredictor

from inquisitor import (
    GDINO_MODEL_ID,
    SAM2_MODEL_ID,
    SAM2_FALLBACK_ID,
    normalize_prompt,
    clean_label,
    object_color,
    overlay_masks,
    search_views,
    class_aware_nms,
    expand_box,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = gdino_model = sam2_predictor = None
_lock = threading.Lock()


def load_models(segment=False):
    global processor, gdino_model, sam2_predictor
    with _lock:
        if gdino_model is None:
            processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
            gdino_model = (
                AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID)
                .to(device)
                .eval()
            )
        if segment and sam2_predictor is None:
            try:
                sam2_predictor = SAM2ImagePredictor.from_pretrained(
                    SAM2_MODEL_ID, device=device
                )
            except Exception:
                sam2_predictor = SAM2ImagePredictor.from_pretrained(
                    SAM2_FALLBACK_ID, device=device
                )
            sam2_predictor.model.eval()
    return device


def detect_view(image, labels, box_threshold, text_threshold, offset_x, offset_y):
    inputs = processor(images=image, text=[labels], return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = gdino_model(**inputs)
    try:
        result = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        result = processor.post_process_grounded_object_detection(
            outputs, input_ids=inputs.input_ids, target_sizes=[image.size[::-1]]
        )[0]
    boxes, scores = result["boxes"].detach().cpu(), result["scores"].detach().cpu()
    found = [str(x) for x in result.get("text_labels", result.get("labels", []))]
    keep = [i for i, s in enumerate(scores) if float(s) >= box_threshold]
    if not keep:
        return torch.empty((0, 4)), torch.empty((0,)), []
    boxes, scores, found = boxes[keep], scores[keep], [found[i] for i in keep]
    boxes[:, [0, 2]] += offset_x
    boxes[:, [1, 3]] += offset_y
    return boxes, scores, found


def explore(
    image,
    prompt,
    box_threshold=0.25,
    text_threshold=0.20,
    nms_iou=0.55,
    tiling=True,
    progress=None,
    segment=True,
):
    labels = normalize_prompt(prompt)
    if not labels:
        raise ValueError("Enter an object description, such as red car or laptop.")
    image = image.convert("RGB")
    # SAM2 remembers the image, so each search needs to finish before the next starts.
    with _lock, torch.inference_mode():
        if gdino_model is None or (segment and sam2_predictor is None):
            raise RuntimeError("The visual search engine is still loading.")
        views = search_views(image, tiling)
        boxes_all, scores_all, labels_all = [], [], []
        for number, (view, x, y) in enumerate(views, 1):
            if progress:
                progress(
                    0.7 * number / len(views),
                    f"🔎 Exploring picture patch {number} of {len(views)}…",
                )
            boxes, scores, found = detect_view(
                view, labels, box_threshold, text_threshold, x, y
            )
            if len(boxes):
                boxes_all.append(boxes)
                scores_all.append(scores)
                labels_all.extend(found)
        if boxes_all:
            boxes, scores, found = class_aware_nms(
                torch.cat(boxes_all), torch.cat(scores_all), labels_all, nms_iou
            )
        else:
            boxes, scores, found = torch.empty((0, 4)), torch.empty(0), []
        masks = []
        if len(boxes) and segment:
            sam2_predictor.set_image(np.asarray(image))
            for index, box in enumerate(boxes):
                if progress:
                    progress(
                        0.7 + 0.25 * (index + 1) / len(boxes),
                        f"🎨 Coloring discovery {index + 1} of {len(boxes)}…",
                    )
                candidates, quality, _ = sam2_predictor.predict(
                    box=expand_box(box.tolist(), image.width, image.height),
                    multimask_output=True,
                )
                masks.append(
                    candidates[int(np.argmax(quality)) if len(quality) else 0] > 0
                )
        result = overlay_masks(image, masks) if masks else image.copy()
        draw = ImageDraw.Draw(result)
        discoveries = []
        for index, (box, label, score) in enumerate(zip(boxes, found, scores)):
            x1, y1, x2, y2 = map(float, box.tolist())
            color = object_color(index)
            draw.rounded_rectangle(
                (x1, y1, x2, y2),
                radius=8,
                outline=color,
                width=max(3, round(min(image.size) / 250)),
            )
            tag = clean_label(label).title()
            top = max(0, y1 - 25)
            draw.rounded_rectangle(
                (x1, top, min(image.width, x1 + max(120, len(tag) * 9)), top + 25),
                radius=5,
                fill=color,
            )
            draw.text((x1 + 6, top + 5), tag, fill="white")
            discoveries.append(
                {
                    "object": clean_label(label).title(),
                    "confidence": float(score),
                    "box": [x1, y1, x2, y2],
                }
            )
        if progress:
            progress(1.0, "🌟 Adventure complete!")
        return result, discoveries
