"""Image-processing helpers for Inquisitor."""

import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms

GDINO_MODEL_ID = "IDEA-Research/grounding-dino-base"
SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"
SAM2_FALLBACK_ID = "facebook/sam2.1-hiera-small"

MASK_ALPHA, TILE_SIZE, TILE_OVERLAP, TILING_START_SIZE, SAM_BOX_PADDING = (
    95,
    1024,
    0.25,
    1400,
    0.05,
)


def normalize_prompt(prompt):
    labels, seen = [], set()
    for part in prompt.split(","):
        label = " ".join(part.strip().lower().split())
        if label and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def clean_label(label):
    label = str(label).strip().lower()
    for prefix in ("a ", "an ", "the "):
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def object_color(index):
    return [
        (23, 105, 255),
        (0, 168, 150),
        (255, 132, 0),
        (142, 68, 255),
        (235, 64, 122),
        (0, 145, 210),
    ][index % 6]


def overlay_masks(image, masks):
    result = image.convert("RGBA")
    for index, mask in enumerate(masks):
        mask = (np.asarray(mask) > 0).astype(np.uint8)
        overlay = np.zeros((result.height, result.width, 4), dtype=np.uint8)
        overlay[mask > 0] = [*object_color(index), MASK_ALPHA]
        result = Image.alpha_composite(result, Image.fromarray(overlay, "RGBA"))
    return result.convert("RGB")


def axis_starts(length):
    if length <= TILE_SIZE:
        return [0]
    step = int(TILE_SIZE * (1 - TILE_OVERLAP))
    starts = list(range(0, length - TILE_SIZE + 1, step))
    if starts[-1] != length - TILE_SIZE:
        starts.append(length - TILE_SIZE)
    return starts


def search_views(image, tiling):
    views = [(image, 0, 0)]
    if tiling and max(image.size) >= TILING_START_SIZE:
        for y in axis_starts(image.height):
            for x in axis_starts(image.width):
                crop = image.crop(
                    (
                        x,
                        y,
                        min(x + TILE_SIZE, image.width),
                        min(y + TILE_SIZE, image.height),
                    )
                )
                views.append((crop, x, y))
    return views


def class_aware_nms(boxes, scores, labels, threshold):
    if not len(boxes):
        return boxes, scores, []
    groups, kept = {}, []
    for index, label in enumerate(labels):
        groups.setdefault(clean_label(label), []).append(index)
    for indices in groups.values():
        selection = torch.tensor(indices, dtype=torch.long)
        local = nms(boxes[selection], scores[selection], threshold)
        kept.extend(selection[local].tolist())
    kept.sort(key=lambda i: float(scores[i]), reverse=True)
    selection = torch.tensor(kept, dtype=torch.long)
    return boxes[selection], scores[selection], [labels[i] for i in kept]


def expand_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    px, py = max(2, (x2 - x1) * SAM_BOX_PADDING), max(2, (y2 - y1) * SAM_BOX_PADDING)
    return np.array(
        [
            max(0, x1 - px),
            max(0, y1 - py),
            min(width - 1, x2 + px),
            min(height - 1, y2 + py),
        ],
        dtype=np.float32,
    )
