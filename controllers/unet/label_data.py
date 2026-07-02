#!/home/prashanna/miniforge3/envs/rl/bin/python
"""
GroundingDINO + SAM auto-labeling for agricultural scenes.

Reads images from dataset/images/
  → GroundingDINO detects objects by text prompt
  → SAM generates pixel-perfect masks for each detection
  → Builds multi-class segmentation mask
Saves masks to dataset/masks/
Saves overlays to dataset/overlays/
"""
import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image

from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
IMG_DIR = os.path.join(BASE, 'dataset', 'images')
MASK_DIR = os.path.join(BASE, 'dataset', 'masks')
OVERLAY_DIR = os.path.join(BASE, 'dataset', 'overlays')
MODEL_DIR = os.path.join(BASE, 'models')

GDINO_WEIGHTS = os.path.join(MODEL_DIR, 'groundingdino_swint_ogc.pth')
GDINO_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'groundingdino', 'config', 'GroundingDINO_SwinT_OGC.py')
SAM_WEIGHTS = os.path.join(MODEL_DIR, 'sam_vit_b_01ec64.pth')

CLASSES = {
    0: {"name": "soil",       "prompt": "sandy soil ground dirt path",      "color": (80, 60, 20)},
    1: {"name": "vegetation", "prompt": "green vegetation tree bush plant",  "color": (30, 150, 30)},
    2: {"name": "obstacle",   "prompt": "building house wall fence",         "color": (200, 30, 30)},
    3: {"name": "pedestrian", "prompt": "person human pedestrian",           "color": (50, 50, 255)},
    4: {"name": "sky",        "prompt": "blue sky clouds",                   "color": (180, 180, 180)},
}

TEXT_PROMPT = " . ".join([c["prompt"] for c in CLASSES.values()])
BATCH_SIZE = 4
TEXT_THRESHOLD = 0.25
BOX_THRESHOLD = 0.30


def load_models(device):
    print("Loading GroundingDINO...")
    groundingdino_path = os.path.join(MODEL_DIR, 'groundingdino_swint_ogc.pth')
    config_path = os.path.join(MODEL_DIR, 'GroundingDINO_SwinT_OGC.py')

    if not os.path.exists(config_path):
        print(f"Downloading GroundingDINO config...")
        import urllib.request
        url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        urllib.request.urlretrieve(url, config_path)

    gd_model = load_model(config_path, groundingdino_path, device=str(device))

    print("Loading SAM...")
    sam = sam_model_registry["vit_b"](checkpoint=SAM_WEIGHTS)
    sam.to(device)
    sam_pred = SamPredictor(sam)

    print("Models loaded.")
    return gd_model, sam_pred


def detect_and_segment(image_np, gd_model, sam_predictor, device):
    h, w = image_np.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    sam_predictor.set_image(image_np)

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(device)

    all_boxes = []
    all_labels = []

    for class_id, class_info in CLASSES.items():
        boxes, logits, phrases = predict(
            model=gd_model,
            image=image_tensor,
            caption=class_info["prompt"],
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            device=device,
        )

        if len(boxes) == 0:
            continue

        for box, logit, phrase in zip(boxes, logits, phrases):
            all_boxes.append(box.cpu().numpy())
            all_labels.append(class_id)

    if len(all_boxes) == 0:
        return mask

    for i, (box, class_id) in enumerate(zip(all_boxes, all_labels)):
        cx, cy, bw, bh = box
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        box_xyxy = np.array([[x1, y1, x2, y2]], dtype=np.float32)

        masks, scores, _ = sam_predictor.predict(
            box=box_xyxy,
            multimask_output=False,
        )

        m = masks[0] if masks.ndim == 3 else masks
        if m.ndim == 3:
            m = m[0]
        mask[m > 0.5] = class_id

    unmask = mask == 0
    if np.sum(unmask) > 0:
        from scipy import ndimage
        soil_region = np.zeros_like(mask, dtype=bool)
        soil_region[h//2:, :] = True
        soil_region = ndimage.binary_fill_holes(soil_region & unmask)
        mask[unmask & soil_region] = 0

    sky_region = np.zeros_like(mask, dtype=bool)
    sky_region[:h//4, :] = True
    mask[unmask & sky_region] = 4

    return mask


def create_overlay(image_np, mask):
    h, w = mask.shape
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, info in CLASSES.items():
        overlay[mask == cid] = info["color"]
    return cv2.addWeighted(image_np, 0.6, overlay, 0.4, 0)


def main():
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(OVERLAY_DIR, exist_ok=True)

    images = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.png')])
    print(f"Found {len(images)} images in {IMG_DIR}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    gd_model, sam_predictor = load_models(device)

    for idx, fname in enumerate(images):
        img_path = os.path.join(IMG_DIR, fname)
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        print(f"[{idx+1}/{len(images)}] Labeling {fname}...", end=" ")

        mask = detect_and_segment(image_rgb, gd_model, sam_predictor, device)

        cv2.imwrite(os.path.join(MASK_DIR, fname), mask)

        overlay = create_overlay(image_rgb, mask)
        cv2.imwrite(os.path.join(OVERLAY_DIR, fname),
                    cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        unique, counts = np.unique(mask, return_counts=True)
        stats = {CLASSES.get(u, {}).get("name", f"class{u}"): int(c)
                 for u, c in zip(unique, counts)}
        print(f"classes: {stats}")

    print(f"\nDone - {len(images)} masks saved to {MASK_DIR}")


if __name__ == '__main__':
    main()
