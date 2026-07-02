#!/home/prashanna/miniforge3/envs/rl/bin/python
"""
Active Segmentation Controller — loads trained U-Net and performs
real-time semantic segmentation on the robot's camera feed while
exploring randomly.

Moves randomly, captures camera frames, runs U-Net inference,
and displays:
  - Left window:  raw camera feed with move info overlay
  - Right window: color-coded segmentation mask overlay

Also saves raw images to dataset/images/ for future retraining.
"""
from controller import Robot, Camera
import cv2
import numpy as np
import os
import time
import random
import torch
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet_model import UNet

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'unet_checkpoint.pth')
IMG_SIZE = 256
NUM_CLASSES = 5
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_COLORS = {
    0: (80, 60, 20),       # soil — brown
    1: (30, 150, 30),      # vegetation — green
    2: (200, 30, 30),      # obstacle — red
    3: (50, 50, 255),      # pedestrian — blue
    4: (180, 180, 180),    # sky — gray
}
CLASS_NAMES = {0: 'soil', 1: 'veg', 2: 'obs', 3: 'ped', 4: 'sky'}

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

lf = robot.getDevice('front left wheel')
rf = robot.getDevice('front right wheel')
lb = robot.getDevice('back left wheel')
rb = robot.getDevice('back right wheel')
for m in [lf, rf, lb, rb]:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

MAX_VEL = 3.0
TURN_VEL = 2.0
MOVE_DURATION = 5.0
CAPTURE_INTERVAL = 5.0
MAX_CONSECUTIVE = 2
SEG_INTERVAL = 0.15

MOVES = {
    "forward":       ( MAX_VEL,  MAX_VEL),
    "backward":      (-MAX_VEL, -MAX_VEL),
    "sharp_left":    (-TURN_VEL, TURN_VEL),
    "sharp_right":   ( TURN_VEL,-TURN_VEL),
    "gentle_left":   ( MAX_VEL * 0.3, MAX_VEL),
    "gentle_right":  ( MAX_VEL, MAX_VEL * 0.3),
    "spin_left":     (-TURN_VEL * 0.5, TURN_VEL * 0.5),
    "spin_right":    ( TURN_VEL * 0.5,-TURN_VEL * 0.5),
}

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'dataset', 'images')
os.makedirs(save_dir, exist_ok=True)


def _test_display():
    try:
        cv2.namedWindow('__test__', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('__test__')
        return True
    except cv2.error:
        return False


def load_model(device):
    model = UNet(n_channels=3, n_classes=NUM_CLASSES, bilinear=True).to(device)
    ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')}, val_iou={ckpt.get('val_iou'):.4f}")
    return model


@torch.no_grad()
def predict(model, frame_rgb, device):
    img = cv2.resize(frame_rgb, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)
    logits = model(img)
    return logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)


def mask_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in CLASS_COLORS.items():
        color[mask == cid] = rgb
    return color


def overlay_mask(frame_bgr, mask_color, alpha=0.45):
    mask_resized = cv2.resize(mask_color, (frame_bgr.shape[1], frame_bgr.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(frame_bgr, 1 - alpha, mask_resized, alpha, 0)


def draw_class_legend(img):
    y = 25
    for cid, name in CLASS_NAMES.items():
        color = CLASS_COLORS[cid]
        cv2.rectangle(img, (10, y - 12), (26, y + 2), color, -1)
        cv2.putText(img, name, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        y += 20


def pick_move(history):
    candidates = [m for m in MOVES if history.count(m) < MAX_CONSECUTIVE]
    if not candidates:
        history.clear()
        candidates = list(MOVES)
    return random.choice(candidates)


def apply_move(name):
    l, r = MOVES[name]
    lf.setVelocity(l)
    lb.setVelocity(l)
    rf.setVelocity(r)
    rb.setVelocity(r)


def main():
    HAS_DISPLAY = _test_display()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(device)

    if HAS_DISPLAY:
        cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera', 960, 720)
        cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Segmentation', 960, 720)

    history = []
    current_move = pick_move(history)
    move_start = time.time()
    last_capture = time.time()
    last_seg = 0
    seg_display = None
    capture_count = len([f for f in os.listdir(save_dir) if f.endswith('.png')])

    print(f"Saving to: {os.path.abspath(save_dir)}")
    print(f"Move duration: {MOVE_DURATION}s | Segmentation: live | Capture every: {CAPTURE_INTERVAL}s")

    while robot.step(timestep) != -1:
        now = time.time()

        if now - move_start >= MOVE_DURATION:
            current_move = pick_move(history)
            history.append(current_move)
            if len(history) > 10:
                history.pop(0)
            move_start = now

        apply_move(current_move)

        if now - last_capture >= CAPTURE_INTERVAL:
            raw = camera.getImage()
            if raw:
                cam_h = camera.getHeight()
                cam_w = camera.getWidth()
                rgb = cv2.cvtColor(
                    np.frombuffer(raw, np.uint8).reshape((cam_h, cam_w, 4)),
                    cv2.COLOR_BGRA2RGB)
                cv2.imwrite(os.path.join(save_dir, f"frame_{capture_count:05d}.png"), rgb)
                capture_count += 1
                last_capture = now
                print(f"  [{capture_count}] Saved frame_{capture_count-1:05d}.png")

        if not HAS_DISPLAY:
            continue

        raw = camera.getImage()
        if not raw:
            continue

        cam_h = camera.getHeight()
        cam_w = camera.getWidth()
        frame_bgr = cv2.cvtColor(
            np.frombuffer(raw, np.uint8).reshape((cam_h, cam_w, 4)),
            cv2.COLOR_BGRA2BGR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        fps_text = ""
        if now - last_seg >= SEG_INTERVAL:
            t0 = time.time()
            mask = predict(model, frame_rgb, device)
            inference_ms = (time.time() - t0) * 1000
            mask_color = mask_to_color(mask)
            overlay = overlay_mask(frame_bgr, mask_color, alpha=0.45)
            seg_display = overlay.copy()
            draw_class_legend(seg_display)
            cv2.putText(seg_display, f"U-Net: {inference_ms:.0f}ms",
                        (10, seg_display.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
            last_seg = now
            fps_text = f" | Seg: {inference_ms:.0f}ms"

        cam_display = frame_bgr.copy()
        cv2.putText(cam_display, f"Move: {current_move}  |  Saved: {capture_count}{fps_text}",
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('Camera', cam_display)
        if seg_display is not None:
            cv2.imshow('Segmentation', seg_display)
        cv2.waitKey(1)

    for m in [lf, lb, rf, rb]:
        m.setVelocity(0)
    if HAS_DISPLAY:
        cv2.destroyAllWindows()
    print(f"\nDone - {capture_count} images saved")


if __name__ == '__main__':
    main()
