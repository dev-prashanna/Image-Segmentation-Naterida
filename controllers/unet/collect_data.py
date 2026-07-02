#!/home/prashanna/miniforge3/envs/rl/bin/python
"""
Data Collection — random walk exploration + camera capture every 10 seconds
Each random move persists for 5 seconds, no move repeats more than 2 times consecutively.
"""
from controller import Robot, Camera, Lidar
import cv2
import numpy as np
import os
import time
import random

robot = Robot()
timestep = int(robot.getBasicTimeStep())

camera = robot.getDevice('camera')
camera.enable(timestep)

lidar = robot.getDevice('lidar')
lidar.enable(timestep)
lidar.enablePointCloud()

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
CAPTURE_INTERVAL = 10.0

MOVES = {
    "forward":        ( MAX_VEL,  MAX_VEL),
    "backward":       (-MAX_VEL, -MAX_VEL),
    "sharp_left":     (-TURN_VEL, TURN_VEL),
    "sharp_right":    ( TURN_VEL,-TURN_VEL),
    "gentle_left":    ( MAX_VEL * 0.3, MAX_VEL),
    "gentle_right":   ( MAX_VEL, MAX_VEL * 0.3),
    "spin_left":      (-TURN_VEL * 0.5, TURN_VEL * 0.5),
    "spin_right":     ( TURN_VEL * 0.5,-TURN_VEL * 0.5),
}

save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'dataset', 'images')
os.makedirs(save_dir, exist_ok=True)

def get_lidar_clearance():
    ranges = lidar.getRangeImage()
    if not ranges:
        return 20.0, 20.0, 20.0
    n = len(ranges)
    t = n // 3
    def avg(vals):
        v = [d for d in vals if 0 < d < 20]
        return np.mean(v) if v else 20.0
    return avg(ranges[:t]), avg(ranges[t:2*t]), avg(ranges[2*t:])

def pick_move(history):
    move_names = list(MOVES.keys())
    candidates = [m for m in move_names if history.count(m) < 2]
    if not candidates:
        history.clear()
        candidates = move_names
    return random.choice(candidates)

def apply_move(name):
    l, r = MOVES[name]
    lf.setVelocity(l)
    lb.setVelocity(l)
    rf.setVelocity(r)
    rb.setVelocity(r)

def avoid_obstacle():
    l, c, r = get_lidar_clearance()
    if c > 1.0:
        return False
    if l > r:
        apply_move("sharp_left")
    else:
        apply_move("sharp_right")
    return True

history = []
current_move = pick_move(history)
move_start = time.time()
last_capture = time.time()
capture_count = 0

print(f"Saving to: {os.path.abspath(save_dir)}")
print(f"Move duration: {MOVE_DURATION}s | Capture every: {CAPTURE_INTERVAL}s")
print(f"Constraints: max 2 consecutive same moves")

while robot.step(timestep) != -1:
    now = time.time()

    if now - move_start >= MOVE_DURATION:
        new_move = pick_move(history)
        if new_move != current_move:
            history.append(new_move)
            if len(history) > 10:
                history.pop(0)
        current_move = new_move
        move_start = now
        print(f"  Move: {current_move}")

    if not avoid_obstacle():
        apply_move(current_move)

    elapsed = now - last_capture
    if elapsed >= CAPTURE_INTERVAL:
        raw = camera.getImage()
        if raw:
            cam_h = camera.getHeight()
            cam_w = camera.getWidth()
            bgr = np.frombuffer(raw, np.uint8).reshape((cam_h, cam_w, 4))
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGRA2RGB)

            filename = f"frame_{capture_count:05d}.png"
            cv2.imwrite(os.path.join(save_dir, filename), rgb)
            capture_count += 1
            last_capture = now
            print(f"  [{capture_count}] Saved {filename} ({cam_w}x{cam_h})")

for m in [lf, lb, rf, rb]:
    m.setVelocity(0)
print(f"\nDone - {capture_count} images saved")
