"""
Live camera view with a big visual cue when the LD2450 radar detects a target.

Radar is wired to /dev/ttyAMA0 (not /dev/serial0 - that alias points to
ttyAMA10 on this Pi). Runs the radar read loop in a background thread so it
doesn't block the camera preview.
"""
import threading
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2

from ld2450_radar import LD2450

RADAR_PORT = "/dev/ttyAMA0"
SAVE_DIR = "radar_detections"
SNAPSHOT_COOLDOWN = 3  # seconds between saved snapshots

# The LD2450 can latch onto static clutter (furniture, walls) and report it as
# a permanent target at speed_cm_s == 0. Gate presence on actual motion so
# those phantom locks don't count, and hold "present" briefly after the last
# real movement so pausing in front of it doesn't flicker to clear.
MOTION_HOLD_SECONDS = 2.0

import os
os.makedirs(SAVE_DIR, exist_ok=True)

state = {"present": False, "targets": []}
state_lock = threading.Lock()


def radar_loop():
    radar = LD2450(port=RADAR_PORT)
    print(f"Radar listening on {RADAR_PORT}")
    last_motion_time = 0.0
    while True:
        targets = radar.read_frame() or []
        moving = [t for t in targets if t["speed_cm_s"] != 0]
        now = time.time()
        if moving:
            last_motion_time = now
        with state_lock:
            state["present"] = (now - last_motion_time) < MOTION_HOLD_SECONDS
            state["targets"] = targets


def main():
    threading.Thread(target=radar_loop, daemon=True).start()

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    )
    picam2.start()
    time.sleep(2)

    last_snapshot = 0
    print("Showing live view. Press q to quit.")

    try:
        while True:
            frame = picam2.capture_array()

            with state_lock:
                present = state["present"]
                targets = list(state["targets"])

            # frame is RGB888 at this point (converted to BGR only when shown/saved below)
            color = (255, 0, 0) if present else (0, 200, 0)
            label = "DETECTED" if present else "clear"

            cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, frame.shape[0] - 1), color, 12)
            cv2.putText(frame, label, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)

            y = 120
            for t in targets:
                text = f"id{t['id']} x={t['x_mm']}mm y={t['y_mm']}mm speed={t['speed_cm_s']}cm/s"
                cv2.putText(frame, text, (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                y += 30

            cv2.imshow("radar view", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            now = time.time()
            if present and now - last_snapshot > SNAPSHOT_COOLDOWN:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"{SAVE_DIR}/radar_{timestamp}.jpg"
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"Saved {filename}")
                last_snapshot = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
