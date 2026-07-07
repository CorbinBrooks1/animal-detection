"""
Radar-triggered camera. The camera stays off until the radar picks up
something moving, so we're not just snapping photos nonstop like the old
setup did.

Once it wakes up, the camera stays on and YOLO checks every frame for
anything living. Each sighting resets a 20-second "nothing's here" clock;
if that clock runs out, the camera powers back down and takes a cooldown
break before it's allowed to wake up again.

Radar is wired to /dev/ttyAMA0 (not /dev/serial0 - that alias actually
points to ttyAMA10 on this Pi, learned that the hard way).

How it flows:
  IDLE     - camera's off, just watching the radar. Ignores the radar for a
             few seconds after boot since it likes to report garbage right
             when it powers on.
  ACTIVE   - camera's on, preview window up, YOLO chewing on every frame.
  COOLDOWN - camera's off and staying off for a bit, even if the radar fires
             again, so it doesn't just immediately snap back on.
"""
import threading
import time
from datetime import datetime

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO

from ld2450_radar import LD2450

RADAR_PORT = "/dev/ttyAMA0"
SAVE_DIR = "living_detections"

# The LD2450 can lock onto furniture/walls and report them as a permanent
# target sitting at speed_cm_s == 0. Only count it as "present" when
# something's actually moving, and keep that presence flag up for a couple
# seconds after the movement stops so it doesn't flicker off if someone
# just pauses in front of it.
MOTION_HOLD_SECONDS = 2.0

ACTIVE_TIMEOUT = 20  # how long the camera waits with nothing living in frame before giving up
COOLDOWN_SECONDS = 30  # how long the camera rests before it's willing to trigger again
SNAPSHOT_COOLDOWN = 5  # don't save more than one snapshot this often while active
STARTUP_SETTLE_SECONDS = 5  # give the radar a few seconds to settle down before trusting it

LIVING_CLASSES = {"person", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
CONFIDENCE_THRESHOLD = 0.45

WINDOW_NAME = "radar triggered view"

import os
os.makedirs(SAVE_DIR, exist_ok=True)

radar_state = {"present": False}
radar_lock = threading.Lock()


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
        with radar_lock:
            radar_state["present"] = (now - last_motion_time) < MOTION_HOLD_SECONDS


def radar_present():
    with radar_lock:
        return radar_state["present"]


def main():
    threading.Thread(target=radar_loop, daemon=True).start()

    model = YOLO("yolov8n.pt")
    picam2 = Picamera2()
    picam2.configure(
        picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
    )

    state = "IDLE"
    boot_time = time.time()
    active_since = 0.0
    last_living_seen = 0.0
    last_snapshot = 0.0
    cooldown_since = 0.0

    print(f"Settling for {STARTUP_SETTLE_SECONDS}s before arming the radar. Press Ctrl+C to stop.")

    try:
        while True:
            now = time.time()

            if state == "IDLE":
                settled = now - boot_time > STARTUP_SETTLE_SECONDS
                if settled and radar_present():
                    print("Radar triggered -- turning camera on.")
                    picam2.start()
                    time.sleep(2)  # give the sensor a moment to wake up properly
                    cv2.namedWindow(WINDOW_NAME)
                    active_since = now
                    last_living_seen = now
                    state = "ACTIVE"
                else:
                    time.sleep(0.1)
                continue

            if state == "ACTIVE":
                frame = picam2.capture_array()
                results = model(frame, verbose=False)

                living_found = False
                for result in results:
                    for box in result.boxes:
                        name = model.names[int(box.cls[0])]
                        confidence = float(box.conf[0])
                        if confidence > CONFIDENCE_THRESHOLD and name in LIVING_CLASSES:
                            living_found = True

                annotated = results[0].plot()
                cv2.imshow(WINDOW_NAME, annotated)
                cv2.waitKey(1)

                if living_found:
                    last_living_seen = now
                    if now - last_snapshot > SNAPSHOT_COOLDOWN:
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{SAVE_DIR}/living_{timestamp}.jpg"
                        cv2.imwrite(filename, annotated)
                        print(f"Living thing detected. Saved {filename}")
                        last_snapshot = now

                if now - last_living_seen > ACTIVE_TIMEOUT:
                    print(f"No living detections for {ACTIVE_TIMEOUT}s -- turning camera off, cooling down.")
                    picam2.stop()
                    cv2.destroyWindow(WINDOW_NAME)
                    cooldown_since = now
                    state = "COOLDOWN"
                continue

            if state == "COOLDOWN":
                if now - cooldown_since > COOLDOWN_SECONDS:
                    print("Cooldown finished -- ready for next trigger.")
                    state = "IDLE"
                else:
                    time.sleep(0.1)
                continue

    except KeyboardInterrupt:
        pass
    finally:
        if state == "ACTIVE":
            picam2.stop()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
