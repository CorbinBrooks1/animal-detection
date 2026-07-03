from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import os

# Things we want to detect
TARGETS = {"cat", "bird"}
IGNORE = {"dog", "person"}

SAVE_DIR = "animal_detections"
os.makedirs(SAVE_DIR, exist_ok=True)

# Load small YOLO model
model = YOLO("yolov8n.pt")

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
)
picam2.start()
time.sleep(2)

prev_gray = None
last_detection_time = 0
cooldown = 5

print("Watching for animals... Press Ctrl+C to stop.")

try:
    while True:
        frame = picam2.capture_array()

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray is None:
            prev_gray = gray
            continue

        diff = cv2.absdiff(prev_gray, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        motion_amount = cv2.countNonZero(thresh)

        if motion_amount > 8000:
            now = time.time()

            if now - last_detection_time > cooldown:
                print("Motion detected, checking with AI...")

                results = model(frame, verbose=False)
                detected_names = []

                for result in results:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        name = model.names[class_id]

                        if confidence > 0.45:
                            detected_names.append(name)

                print("Detected:", detected_names)

                found_target = any(name in TARGETS for name in detected_names)
                found_ignore = any(name in IGNORE for name in detected_names)

                if found_target and not found_ignore:
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{SAVE_DIR}/animal_{timestamp}.jpg"

                    annotated = results[0].plot()
                    cv2.imwrite(filename, annotated)

                    print(f"TARGET ANIMAL FOUND! Saved {filename}")

                    # Later: trigger speaker/light here
                    # Example:
                    # activate_deterrent()

                elif found_ignore:
                    print("Ignored because dog/person was detected.")

                else:
                    print("Motion, but no target animal detected.")

                last_detection_time = now

        prev_gray = gray

except KeyboardInterrupt:
    print("Stopped.")

finally:
    picam2.stop()
