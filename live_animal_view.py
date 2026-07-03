from picamera2 import Picamera2
from ultralytics import YOLO
import cv2
import time
import os
from datetime import datetime

model = YOLO("yolov8n.pt")

# Folder where detected human images will be saved
SAVE_FOLDER = "human_detections"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# 2 minutes = 120 seconds
COOLDOWN_SECONDS = 120
last_save_time = 0

picam2 = Picamera2()
picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720), "format": "RGB888"}
    )
)
picam2.start()
time.sleep(2)

print("Live AI view running. Press q to quit.")
print("Images will save when a person is detected, max once every 2 minutes.")

while True:
    frame = picam2.capture_array()

    results = model(frame, verbose=False)

    annotated_frame = results[0].plot()

    person_detected = False

    # Check all detected boxes
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        if class_name == "person":
            person_detected = True
            break

    current_time = time.time()

    if person_detected and current_time - last_save_time >= COOLDOWN_SECONDS:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{SAVE_FOLDER}/person_{timestamp}.jpg"

        # Save the annotated image with boxes/labels
        cv2.imwrite(filename, annotated_frame)

        print(f"Person detected. Saved image: {filename}")

        last_save_time = current_time

    cv2.imshow("Animal Detector", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()