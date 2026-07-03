from picamera2 import Picamera2, Preview
import time

picam2 = Picamera2()

picam2.configure(
    picam2.create_preview_configuration(
        main={"size": (1280, 720)}
    )
)

picam2.start_preview(Preview.QTGL)
picam2.start()

print("Camera preview running. Press Ctrl+C to stop.")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    picam2.stop()
