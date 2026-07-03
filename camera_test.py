from picamera2 import Picamera2
import time
import cv2

picam2 = Picamera2()
picam2.start()
time.sleep(2)

previous = None

while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if previous is None:
        previous = gray
        continue
    
    diff = cv2.absdiff(previous, gray)
    thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]

    motion_amount = cv2.countNonZero(thresh)

    if motion_amount > 1000:
        print("Motion Detected!")
    
    previous = gray
    time.sleep(.1)