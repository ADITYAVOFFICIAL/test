import cv2
import threading
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Open the laptop camera (usually index 0)
cap = cv2.VideoCapture(0)

# Variables to hold the latest frame
frame = None
lock = threading.Lock()

def capture_frames():
    global frame
    while cap.isOpened():
        ret, new_frame = cap.read()
        if ret:
            with lock:
                frame = new_frame

# Start a thread to capture frames from the camera
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Loop through the video frames
while True:
    with lock:
        if frame is not None:
            # Run YOLOv8 inference on the frame
            results = model(frame)
            annotated_frame = results[0].plot()  # Annotate the frame with detections
            
            # Display the annotated frame
            cv2.imshow("YOLOv10 Tracking", annotated_frame)
        
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the camera capture object and close the display window
cap.release()
cv2.destroyAllWindows()
