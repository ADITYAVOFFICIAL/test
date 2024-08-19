from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

# Load the YOLO models
movement_model = YOLO("yolov10b.pt")
plate_model = YOLO("test.pt")

# Open the video file
video_path = "test.mp4"
cap = cv2.VideoCapture(video_path)

# Get video writer setup for exporting the output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
output_path = "output.mp4"
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Store the track history
track_history = defaultdict(lambda: [])

# Define movement threshold (you can adjust this value)
movement_threshold = 5  # pixels

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv10b tracking on the frame, persisting tracks between frames
        results = movement_model.track(frame, persist=True, device='mps')

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            if len(track) > 0:
                # Calculate the movement
                last_x, last_y = track[-1]
                movement = np.sqrt((x - last_x)**2 + (y - last_y)**2)

                # Check if the car is moving or stopped
                if movement < movement_threshold:
                    status = "Stopped"
                    color = (0, 0, 255)  # Red color for stopped
                else:
                    status = "Moving"
                    color = (0, 255, 0)  # Green color for moving

                # Run number plate detection on the vehicle
                x1 = int(max(x - w/2, 0))
                y1 = int(max(y - h/2, 0))
                x2 = int(min(x + w/2, frame.shape[1]))
                y2 = int(min(y + h/2, frame.shape[0]))
                car_img = frame[y1:y2, x1:x2]

                if car_img.size > 0:  # Ensure the crop is valid
                    plate_results = plate_model(car_img, conf=0.25)  # Adjust the confidence threshold

                    # Plot the number plate detection results
                    for plate_box, conf in zip(plate_results[0].boxes.xywh.cpu(), plate_results[0].boxes.conf.cpu()):
                        px, py, pw, ph = plate_box
                        label = f"Plate: {conf:.2f}"
                        cv2.rectangle(annotated_frame,
                                      (x1 + int(px - pw / 2), y1 + int(py - ph / 2)),
                                      (x1 + int(px + pw / 2), y1 + int(py + ph / 2)),
                                      (255, 0, 0), 2)  # Blue color for plate detection
                        cv2.putText(annotated_frame, label, (x1 + int(px - pw / 2), y1 + int(py - ph / 2) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Display status on the frame
                cv2.putText(annotated_frame, status, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update the track history
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 tracks for recent frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

        # Write the frame to the output video
        out.write(annotated_frame)

        # Display the annotated frame
        cv2.imshow("YOLOv10b Tracking and Plate Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object, writer, and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()
