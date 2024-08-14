import cv2
from ultralytics import YOLO
import easyocr
import numpy as np

# Load the YOLO models
vehicle_model = YOLO('yolov10b.pt')  # Model for detecting vehicles
plate_model = YOLO('test.pt')        # Model for detecting number plates

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Open the video file
video_path = './test_videos/1.mp4'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize variables
batch_size = 16
frames = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frames.append(frame)
    frame_count += 1

    # Process in batches
    if len(frames) == batch_size:
        # Run inference on the batch to detect vehicles
        vehicle_results = vehicle_model(frames)

        for i, vehicle_result in enumerate(vehicle_results):
            # Filter results to detect only vehicles
            vehicle_detected = False
            for bbox in vehicle_result.boxes:
                # Assuming vehicle class ID is 2 (this needs to be verified with the actual class ID in the model)
                if bbox.cls == 2:  # Update with the correct class ID for vehicle
                    vehicle_detected = True
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())  # Convert tensor to list

                    # Annotate the detected vehicle
                    cv2.rectangle(frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Extract the region of interest (ROI) for number plate detection
                    roi = frames[i][y1:y2, x1:x2]

                    # Run inference on the ROI to detect number plates
                    plate_results = plate_model([roi])

                    for plate_result in plate_results:
                        for plate_bbox in plate_result.boxes:
                            px1, py1, px2, py2 = map(int, plate_bbox.xyxy[0].tolist())
                            
                            # Annotate the detected number plate
                            cv2.rectangle(roi, (px1, py1), (px2, py2), (0, 0, 255), 2)
                            
                            # Extract the number plate text using EasyOCR
                            plate_roi = roi[py1:py2, px1:px2]
                            result = reader.readtext(plate_roi)

                            # Annotate the frame with the detected number plate text
                            if result:
                                number_plate_text = result[0][-2]
                                cv2.putText(roi, number_plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Write the annotated frame to the output video only if a vehicle is detected
            if vehicle_detected:
                out.write(frames[i])

        # Clear frames list for the next batch
        frames = []

# Process remaining frames
if frames:
    vehicle_results = vehicle_model(frames)
    for i, vehicle_result in enumerate(vehicle_results):
        vehicle_detected = False
        for bbox in vehicle_result.boxes:
            if bbox.cls == 2:  # Update with the correct class ID for vehicle
                vehicle_detected = True
                x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())
                cv2.rectangle(frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi = frames[i][y1:y2, x1:x2]

                plate_results = plate_model([roi])
                for plate_result in plate_results:
                    for plate_bbox in plate_result.boxes:
                        px1, py1, px2, py2 = map(int, plate_bbox.xyxy[0].tolist())
                        cv2.rectangle(roi, (px1, py1), (px2, py2), (0, 0, 255), 2)
                        plate_roi = roi[py1:py2, px1:px2]
                        result = reader.readtext(plate_roi)
                        if result:
                            number_plate_text = result[0][-2]
                            cv2.putText(roi, number_plate_text, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        if vehicle_detected:
            out.write(frames[i])

# Release the video capture and writer objects
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Inference completed. Output video saved to {output_path}")
