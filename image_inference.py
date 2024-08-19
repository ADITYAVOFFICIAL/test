import os
from ultralytics import YOLO
import cv2

# Load the pre-trained YOLO model
model = YOLO("latest.pt")  # Replace with the appropriate YOLO version if needed

# Define the input and output directories
input_dir = './test_images'
output_dir = './res'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through all files in the input directory
for file_name in os.listdir(input_dir):
    # Check if the file is an image
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Construct the full file path
        file_path = os.path.join(input_dir, file_name)
        
        # Run inference on the image
        results = model(file_path)
        
        # Extract bounding boxes, labels, and scores
        boxes = results[0].boxes
        img = cv2.imread(file_path)
        
        # Iterate through detected objects
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract label and score
            label = box.cls
            score = box.conf
            
            # Convert score to percentage and format label
            score_percentage = int(score * 100)
            label_text = f"{model.names[int(label)]}: {score_percentage}%"
            
            # Draw the bounding box on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Display label and score above the bounding box with increased text size
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Save the annotated image to the output directory
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, img)
        
        print(f'Processed {file_name}')

print('Inference and saving complete.')
