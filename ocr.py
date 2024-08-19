import os
from ultralytics import YOLO
import cv2

# Load the pre-trained YOLO model for number plate character detection
model = YOLO("number.pt")  # Replace with the appropriate YOLO model

# Define the input directory
input_dir = './number_plate_images'

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
        
        # List to store detected characters and their x-coordinate
        characters = []
        
        # Iterate through detected objects
        for box in boxes:
            # Extract the bounding box coordinates and label
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls)]
            
            # Store the character and its x-coordinate (for sorting)
            characters.append((label.upper(), x1))
        
        # Sort characters based on the x-coordinate (left to right)
        characters.sort(key=lambda x: x[1])
        
        # Extract the sorted characters and concatenate them
        plate_text = ''.join([char[0] for char in characters])
        
        # Print the detected number plate text in the terminal
        print(f'Detected number plate text for {file_name}: {plate_text}')
        
        # Display the detected text on the image
        cv2.putText(img, plate_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        
        # Show the image with detected number plate text in a window
        cv2.imshow(f'Detected Number Plate - {file_name}', img)
        
        # Wait until a key is pressed, then close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()

print('Number plate detection, extraction, and display complete.')
