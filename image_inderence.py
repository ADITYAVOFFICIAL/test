import os
from ultralytics import YOLO
import easyocr
import cv2

# Load the pre-trained YOLO model
model = YOLO("test.pt")  # Replace with the appropriate YOLO version if needed

# Define the input and output directories
input_dir = './test_images'
output_dir = './res'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Specify the languages you want to support

# Iterate through all files in the input directory
for file_name in os.listdir(input_dir):
    # Check if the file is an image
    if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        # Construct the full file path
        file_path = os.path.join(input_dir, file_name)
        
        # Run inference on the image
        results = model(file_path)
        
        # Extract bounding boxes and labels
        boxes = results[0].boxes
        img = cv2.imread(file_path)
        
        # Iterate through detected objects
        for box in boxes:
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Crop the detected region
            cropped_img = img[y1:y2, x1:x2]
            
            # Use EasyOCR to extract text from the cropped region
            text = reader.readtext(cropped_img, detail=0)
            extracted_text = " ".join(text)
            
            # Draw the bounding box and extracted text on the image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, extracted_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save the annotated image to the output directory
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, img)
        
        print(f'Processed {file_name} with text extraction')

print('Inference, text extraction, and saving complete.')
