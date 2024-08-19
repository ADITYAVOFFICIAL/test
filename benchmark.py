import cv2
from ultralytics import YOLO
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import os

# Load the model
model = YOLO('test.pt')

# Define the path to the test dataset
test_images_path = './train/images/'
test_labels_path = './train/labels/'

# Function to load ground truth labels
def load_ground_truth_labels(label_path):
    labels = {}
    for label_file in os.listdir(label_path):
        with open(os.path.join(label_path, label_file)) as f:
            labels[label_file] = [list(map(float, line.strip().split())) for line in f.readlines()]
    return labels

# Load ground truth labels
ground_truth_labels = load_ground_truth_labels(test_labels_path)

# Initialize performance metrics
y_true = []
y_scores = []
image_ids = []

# Run inference and collect predictions
for image_file in os.listdir(test_images_path):
    image_path = os.path.join(test_images_path, image_file)
    image = cv2.imread(image_path)

    if image is None:
        continue

    # Run inference
    results = model(image)

    # Collect true and predicted labels
    pred_boxes = results[0].boxes.xyxy.cpu().numpy()
    pred_scores = results[0].boxes.conf.cpu().numpy()
    true_boxes = np.array(ground_truth_labels.get(image_file, []))

    y_true.extend([1] * len(true_boxes))  # Assuming all true boxes are positive
    y_scores.extend(pred_scores.tolist())

    # Optional: Calculate precision, recall, and mAP here if ground truth data is in COCO format

# Calculate precision, recall, and average precision
precision, recall, _ = precision_recall_curve(y_true, y_scores)
average_precision = average_precision_score(y_true, y_scores)

print(f'Average Precision: {average_precision:.4f}')
print(f'Precision: {precision[-1]:.4f}')
print(f'Recall: {recall[-1]:.4f}')
