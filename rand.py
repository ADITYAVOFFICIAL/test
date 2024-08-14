import cv2
import numpy as np
import easyocr

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Increase contrast
    image = cv2.equalizeHist(image)

    # Binarize the image
    _, binarized_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((1, 1), np.uint8)
    cleaned_image = cv2.morphologyEx(binarized_image, cv2.MORPH_CLOSE, kernel)

    # Deskewing
    coords = np.column_stack(np.where(cleaned_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    center = tuple(np.array(cleaned_image.shape[1::-1]) / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed_image = cv2.warpAffine(cleaned_image, M, cleaned_image.shape[1::-1], flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed_image

def perform_ocr(image_path):
    preprocessed_image = preprocess_image(image_path)
    
    # Save preprocessed image (for debugging)
    cv2.imwrite('preprocessed_image.png', preprocessed_image)
    
    # Perform OCR
    results = reader.readtext(preprocessed_image)
    
    return results

# Example usage
image_path = './test_images/a.jpg'
results = perform_ocr(image_path)
print("OCR Results:", results)
