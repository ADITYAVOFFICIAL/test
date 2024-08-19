from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from IPython.display import display

# Load OCR models
handwritten_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
handwritten_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")

printed_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
printed_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')

# Function to visualize images
def show_image(pathStr):
    img = Image.open(pathStr).convert("RGB")
    display(img)
    return img

# Function for OCR on handwritten text
def ocr_handwritten_image(src_img):
    pixel_values = handwritten_processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = handwritten_model.generate(pixel_values)
    return handwritten_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Function for OCR on printed text
def ocr_printed_image(src_img):
    pixel_values = printed_processor(images=src_img, return_tensors="pt").pixel_values
    generated_ids = printed_model.generate(pixel_values)
    return printed_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Example usage for handwritten image
hw_image = show_image('./test_images/product.jpeg')
hw_image_cropped = hw_image.crop((0, 10, hw_image.size[0], 40))  # Adjust cropping as needed
display(hw_image_cropped)
handwritten_text = ocr_handwritten_image(hw_image_cropped)
print(f"Handwritten OCR Result: {handwritten_text}")

# Example usage for printed image
invoice_image = show_image('./test_images/product.jpeg')
printed_text = ocr_printed_image(invoice_image)
print(f"Printed OCR Result: {printed_text}")
