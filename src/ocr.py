import easyocr
import pandas as pd
import os
import re
import argparse
from utils import download_images, parse_string
import constants

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Define the allowed units and entity unit mapping
entity_unit_map = {
    'width': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'depth': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'height': {'centimetre', 'foot', 'inch', 'metre', 'millimetre', 'yard'},
    'item_weight': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'maximum_weight_recommendation': {'gram', 'kilogram', 'microgram', 'milligram', 'ounce', 'pound', 'ton'},
    'voltage': {'kilovolt', 'millivolt', 'volt'},
    'wattage': {'kilowatt', 'watt'},
    'item_volume': {'centilitre', 'cubic foot', 'cubic inch', 'cup', 'decilitre', 'fluid ounce', 'gallon', 'imperial gallon', 'litre', 'microlitre', 'millilitre', 'pint', 'quart'}
}

allowed_units = {unit for entity in entity_unit_map for unit in entity_unit_map[entity]}

def extract_text_from_image(image_path):
    result = reader.readtext(image_path)
    text = ' '.join([res[1] for res in result])
    return text

def parse_entity_value(text, entity_name):
    # Add your parsing logic here
    entity_units = entity_unit_map.get(entity_name, set())
    for unit in entity_units:
        if unit in text:
            value = re.search(r'\d+\.?\d*', text)
            if value:
                return f"{value.group()} {unit}"
    return ""

def sanity_check(test_filename, output_filename):
    check_file(test_filename)
    check_file(output_filename)
    
    try:
        test_df = pd.read_csv(test_filename)
        output_df = pd.read_csv(output_filename)
    except Exception as e:
        raise ValueError(f"Error reading the CSV files: {e}")
    
    if 'index' not in test_df.columns:
        raise ValueError("Test CSV file must contain the 'index' column.")
    
    if 'index' not in output_df.columns or 'prediction' not in output_df.columns:
        raise ValueError("Output CSV file must contain 'index' and 'prediction' columns.")
    
    missing_index = set(test_df['index']).difference(set(output_df['index']))
    if len(missing_index) != 0:
        print("Missing index in test file: {}".format(missing_index))
        
    extra_index = set(output_df['index']).difference(set(test_df['index']))
    if len(extra_index) != 0:
        print("Extra index in test file: {}".format(extra_index))
        
    output_df.apply(lambda x: parse_string(x['prediction']), axis=1)
    print("Parsing successful for file: {}".format(output_filename))

def check_file(filename):
    if not filename.lower().endswith('.csv'):
        raise ValueError("Only CSV files are allowed.")
    if not os.path.exists(filename):
        raise FileNotFoundError("Filepath: {} invalid or not found.".format(filename))

def main():
    # Define the download folder
    download_folder = 'images'
    os.makedirs(download_folder, exist_ok=True)

    # Load test data
    test_df = pd.read_csv('dataset/test.csv')
    
    # Create output dataframe
    output_df = pd.DataFrame(columns=['index', 'prediction'])
    
    for _, row in test_df.iterrows():
        image_url = row['image_link']
        image_path = download_images(image_url, download_folder)  # Ensure this function downloads and returns the image path
        text = extract_text_from_image(image_path)
        prediction = parse_entity_value(text, row['entity_name'])
        output_df = output_df.append({'index': row['index'], 'prediction': prediction}, ignore_index=True)
    
    # Save output to CSV
    output_df.to_csv('out.csv', index=False)
    
    # Perform sanity check
    sanity_check('dataset/test.csv', 'out.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR Extraction and Formatting")
    parser.add_argument("--test_filename", type=str, default='dataset/test.csv', help="The test CSV file name.")
    parser.add_argument("--output_filename", type=str, default='out.csv', help="The output CSV file name.")
    args = parser.parse_args()
    
    # Update filenames if provided through command line arguments
    test_filename = args.test_filename
    output_filename = args.output_filename
    
    # Run the main function
    main()
