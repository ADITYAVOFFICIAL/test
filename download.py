import requests
import os
import pandas as pd

# Function to download images
def download_images(image_links, save_dir):
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for index, url in enumerate(image_links):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image_path = os.path.join(save_dir, f'image_{index+1}.jpg')
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded: {image_path}")
            else:
                print(f"Failed to download image {index+1}: {url}")
        except Exception as e:
            print(f"Error downloading image {index+1}: {url} - {e}")

# Load test data
test_df = pd.read_csv('./dataset/test.csv')

# Download images
download_images(test_df['image_link'], 'images')
