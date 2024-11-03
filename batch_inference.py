import os
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

# Define paths and server details
input_dir = "/Users/jamesdowney/Documents/Project_Data/Cleaned_and_Labelled/TSS/Generated Segmentations/batch/path_batch_input_path"
output_dir = "/Users/jamesdowney/Documents/Project_Data/Cleaned_and_Labelled/TSS/Generated Segmentations/batch/flat_b1000_batch_output2"
server_url = "http://54.206.132.176:8003/infer/segmentation"
model_name = "segmentation"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to strip multipart headers
def strip_multipart_headers(response_content):
    boundary = response_content.split(b'\r\n')[0]
    parts = response_content.split(boundary)
    for part in parts:
        if b'Content-Type: application/octet-stream' in part:
            return part.split(b'\r\n\r\n', 1)[1].rsplit(b'\r\n', 1)[0]
    return None

# Set up retry strategy
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"]  # Updated argument name
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

try:
    # Iterate over images and send inference requests
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        with open(img_path, 'rb') as file:
            files = {'file': file}
            data = {'model': model_name}

            try:
                with http.post(server_url, files=files, data=data) as response:
                    response.raise_for_status()
                    stripped_content = strip_multipart_headers(response.content)
                    if stripped_content:
                        output_path = os.path.join(output_dir, img_name)
                        with open(output_path, 'wb') as f:
                            f.write(stripped_content)
                            logging.info(f"{img_name} saved to output directory.")
                    else:
                        logging.error(f"Failed to strip headers for {img_name}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Failed to process {img_name}: {e}")
except Exception as e:
    logging.critical(f"An error occurred: {e}")
finally:
    http.close()

logging.info("Batch inference completed.")
