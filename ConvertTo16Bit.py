import nibabel as nib
import numpy as np
import os

# Function to convert image data type to 16-bit integer
def convert_to_int16(image_path, output_path):
    # Load the image
    img = nib.load(image_path)
    
    # Get the image data
    img_data = img.get_fdata()
    
    # Convert the data to 16-bit integer (int16)
    img_data_int16 = img_data.astype(np.int16)
    
    # Save the modified image with int16 data
    modified_img = nib.Nifti1Image(img_data_int16, img.affine, img.header)
    # Update the data type in the header to reflect the new data type
    modified_img.header.set_data_dtype(np.int16)
    
    # Save the new image
    nib.save(modified_img, output_path)
    print(f"Converted to int16 and saved image to {output_path}")

# Main function to process all files in a directory
def process_directory(input_directory, output_directory):
    # Check if output directory exists, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Loop through all .nii and .nii.gz files in the input directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            image_path = os.path.join(input_directory, filename)
            output_path = os.path.join(output_directory, filename)
            print(f"Processing {filename}...")
            convert_to_int16(image_path, output_path)

# Specify input and output directories
input_directory = 'F:/Registration Code/CroppedNCCTAnatomy/labels/final/final_Labels/New folder'  # Replace with your input directory path
output_directory = 'F:/Registration Code/CroppedNCCTAnatomy/labels/final/final_Labels/New folder'  # Replace with your output directory path

# Call the process directory function to convert all images to int16
process_directory(input_directory, output_directory)
