import os
import nibabel as nib
import csv

# Define the directory containing the NIFTI files
nifti_dir = "F:/Registration Code/CTA_Kevin_Training_16Septv2andNormsComplete/NEW/CroppedNCCTAnatomy/CroppedNCCTAnatomy/output/labels/final"

# Define the path for the output CSV file
output_csv = os.path.join(nifti_dir, "nifti_headers_with_original_pixdim.csv")

# List of header fields to extract from NIFTI files
header_fields = [
    "sizeof_hdr", "dim_info", "dim", "intent_p1", "intent_p2", "intent_p3", "intent_code",
    "datatype", "bitpix", "slice_start", "pixdim", "vox_offset", "scl_slope", "scl_inter",
    "slice_end", "slice_code", "xyzt_units", "cal_max", "cal_min", "slice_duration",
    "toffset", "glmax", "glmin"
]

# Open the CSV file for writing
with open(output_csv, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row (filename + header fields + original pixdim)
    writer.writerow(["Filename"] + header_fields + ["Original_pixdim"])

    # Loop through all files in the directory
    for filename in os.listdir(nifti_dir):
        # Check if the file is a NIFTI file (.nii or .nii.gz)
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            # Construct the full path to the file
            file_path = os.path.join(nifti_dir, filename)
            
            # Load the NIFTI file
            try:
                nii = nib.load(file_path)
                header = nii.header
                
                # Extract header values based on the specified fields
                header_values = [header.get(field, "N/A") for field in header_fields]

                # Extract the original pixdim values before any modifications
                original_pixdim = header['pixdim']

                # Write the filename, header values, and original pixdim to the CSV
                writer.writerow([filename] + header_values + [original_pixdim.tolist()])
            except Exception as e:
                # In case of error, write the filename and the error message
                writer.writerow([filename] + [f"Error: {e}"] * len(header_fields) + ["Error"])

print(f"NIFTI headers and original pixdim values have been saved to {output_csv}")
