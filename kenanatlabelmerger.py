import SimpleITK as sitk
import os
import numpy as np
import glob
import csv

# head_segment_predictions are JCai's prediction labels (anatomy)
# ich_segmentations are ground truth ischaemic infarct labels


# Define input and output paths
head_segment_predictions_folder = "D:/Matlab Registration Code/NCCT_Anatomy20SeptInt/NCCT_Anatomy20SeptInt/Cropped/labelmerger/UnmergedAnat"
ich_segmentations_folder = "D:/Matlab Registration Code/NCCT_Anatomy20SeptInt/NCCT_Anatomy20SeptInt/Cropped/labelmerger/Ischemic"
output_folder = "D:/Matlab Registration Code/NCCT_Anatomy20SeptInt/NCCT_Anatomy20SeptInt/Cropped/labelmerger"
log_file_path = os.path.join(
    output_folder,
    "D:/Matlab Registration Code/NCCT_Anatomy20SeptInt/NCCT_Anatomy20SeptInt/Cropped/labelmerger/segmentation_log.csv",
)

# Ensure the output directory exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define the mapping from segment names to unique segment IDs for the first segmentation file
segment_dict1 = {
    "Segment_1": 1,
    "Segment_2": 2,
    "Segment_3": 3,
    "Segment_4": 4,
    "Segment_5": 5,
    "Segment_6": 6,
    "Segment_7": 7,
    "Segment_8": 8,
    "Segment_9": 9,
    "Segment_10": 10,
    "Segment_11": 2,  # Merge with Segment_2
    "Segment_12": 11,
    "Segment_13": 11,  # Merge with Segment_12 (now 11 to avoid higher label numbers than number of segments for segresnet)
    "Segment_14": 11,  # Merge with Segment_12
    "Segment_15": 11,  # Merge with Segment_12
    "Segment_16": 12,
}

# Define the mapping from segment names to unique segment IDs for the second segmentation file (edited for single label ischaemic infarct)
segment_dict2 = {
    "Segment_1": 13,
    "Segment_2": 102,
    "Segment_3": 103,
    "Segment_4": 104,
}

# Define the mapping from segment names to semantic names
semantic_dict = {
    "Segment_1": "Brain Parenchyma NOS",
    "Segment_2": "Subarachnoid Space",
    "Segment_3": "Dural Folds/Venous Sinuses",
    "Segment_4": "Septum Pellucidum",
    "Segment_5": "Cerebellum",
    "Segment_6": "Caudate Nucleus",
    "Segment_7": "Lentiform Nucleus",
    "Segment_8": "Insular Cortex",
    "Segment_9": "Internal Capsule",
    "Segment_10": "Ventricular System",
    "Segment_11": "Cerebrum",
    "Segment_11": "Cerebrum",
    "Segment_11": "Cerebrum",
    "Segment_11": "Cerebrum",
    "Segment_12": "Thalamus",
    "Segment_13": "Ischaemic_Infarct",
}

# Old ICH labels:
#    "Segment_101": "Intracerebral Haemorrhage",
#    "Segment_102": "Intraventricular Haemorrhage",
#    "Segment_103": "Subarachnoid Haemorrhage",
#    "Segment_104": "Acute Subdural Haemorrhage",

# Open the CSV file to write the log
with open(log_file_path, "w", newline="") as log_file:
    log_writer = csv.writer(log_file)
    # Write the header row
    log_writer.writerow(
        [
            "HeadCT_Segmentation",
            "ICH_Segmentation",
            "Combined_Segmentation",
            "HeadCT_Dimensions",
            "HeadCT_Origin",
            "ICH_Dimensions",
            "ICH_Origin",
            "Combined_Dimensions",
            "Combined_Origin",
        ]
    )

    # Helper function to get the root filename without extension
    def get_root_filename(filename):
        base_name = os.path.basename(filename)
        if base_name.endswith(".nii.gz"):
            return base_name[:-7]  # Remove the .nii.gz extension correctly
        elif base_name.endswith(".nrrd"):
            return base_name[:-5]  # Remove the .nrrd extension
        else:
            return os.path.splitext(base_name)[0]  # General case

    # Get the list of segmentation files
    head_segment_filenames = sorted(
        glob.glob(os.path.join(head_segment_predictions_folder, "*.nii.gz"))
    )
    ich_segment_filenames = sorted(
        glob.glob(os.path.join(ich_segmentations_folder, "*.nii.gz"))
        + glob.glob(os.path.join(ich_segmentations_folder, "*.nrrd"))
    )

    # Loop over each pair of segmentation files
    for head_segment_filename in head_segment_filenames:
        head_filename_root = get_root_filename(head_segment_filename)
        ich_segment_filename = next(
            (
                f
                for f in ich_segment_filenames
                if get_root_filename(f) == head_filename_root
            ),
            None,
        )

        if ich_segment_filename is None:
            print(
                f"Could not find matching ICH segmentation file for {head_filename_root}"
            )
            log_writer.writerow(
                [
                    os.path.basename(head_segment_filename),
                    "NOT FOUND",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                    "N/A",
                ]
            )
            continue

        # Load the segmentation files
        head_seg = sitk.ReadImage(head_segment_filename)
        ich_seg = sitk.ReadImage(ich_segment_filename)

        # Convert to NumPy arrays
        head_seg_array = sitk.GetArrayFromImage(head_seg)
        ich_seg_array = sitk.GetArrayFromImage(ich_seg)

        # Perform the necessary segment merging and renumbering for the first segmentation file
        for old, new in segment_dict1.items():
            head_seg_array[head_seg_array == int(old.split("_")[1])] = new

        # Perform the necessary segment merging and renumbering for the second segmentation file
        for old, new in segment_dict2.items():
            ich_seg_array[ich_seg_array == int(old.split("_")[1])] = new

        # Combine the two segmentation arrays
        combined_seg_array = np.maximum(head_seg_array, ich_seg_array)

        # Save the combined segmentation as a new NRRD file
        combined_seg = sitk.GetImageFromArray(combined_seg_array)
        combined_seg.CopyInformation(
            head_seg
        )  # Copy spatial information from the original image

        # Save the combined segmentation as a new NRRD file
        output_filename = os.path.join(output_folder, head_filename_root + ".nrrd")
        sitk.WriteImage(combined_seg, output_filename)
        print(output_filename, "saved")

        # Log the details
        log_writer.writerow(
            [
                os.path.basename(head_segment_filename),
                os.path.basename(ich_segment_filename),
                os.path.basename(output_filename),
                str(head_seg.GetSize()),
                str(head_seg.GetOrigin()),
                str(ich_seg.GetSize()),
                str(ich_seg.GetOrigin()),
                str(combined_seg.GetSize()),
                str(combined_seg.GetOrigin()),
            ]
        )