import os
import csv
import numpy as np
import nibabel as nib
from sklearn.metrics import mean_absolute_error
from scipy.spatial.distance import directed_hausdorff
from concurrent.futures import ProcessPoolExecutor
from numba import jit

@jit(nopython=True)
def calculate_iou_dice_pixel_accuracy(ground_truth_flat, prediction_flat):
    intersection = np.logical_and(ground_truth_flat, prediction_flat)
    union = np.logical_or(ground_truth_flat, prediction_flat)
    iou = np.sum(intersection) / np.sum(union)
    
    dice = 2 * np.sum(intersection) / (np.sum(ground_truth_flat) + np.sum(prediction_flat))
    
    pixel_accuracy = np.sum(ground_truth_flat == prediction_flat) / len(ground_truth_flat)
    
    return iou, dice, pixel_accuracy

def calculate_hausdorff_3d(ground_truth, prediction):
    max_distance = 0
    for i in range(ground_truth.shape[0]):
        gt_slice = ground_truth[i, :, :]
        pred_slice = prediction[i, :, :]
        distance = max(directed_hausdorff(gt_slice, pred_slice)[0], directed_hausdorff(pred_slice, gt_slice)[0])
        if distance > max_distance:
            max_distance = distance
    return max_distance

def calculate_precision_recall_f1(ground_truth, prediction):
    true_positive = np.sum((ground_truth == 1) & (prediction == 1))
    false_positive = np.sum((ground_truth == 0) & (prediction == 1))
    false_negative = np.sum((ground_truth == 1) & (prediction == 0))
    
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def calculate_metrics(ground_truth, prediction):
    ground_truth_flat = ground_truth.flatten()
    prediction_flat = prediction.flatten()
    
    iou, dice, pixel_accuracy = calculate_iou_dice_pixel_accuracy(ground_truth_flat, prediction_flat)
    
    precision, recall, f1 = calculate_precision_recall_f1(ground_truth_flat, prediction_flat)
    
    mae = mean_absolute_error(ground_truth_flat, prediction_flat)
    
    hausdorff_distance = calculate_hausdorff_3d(ground_truth, prediction)
    
    return iou, dice, pixel_accuracy, precision, recall, f1, mae, hausdorff_distance

def calculate_metrics_per_segmentation(ground_truth, prediction, gt_path):
    unique_labels = np.unique(ground_truth)
    metrics_per_segmentation = {}
    
    for label in unique_labels:
        if label == 0:
            continue
        gt_segmentation = (ground_truth == label).astype(int)
        pred_segmentation = (prediction == label).astype(int)
        
        try:
            metrics = calculate_metrics(gt_segmentation, pred_segmentation)
            gt_voxel_count = np.count_nonzero(gt_segmentation)
            pred_voxel_count = np.count_nonzero(pred_segmentation)
            
            # Calculate voxel size and segment volume
            voxel_size = np.prod(nib.load(gt_path).header.get_zooms()) / 1000
            gt_segment_volume = voxel_size * gt_voxel_count
            pred_segment_volume = voxel_size * pred_voxel_count
            
            metrics_per_segmentation[label] = metrics + (gt_voxel_count, pred_voxel_count, voxel_size, gt_segment_volume, pred_segment_volume)
            print(f"Processed label {label} for file")
        except Exception as e:
            print(f"Skipping label {label} due to error: {e}")
            continue
    
    combined_gt = np.logical_or(ground_truth == 1, ground_truth == 2).astype(int)
    combined_pred = np.logical_or(prediction == 1, prediction == 2).astype(int)
    
    combined_metrics = calculate_metrics(combined_gt, combined_pred)
    combined_gt_voxel_count = np.count_nonzero(combined_gt)
    combined_pred_voxel_count = np.count_nonzero(combined_pred)
    
    # Calculate voxel size and segment volume for combined segments
    voxel_size = np.prod(nib.load(gt_path).header.get_zooms()) / 1000
    combined_gt_segment_volume = voxel_size * combined_gt_voxel_count
    combined_pred_segment_volume = voxel_size * combined_pred_voxel_count
    
    metrics_per_segmentation['combined_1_2'] = combined_metrics + (combined_gt_voxel_count, combined_pred_voxel_count, voxel_size, combined_gt_segment_volume, combined_pred_segment_volume)
    
    return metrics_per_segmentation

def process_file(gt_file, ground_truth_dir, inference_dir):
    base_name = os.path.splitext(gt_file)[0]
    gt_path = os.path.join(ground_truth_dir, gt_file)
    pred_path = os.path.join(inference_dir, gt_file)
    
    if os.path.exists(pred_path):
        try:
            print(f"Processing file: {gt_path}")  # Print statement to identify the file
            gt_data = nib.load(gt_path).get_fdata()
            pred_data = nib.load(pred_path).get_fdata()
            
            metrics_per_segmentation = calculate_metrics_per_segmentation(gt_data, pred_data, gt_path)
            
            results = []
            for label, metrics in metrics_per_segmentation.items():
                results.append([base_name, label] + list(metrics))
            return results
        except Exception as e:
            print(f"Error processing file {gt_path}: {e}")
    return []

def main(ground_truth_dir, inference_dir, output_csv):
    ground_truth_files = [f for f in os.listdir(ground_truth_dir) if f.endswith('.nii.gz')]
    total_files = len(ground_truth_files)
    
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Case', 'Segmentation Label', 'IoU (Jaccard Index)', 'Dice', 'Pixel Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean Absolute Error', 'Hausdorff Distance', 'Ground Truth Voxel Count', 'Inference Voxel Count', 'GT Voxel Size (cm^3)', 'Ground Truth Segment Volume (cm^3)', 'Inference Segment Volume (cm^3)'])
        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(process_file, gt_file, ground_truth_dir, inference_dir) for gt_file in ground_truth_files]
            for i, future in enumerate(futures):
                results = future.result()
                for result in results:
                    writer.writerow(result)
                print(f"Processed {i + 1}/{total_files} files")

if __name__ == "__main__":
    main('', '/Users/jamesdowney/Documents/Project_Data/Cleaned_and_Labelled/TSS/Generated Segmentations/batch/completed/extras/flat_anatomy_batch_output_delete_and_remap_with_extras', '/Users/jamesdowney/Library/CloudStorage/OneDrive-Personal/UNI/UNSW/Phase 2/Honours/Stats/rerun_path/anatomy_flat.csv')


###main: 'GT dir', 'inference dir', 'output.csv')