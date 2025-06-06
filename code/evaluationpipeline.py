import numpy as np
import torch
import cv2
from tqdm import tqdm
from egodexter import EgoDexter
import track_utils
import solvertorch
from zimeval import EvalUtil  # Make sure to import EvalUtil
import pandas as pd
import os


def image_to_camera_coords(pred_2d, intrinsics):
    """
    Convert 2D image coordinates to 3D camera coordinates using the intrinsic matrix.
    Assumes pred_2d is of shape (N, 3) where Z is already estimated depth.
    """
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    cam_coords = np.zeros_like(pred_2d)
    cam_coords[:, 0] = (pred_2d[:, 0] - cx) * pred_2d[:, 2] / fx  # X in camera space
    cam_coords[:, 1] = (pred_2d[:, 1] - cy) * pred_2d[:, 2] / fy  # Y in camera space
    cam_coords[:, 2] = pred_2d[:, 2]  # Z remains unchanged (depth)
    
    return cam_coords

def process_and_evaluate_images_from_egodexter():
    # Initialize the evaluation utility
    eval_util = EvalUtil(num_kp=5)  # Only 5 joints now (fingertips)

    # Set up the EgoDexter dataset
    ed = EgoDexter(
        data_root="C:/Users/chris/Documents/GitHub/HandRepo/data",
        data_split='test',
        hand_side='left',
        njoints=21,
        use_cache=False,
        vis=False
    )

    valid_samples = 0  # Counter for valid samples

    # Loop through the dataset and process each image
    for id in tqdm(range(len(ed))):
    # for id in tqdm(range(5)):
        print(f"Processing image {id}...")
        data = ed[id]

        # Get the image, ground truth (3D joint annotations), and intrinsics
        clr_image = data['clr']  # This is a PIL Image
        gt_landmarks = np.array(data['tips'])  # 3D ground truth coordinates (fingertips)
        intrinsics = ed.color_intrisics  # Intrinsics matrix from EgoDexter


        # Convert PIL image to NumPy array
        clr_image = np.array(clr_image)  # Now clr_image is in HWC format (Height, Width, Channels)

        # Ensure image is RGB (MediaPipe expects RGB)
        # clr_image = cv2.cvtColor(clr_image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB (if needed)

        # Extract hand landmarks using track_landmarks_from_img
        results = track_utils.track_landmarks_from_img(clr_image)

        if not results.hand_landmarks:
            print(f"No hand landmarks found for image {id}")
            continue
        # If we have valid landmarks, process them
        # Extract 2D landmarks and convert to real-world 3D space
        x_real, y_real, z_real, h, w, IsLeft = track_utils.get_real_landmarks(clr_image, results)
        real_landmarks = np.column_stack((x_real, y_real, z_real))

        # Fit a 3D model using solver (gets result_keypoints in image space)
        Output, init_pose, init_shape, result_keypoints = solvertorch.solveminimize2stages(real_landmarks, clr_image, IsLeft=True)
        
        # Select only fingertip keypoints
        fingertip_indices = [4, 8, 12, 16, 20]
        result_fingertips = result_keypoints[fingertip_indices, :]

         # Identify valid (non-NaN) indices
        valid_indices = ~np.isnan(gt_landmarks[:, 0]) & \
                        ~np.isnan(gt_landmarks[:, 1]) & \
                        ~np.isnan(gt_landmarks[:, 2]) & \
                        ~np.isnan(result_fingertips[:, 0]) & \
                        ~np.isnan(result_fingertips[:, 1]) & \
                        ~np.isnan(result_fingertips[:, 2])

        # Select only valid keypoints
        gt_landmarks_valid = gt_landmarks[valid_indices].reshape(-1, 3)
        result_fingertips_valid = result_fingertips[valid_indices].reshape(-1, 3)

        # Align Z-axis using centroid translation
        gt_centroid_z = np.mean(gt_landmarks_valid[:, 2])  #Mean Z of GT Ignore NaNs in GT
        pred_centroid_z = np.mean(result_fingertips_valid[:, 2])  #Mean Z of predictions Ignore NaNs in predictions

        # Check if all Z-values are NaN
        if np.isnan(gt_centroid_z) or np.isnan(pred_centroid_z):
            print(f"Skipping image {id} due to all NaN values in Z coordinates.")
            continue  # Skip this image

        translation_z = gt_centroid_z - pred_centroid_z
        result_fingertips_valid[:, 2] = result_fingertips_valid[:, 2] +  translation_z  # Adjust only Z values
        
        # Convert image coordinates to camera coordinates
        result_fingertips_cam = image_to_camera_coords(result_fingertips_valid, intrinsics)

        # Ensure we have at least one valid keypoint
        if gt_landmarks_valid.shape[0] == 0:
            print(f"Skipping image {id} due to all NaN values after filtering.")
            continue  # Skip this image

        # Feed the keypoints to the evaluation utility

        # print(f"Image {id}: gt_landmarks_valid shape = {gt_landmarks_valid.shape}, result_fingertips_valid shape = {result_fingertips_valid.shape}")
        # Compute centroids
        pred_centroid = np.mean(result_fingertips_cam, axis=0)
        gt_centroid = np.mean(gt_landmarks_valid, axis=0)

        # Compute translation vector
        translation = gt_centroid - pred_centroid

        # Apply only translation
        result_fingertips_cam = result_fingertips_cam + translation


        eval_util.feed(keypoint_gt=gt_landmarks_valid, keypoint_pred=result_fingertips_cam)
        valid_samples += 1  # Only count valid samples
        

    # After processing all images, calculate the evaluation metrics
    val_min, val_max, steps = 20, 50, 100  # PCK thresholds
    epe_mean_all, epe_mean_joint, epe_median_all, auc_all, pck_curve_all, thresholds = eval_util.get_measures(val_min, val_max, steps)

    # Print out the metrics
    print(f"Average EPE Mean: {epe_mean_all}")
    print(f"Joint-wise EPE Mean: {epe_mean_joint}")
    print(f"Average EPE Median: {epe_median_all}")
    print(f"Average AUC: {auc_all}")
    print(f"PCK curve: {pck_curve_all}")
    print(f"Thresholds: {thresholds}")
    print(f"valid_samples: {valid_samples}")


    # Define the results (make sure you have the right data for each variable)
    data_to_save = {
    "average_epe_mean": [epe_mean_all],
    "jointwise_epe_mean_1": [epe_mean_joint[0]],
    "jointwise_epe_mean_2": [epe_mean_joint[1]],
    "jointwise_epe_mean_3": [epe_mean_joint[2]],
    "jointwise_epe_mean_4": [epe_mean_joint[3]],
    "jointwise_epe_mean_5": [epe_mean_joint[4]],
    "average_epe_median": [epe_median_all],
    "average_auc": [auc_all],
    "pck_curve": [list(pck_curve_all)],  # Store PCK as a list
    "thresholds": [list(thresholds)]  # Store thresholds as a list
    }


    # Create a pandas DataFrame from the dictionary
    df = pd.DataFrame(data_to_save)

    # Define output directory and filename
    output_dir = "output_metrics"
    os.makedirs(output_dir, exist_ok=True)  # Create the folder if it doesn't exist
    output_file = os.path.join(output_dir, "metrics_ED_H.csv")

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)

    print(f"Metrics saved to {output_file}")

# Call the function to process and evaluate images
process_and_evaluate_images_from_egodexter()
