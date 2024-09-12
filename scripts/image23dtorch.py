import numpy as np
import track_utils
import display_utils
import cv2
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import solvertorch


# reads an image in the BGR format and converts it to RGB
img = cv2.imread('C:/Users/chris/Documents/GitHub/manotorch/images/image_00010_color.png')
print(img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gets the landmark results from mediapipe and given image
results = track_utils.track_landmarks_from_img(img)
# Extract the x, y, and z values into a list of lists
# landmarks = [[(lm.x)*1, (1-lm.y)*1, (1-lm.z)*1] for lm in results.hand_landmarks[0][:]]
landmarks = [[(lm.x)*1, (lm.y)*1, (lm.z)*1] for lm in results.hand_landmarks[0][:]]
print(landmarks)
x_real,y_real,z_real,h,w = track_utils.get_real_landmarks(img,results)
real_landmarks = np.column_stack((x_real, y_real, z_real))
print(real_landmarks)

def recalculate_keypoints(K, real_landmarks):
    """
    Recalculates the keypoints using the intrinsic matrix K and real landmarks.

    Args:
    - K: 3x3 intrinsic matrix.
    - real_landmarks: Nx3 array of 3D real landmarks (x, y, z).

    Returns:
    - K_landmarks: Nx3 array of recalculated keypoints using K.
    """
    # Ensure real_landmarks are in homogeneous coordinates (i.e., [x, y, z, 1])
    num_landmarks = real_landmarks.shape[0]
    # Homogeneous coordinates are [x, y, z, 1], so we append a column of ones
    homogeneous_landmarks = np.hstack((real_landmarks, np.ones((num_landmarks, 1))))
    
    # Perform matrix multiplication for each real landmark with the intrinsic matrix K
    K_landmarks = np.dot(homogeneous_landmarks[:, :3], K.T)
    
    return K_landmarks

# Define the intrinsic matrix K
K = np.array([
    [617.173, 0, 315.453],
    [0, 617.173, 242.259],
    [0, 0, 1]
])

# Call the function to recalculate keypoints
K_landmarks = recalculate_keypoints(K, real_landmarks)

# Print the recalculated keypoints
print(K_landmarks)

Output = solvertorch.solve(K_landmarks,img)


