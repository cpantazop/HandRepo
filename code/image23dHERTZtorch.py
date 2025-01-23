import numpy as np
import track_utils
import cv2
import solvertorch
import time
import os

# Define the folder containing the images and the file pattern
image_folder = 'C:/Users/chris/Documents/GitHub/HandRepo/manotorch/images/'
image_pattern = 'image_{:05d}_color.png'  # Pattern for file names

# Initialize variables for timing
num_frames = 11  # Process images from image_00000_color.png to image_00010_color.png
start_time = time.time()  # Start the timer

# # Initialize init_pose and init_shape as None, as the first iteration will set them
# init_pose = None
# init_shape = None

for i in range(num_frames):
    # Generate the image file path
    image_path = os.path.join(image_folder, image_pattern.format(i))
    
    # Read the image in BGR format and convert to RGB
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    # Process the image (track landmarks and estimate hand pose)
    results = track_utils.track_landmarks_from_img(img)
    if not results.hand_landmarks :
        print(f"No hand landmarks found in image: {image_path}")
        continue
    
    # Extract landmarks
    landmarks = [[(lm.x), (lm.y), (lm.z)] for lm in results.hand_landmarks[0][:]]
    x_real, y_real, z_real, h, w, IsLeft = track_utils.get_real_landmarks(img, results)
    real_landmarks = np.column_stack((x_real, y_real, z_real))
    
    # Run the solver
    Output, init_pose, init_shape = solvertorch.solveminimize(real_landmarks, img, IsLeft=IsLeft, init_pose=None, init_shape=None)

# End timing
end_time = time.time()

# Calculate elapsed time and runtime in Hz
elapsed_time = end_time - start_time
runtime_hz = num_frames / elapsed_time

print(f"Processed {num_frames} frames in {elapsed_time:.2f} seconds")
print(f"Runtime: {runtime_hz:.2f} Hz")
