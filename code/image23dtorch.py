import numpy as np
import track_utils
import cv2
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
import solvertorch

import time

start_time = time.time()  # Start the timer
# reads an image in the BGR format and converts it to RGB
img = cv2.imread('../images/Mona_Lisa.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gets the landmark results from mediapipe and given image
results = track_utils.track_landmarks_from_img(img)
# print(results)
# If no hand landmarks are found, print a message and continue
if not results.hand_landmarks :
    print("No hand landmarks found in image")
else:
    # Extract the x, y, and z values into a list of lists
    landmarks = [[(lm.x), (lm.y), (lm.z)] for lm in results.hand_landmarks[0][:]]
    # print(landmarks)
    x_real,y_real,z_real,h,w,IsLeft = track_utils.get_real_landmarks(img,results)
    real_landmarks = np.column_stack((x_real, y_real, z_real))
    # print(real_landmarks,IsLeft)

    # rr.init("Keypoints2ManoHand", spawn=True)  # Initialize Rerun viewer
    # rr.log("image", rr.Image(img))
    Output,init_pose,init_shape, result_keypoints = solvertorch.solveminimize(real_landmarks,img,IsLeft=IsLeft)
    print("result_keypoints",result_keypoints)
   
end_time = time.time()  # End the timer

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")