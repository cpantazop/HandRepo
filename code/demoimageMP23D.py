print("Hello")
import track_utils
import display_utils
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 
from solver import *
from armatures import *
from models import *
import config
from vedo import *

# reads an image in the BGR format and converts it to RGB
img = cv2.imread('images/leftdown.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gets the landmark results from mediapipe and given image
results = track_utils.track_landmarks_from_img(img)
# Extract the x, y, and z values into a list of lists
landmarks = [[lm.x*1, (1-lm.y)*1, lm.z*1] for lm in results.hand_landmarks[0][:]]
print(landmarks)
print("###############################################################")

# Desired order of indices
desired_order = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]

# Create a new list using the desired order
reordered_landmarks = [landmarks[i] for i in desired_order]

landmarks_array = np.array(reordered_landmarks)

############## normalize and standarize them
# Calculate the mean of each column (axis=0)
mean_x = np.mean(landmarks_array[:, 0])
mean_y = np.mean(landmarks_array[:, 1])
mean_z = np.mean(landmarks_array[:, 2])
std_x = np.std(landmarks_array[:, 0])
std_y = np.std(landmarks_array[:, 1])
std_z = np.std(landmarks_array[:, 2])
# Normalize by subtracting the mean and dividing by the standard deviation
standarized_landmarks = (landmarks_array - np.array([mean_x, mean_y, mean_z])) / np.array([std_x, std_y, std_z])

# Calculate the min of each column (axis=0)
min_x = np.min(landmarks_array[:, 0])
min_y = np.min(landmarks_array[:, 1])
min_z = np.min(landmarks_array[:, 2])
# Calculate the max of each column (axis=0)
max_x = np.max(landmarks_array[:, 0])
max_y = np.max(landmarks_array[:, 1])
max_z = np.max(landmarks_array[:, 2])
# Normalize by subtracting the min and dividing by max-min
normalized_landmarks = (landmarks_array - np.array([min_x, min_y, min_z])) / np.array([max_x-min_x, max_y-min_y, max_z-min_z])
#transform to [-0.5, 0.5]
normalized_landmarks = (normalized_landmarks - np.array([0.5, 0.5, 0.5]))
print(normalized_landmarks)

# displays original image
imgShow = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
cv2.imshow('image',imgShow)
cv2.waitKey(0)
cv2.destroyAllWindows()
#creates and displays annotated image
annotated_image = display_utils.draw_landmarks_on_image(img, results)
annotated_image = cv2.circle(annotated_image, (0,0), radius=10, color=(0, 0, 255), thickness=-1)
annotated_imageShow = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow('tracked',annotated_imageShow)

cv2.waitKey(0)
cv2.destroyAllWindows()

n_pose = 12 # number of pose pca coefficients, in mano the maximum is 45
mesh = KinematicModel(config.MANO_MODEL_PATH, MANOArmature, scale=1)

wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
solver = Solver(verbose=True,max_iter=119)

keypoints = normalized_landmarks
print("Given keypoints are:",keypoints)
params_est = solver.solve(wrapper, keypoints)

shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)
print('----------------------------------------------------------------------')
print('estimated parameters')
print('pose pca coefficients:', pose_pca_est)
print('pose global rotation:', pose_glb_est)
print('shape: pca coefficients:', shape_est)


mesh.set_params(pose_pca=pose_pca_est)
mesh.save_obj('./est.obj')
hand3d = Mesh("est.obj",)
p1 = Point([0,0,0], c='red') # locate 0,0,0
show(hand3d,p1,axes=1)
# hand3d.show(axes=1)
print('estimated meshes are saved into est.obj')