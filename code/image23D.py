print("Hello")
import track_utils
import display_utils
import cv2
import numpy as np
from matplotlib import pyplot as plt
# 
from solver import *
from torchsolver import *
from armatures import *
from models import *
import config

# reads an image in the BGR format and converts it to RGB
img = cv2.imread('images/example_left.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# gets the landmark results from mediapipe and given image
results = track_utils.track_landmarks_from_img(img)
# Extract the x, y, and z values into a list of lists
landmarks = [[(lm.x)*1, (1-lm.y)*1, (1-lm.z)*1] for lm in results.hand_landmarks[0][:]]
# landmarks = [[(lm.x)*1, (lm.y)*1, (lm.z)*1] for lm in results.hand_landmarks[0][:]]
print(landmarks)
x_real,y_real,z_real,h,w = track_utils.get_real_landmarks(img,results)
real_landmarks = np.column_stack((x_real, y_real, z_real))
print(real_landmarks)
# landmarks =real_landmarks

print("###############################################################")

# Desired order of indices
# desired_order = [0, 13, 14, 15, 20, 1, 2, 3, 16, 4, 5, 6, 17, 10, 11, 12, 19, 7, 8, 9, 18]
desired_order = [0,5,6,7,9,10,11,17,18,19,13,14,15,1,2,3,8,12,20,16,4]
# Create a new list using the desired order
landmarks_array = np.array([landmarks[i] for i in desired_order])
real_landmarks_array = np.array([real_landmarks[i] for i in desired_order])
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
# x_real,y_real,z_real = display_utils.get_real_landmarks(img,results)
# Draw landmarks
for x, y in zip(x_real, y_real):
    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), thickness=-1)  # Draw a green dot
annotated_imageShowReal = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
cv2.imshow('tracked+real',annotated_imageShowReal)
cv2.waitKey(0)
cv2.destroyAllWindows()

n_pose = 12 # number of pose pca coefficients, in mano the maximum is 45
mesh = KinematicModel(config.MANO_MODEL_PATH, MANOArmature, scale=10)

wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
solver = SolverBFGS(verbose=False,mse_threshold = 1e-30,eps= 1e-4, max_iter=2)

keypoints = real_landmarks_array
print("Given keypoints are:",keypoints)
# params_est, BF_keypoints = solver.solveWrist(wrapper, keypoints)
params_est, BF_keypoints = solver.solve(wrapper, keypoints)
# params_est, BF_keypoints = solver.solve(wrapper, keypoints, params_est)
shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)
print('----------------------------------------------------------------------')
print('estimated parameters')
print('pose pca coefficients:', pose_pca_est)
print('pose global rotation:', pose_glb_est)
print('shape: pca coefficients:', shape_est)


mesh.set_params(pose_pca=pose_pca_est)
mesh.save_obj('./est.obj')
# hand3d = Mesh("est.obj",)



# import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!
# rr.init("rerun_example_my_data", spawn=True)

# # positions = keypoints

# rr.log("real",rr.Points3D(keypoints,colors=(0,0,255), radii=1))
# rr.log("estimated",rr.Points3D(BF_keypoints,colors=(0,255,0), radii=1))
# rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
# rr.log("image", rr.Image(img))
# rr.log("estimated", rr.Mesh3D("est.obj"))

# print('estimated meshes are saved into est.obj')