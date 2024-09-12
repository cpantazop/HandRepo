import numpy as np
import track_utils
import display_utils
import cv2

# import torch

# from manotorch.anchorlayer import AnchorLayer
# from manotorch.axislayer import AxisLayerFK
# from manotorch.manolayer import ManoLayer, MANOOutput

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
x_real_adjusted = [x - x_real[0] for x in x_real]
y_real_adjusted = [y - y_real[0] for y in y_real]
z_real_adjusted = [z - z_real[0] for z in z_real]
print(x_real_adjusted)
real_landmarks = np.column_stack((x_real, y_real, z_real))
print(real_landmarks)
real_landmarks_adjusted= np.column_stack((x_real_adjusted, y_real_adjusted, z_real_adjusted))
# print(real_landmarks_adjusted)

# displays original image
imgShow = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# cv2.imshow('image',imgShow)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#creates and displays annotated image
annotated_image = display_utils.draw_landmarks_on_image(img, results)
annotated_image = cv2.circle(annotated_image, (0,0), radius=10, color=(0, 0, 255), thickness=-1)
annotated_imageShow = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
# cv2.imshow('tracked',annotated_imageShow)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# x_real,y_real,z_real = display_utils.get_real_landmarks(img,results)
# Draw landmarks
for x, y in zip(x_real, y_real):
    cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), thickness=-1)  # Draw a green dot
annotated_imageShowReal = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
# cv2.imshow('tracked+real',annotated_imageShowReal)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

Output = solvertorch.solve(real_landmarks,img)

# mano_layer = ManoLayer(
#         rot_mode="axisang",
#         use_pca=False,
#         side="right",
#         center_idx=None,
#         mano_assets_root="assets/mano",
#         flat_hand_mean=False,
#     )
# axis_layer = AxisLayerFK(mano_assets_root="assets/mano")
# anchor_layer = AnchorLayer(anchor_root="assets/anchor")

# BS = 1 #Batch Size
# zero_shape = torch.zeros(BS, 10)
# root_pose = torch.tensor([[0, np.pi / 2, 0]]).repeat(BS, 1)
# finger_pose = torch.zeros(BS, 45)
# hand_pose = torch.cat([root_pose, finger_pose], dim=1)

# hand_pose.requires_grad_(True)

# mano_results: MANOOutput = mano_layer(hand_pose, zero_shape)
#     J = mano_results.joints
#     Jnum = mano_results.joints[0].numpy()
#     print("Jnum is")
#     print(Jnum)
#     print("with shape")
#     print(Jnum.shape)
#     verts = mano_results.verts
#     faces = mano_layer.th_faces
#     V = verts[0].numpy()
#     F = faces.numpy()

#     rr.init("rerun_example_my_data", spawn=True)  # Initialize Rerun viewer
#     rr.log("real", rr.Points3D(Jnum, colors=(0, 0, 255),labels=[str(i) for i in range(21)], radii=1))

#      # Log the hand mesh to the Rerun viewer
#     # rr.log("hand_mesh",rr.Mesh3D(vertex_positions=V,triangle_indices=F,#vertex_colors=(255, 0, 0,0), #Opacity=0.5 #opacity
#                                 #  ))
    