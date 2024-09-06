from math import pi

import numpy as np
# import open3d as o3d
import torch
import tqdm

from manotorch.anchorlayer import AnchorLayer
from manotorch.axislayer import AxisAdaptiveLayer, AxisLayerFK
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.manolayer import ManoLayer, MANOOutput
# from manotorch.utils.visutils import VizContext, create_coord_system_can
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

def solve(target_keypoints,image):
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root="assets/mano",
        flat_hand_mean=False,
    )
    axis_layer = AxisLayerFK(mano_assets_root="assets/mano")
    anchor_layer = AnchorLayer(anchor_root="assets/anchor")

    BS = 1 #Batch Size
    zero_shape = torch.zeros(BS, 10, requires_grad=True)
    root_pose = torch.tensor([[0, np.pi / 2, 0]]).repeat(BS, 1)
    finger_pose = torch.zeros(BS, 45)
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)
    hand_pose.requires_grad_(True)

    param = []
    param.append({"params": [hand_pose,zero_shape]})
    optimizer = torch.optim.LBFGS(param, lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    anatomy_loss = AnatomyConstraintLossEE()
    anatomy_loss.setup()
    proc_bar = tqdm.tqdm(range(1000))
    rr.init("rerun_example_my_data", spawn=True)  # Initialize Rerun viewer
    rr.log("real", rr.Points3D(target_keypoints, colors=(0, 0, 255),labels=[str(i) for i in range(21)], radii=0.1))
    rr.log("image", rr.Image(image))

    target_keypoints = torch.tensor(target_keypoints, dtype=torch.float32)

    def closure():
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        mano_output: MANOOutput = mano_layer(hand_pose, zero_shape)
        J = mano_output.joints[0]
        T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
        T_g_a, R, ee = axis_layer(T_g_p)  # ee (B, 16, 3)

        # Compute the loss
        # MSEloss = torch.nn.functional.mse_loss(J[:, :2], target_keypoints[:, :2])
        MSEloss = torch.nn.functional.mse_loss(J, target_keypoints)
        losszero = torch.nn.functional.mse_loss(J[0] ,target_keypoints[0])
        shape_reg = torch.norm(zero_shape) * 1e-2  # Regularization term
        anatomy_constraint = anatomy_loss(ee) * 1e-2

        total_loss = MSEloss + anatomy_constraint + shape_reg 
        # total_loss = losszero*100 + anatomy_constraint + shape_reg

        # Backward pass
        total_loss.backward()

        return total_loss
    
    for i, _ in enumerate(proc_bar):
        # Perform a single optimization step
        total_loss = optimizer.step(closure)
        scheduler.step()

        # Update progress bar with the current loss
        proc_bar.set_description(f"MSE loss: {total_loss.item():.5f}")

        # Log the estimated keypoints to Rerun
        Jnum = mano_layer(hand_pose, zero_shape).joints[0].detach().numpy()
        rr.log("estimated", rr.Points3D(Jnum, colors=(0, 255, 0), labels=[str(i) for i in range(21)], radii=0.1))

    return mano_layer(hand_pose, zero_shape)


def solve2stages(target_keypoints,image):
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="right",
        center_idx=None,
        mano_assets_root="assets/mano",
        flat_hand_mean=False,
    )
    axis_layer = AxisLayerFK(mano_assets_root="assets/mano")
    anchor_layer = AnchorLayer(anchor_root="assets/anchor")

    BS = 1 #Batch Size
    zero_shape = torch.zeros(BS, 10, requires_grad=True)
    # root_pose = torch.tensor([[0, np.pi / 2, 0]]).repeat(BS, 1)
    root_pose = torch.tensor([[0, np.pi / 2, 0]]).repeat(BS, 1)
    finger_pose = torch.zeros(BS, 45)
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)
    hand_pose.requires_grad_(True)

    param = []
    param.append({"params": [hand_pose,zero_shape]})
    # param.append({"params": [root_pose]})
    optimizer = torch.optim.LBFGS(param, lr=1e-1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)
    anatomy_loss = AnatomyConstraintLossEE()
    anatomy_loss.setup()
    proc_bar = tqdm.tqdm(range(1000))
    rr.init("rerun_example_my_data", spawn=True)  # Initialize Rerun viewer
    rr.log("real", rr.Points3D(target_keypoints, colors=(0, 0, 255),labels=[str(i) for i in range(21)], radii=0.1))
    rr.log("image", rr.Image(image))

    target_keypoints = torch.tensor(target_keypoints, dtype=torch.float32)

    def closure():
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        mano_output: MANOOutput = mano_layer(hand_pose, zero_shape)
        J = mano_output.joints[0]
        T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
        T_g_a, R, ee = axis_layer(T_g_p)  # ee (B, 16, 3)

        # Compute the loss
        # MSEloss = torch.nn.functional.mse_loss(J, target_keypoints)
        losszero = torch.nn.functional.mse_loss(J[0] ,target_keypoints[0])
        # lossnine = torch.nn.functional.mse_loss(J[9] ,target_keypoints[9])
        # print("target_keypoints[0]  is")
        # print(target_keypoints[0])
        shape_reg = torch.norm(zero_shape) * 1e-2  # Regularization term
        # loss = torch.nn.functional.mse_loss(J, target_keypoints) + shape_reg

        anatomy_constraint = anatomy_loss(ee) * 1e-2
        # total_loss = MSEloss + anatomy_constraint + shape_reg #+ 100*losszero
        total_loss = 10*losszero + anatomy_constraint + shape_reg

        # Backward pass
        total_loss.backward()

        return total_loss
    
    def closure2():
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        mano_output: MANOOutput = mano_layer(hand_pose, zero_shape)
        J = mano_output.joints[0]
        T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
        T_g_a, R, ee = axis_layer(T_g_p)  # ee (B, 16, 3)

        # Compute the loss
        MSEloss = torch.nn.functional.mse_loss(J, target_keypoints)
        # losszero = torch.nn.functional.mse_loss(J[0] ,target_keypoints[0])
        # print("target_keypoints[0]  is")
        # print(target_keypoints[0])
        shape_reg = torch.norm(zero_shape) * 1e-3  # Regularization term
        # loss = torch.nn.functional.mse_loss(J, target_keypoints) + shape_reg

        anatomy_constraint = anatomy_loss(ee) * 1e-1
        total_loss = MSEloss + anatomy_constraint + shape_reg #+ 100*losszero
        # total_loss = losszero*100 + anatomy_constraint + shape_reg

        # Backward pass
        total_loss.backward()

        return total_loss
    
    for i in range(100):
        # Perform a single optimization step
        total_loss = optimizer.step(closure)
        scheduler.step()

        # Update progress bar with the current loss
        proc_bar.set_description(f"Zero loss: {total_loss.item():.5f}")

        # Log the estimated keypoints to Rerun
        Jnum = mano_layer(hand_pose, zero_shape).joints[0].detach().numpy()
        rr.log("estimated", rr.Points3D(Jnum, colors=(0, 255, 0), labels=[str(i) for i in range(21)], radii=0.1))

    optimizer = torch.optim.LBFGS(param, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)


    for i, _ in enumerate(proc_bar):
        # Perform a single optimization step
        total_loss = optimizer.step(closure2)
        scheduler.step()

        # Update progress bar with the current loss
        proc_bar.set_description(f"MSE loss: {total_loss.item():.5f}")

        # Log the estimated keypoints to Rerun
        Jnum = mano_layer(hand_pose, zero_shape).joints[0].detach().numpy()
        rr.log("estimated", rr.Points3D(Jnum, colors=(0, 255, 0), labels=[str(i) for i in range(21)], radii=0.1))


    return mano_layer(hand_pose, zero_shape)

    # for i, _ in enumerate(proc_bar):

    #     # curr_pose = torch.cat(hand_pose, dim=1).reshape(1, -1)
    #     curr_pose = hand_pose
    #     mano_output: MANOOutput = mano_layer(curr_pose, zero_shape)
    #     hand_verts_curr = mano_output.verts
    #     J = mano_output.joints[0]

    #     Jnum = J.detach().numpy()

    #     # T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
    #     # T_g_a, R, ee = axisFK(T_g_p)  # ee (B, 16, 3)

    #     # loss = anatomyLoss(ee)

    #     loss = torch.nn.functional.mse_loss(J, target_keypoints)

    #     proc_bar.set_description(f"MSE loss: {loss.item():.5f}")
    #     loss.backward()
    #     optimizer.step()
    #     scheduler.step()

        
    #     rr.log("estimated", rr.Points3D(Jnum, colors=(0, 0, 255),labels=[str(i) for i in range(21)], radii=1))




    # return(mano_output)