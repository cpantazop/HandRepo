from math import pi
import numpy as np
import torch
import tqdm

from manotorch.axislayer import AxisLayerFK
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.manolayer import ManoLayer, MANOOutput
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

    # Initialization of MANO parameters
    BS = 1 #Batch Size
    betas_shape = torch.zeros(BS, 10, requires_grad=True)
    root_pose = torch.tensor([[0, np.pi / 2, 0]]).repeat(BS, 1)
    finger_pose = torch.zeros(BS, 45)
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)
    hand_pose.requires_grad_(True)

    param = []
    param.append({"params": [hand_pose,betas_shape]}) #choose the trainable parameters
    optimizer = torch.optim.LBFGS(param, lr=1e-2) #choose the optimizer
    #scheduler is optional to reduce the Learning Rate by 'gamma' every 'step_size' 
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5) 
    proc_bar = tqdm.tqdm(range(1000))#the progress bar on the terminal and the steps of the optimization
    rr.init("Keypoints2ManoHand", spawn=True)  # Initialize Rerun viewer
    rr.log("real", rr.Points3D(target_keypoints, colors=(0, 0, 255),labels=[str(i) for i in range(21)], radii=0.1))#show MP real keypoints
    rr.log("image", rr.Image(image))

    target_keypoints = torch.tensor(target_keypoints, dtype=torch.float32)

    # Geman-McClure loss calculation function
    def GMLossCalc(residual,rho=1):#residual is the difference x_estimated-x_real. Default rho is 1
        gm_loss =torch.div(rho ** 2 *residual ** 2, residual ** 2 + rho ** 2)
        return gm_loss.mean()  # .mean() or .sum() to ensure its a scalar for loss.backward()

    def closure(): #required by LBFGS as forward function
        optimizer.zero_grad()  # Zero the gradients

        # Forward pass
        mano_output: MANOOutput = mano_layer(hand_pose, betas_shape)
        J = mano_output.joints[0]

        # Compute the loss
        # MSEloss = torch.nn.functional.mse_loss(J, target_keypoints)
        GMLoss = GMLossCalc(J - target_keypoints,1)
        # shape_reg = torch.norm(betas_shape) * 1e-2  # Regularization term

        total_loss = GMLoss

        # Backward pass
        total_loss.backward()

        return total_loss
    
    for i, _ in enumerate(proc_bar):
        # Perform a single optimization step
        total_loss = optimizer.step(closure)
        # scheduler.step()

        # Update progress bar with the current loss
        proc_bar.set_description(f"Total loss: {total_loss.item():.5f}")

        # Log the estimated keypoints to Rerun
        Jnum = mano_layer(hand_pose, betas_shape).joints[0].detach().numpy()
        rr.log("estimated", rr.Points3D(Jnum, colors=(0, 255, 0), labels=[str(i) for i in range(21)], radii=0.1))

    return mano_layer(hand_pose, betas_shape)