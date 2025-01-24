from math import pi,dist
import numpy as np
import torch
import tqdm

import open3d
import transforms3d.quaternions

from scipy.optimize import least_squares
from scipy.optimize import minimize

from manotorch.axislayer import AxisLayerFK
from manotorch.anatomy_loss import AnatomyConstraintLossEE
from manotorch.manolayer import ManoLayer, MANOOutput
import rerun as rr  # NOTE: `rerun`, not `rerun-sdk`!

# find the root (0) joint of the hand
def findzero(keypoints):
    return(keypoints[0:1])

# find the 6 palm joints of the hand
def palm(keypoints):
    palm_indices = [0, 1, 5, 9, 13, 17]# Indices of the palm keypoints
    return (keypoints[palm_indices, :])
    

def rigitTrans(source, target):
    """
  registers two pointclouds by rigid transformation
  target_x = target_T_source * source_x
  """
    assert(len(source) == len(target))
    psource = open3d.geometry.PointCloud()
    psource.points = open3d.utility.Vector3dVector(source)
    ptarget = open3d.geometry.PointCloud()
    ptarget.points = open3d.utility.Vector3dVector(target)
    c = [[i, i] for i in range(len(source))]
    c = open3d.utility.Vector2iVector(c)
    r = open3d.pipelines.registration.TransformationEstimationPointToPoint()
    r.with_scaling = False
    tTs = r.compute_transformation(psource, ptarget, c)
    pst = psource.transform(tTs)

    return(tTs)

# Reverse rigitTrans transformation
def reverse_rigid_transformation(keypoints, transformation_matrix):
    """
    Reverses a rigid transformation applied to keypoints.
    
    Args:
    - keypoints (np.ndarray): Nx3 array of transformed 3D keypoints.
    - transformation_matrix (np.ndarray): 4x4 rigid transformation matrix.
    
    Returns:
    - reversed_keypoints (np.ndarray): Nx3 array of keypoints in the original coordinate space.
    """
    # Extract rotation and translation from the transformation matrix
    R = transformation_matrix[:3, :3]  # 3x3 rotation matrix
    t = transformation_matrix[:3, 3]  # 3x1 translation vector

    # Compute the inverse transformation
    R_inv = R.T  # Transpose of rotation matrix
    t_inv = -np.dot(R_inv, t)  # Inverse translation vector

    # Apply the inverse transformation to keypoints
    keypoints_transformed = np.dot(keypoints - t, R_inv.T)
    return keypoints_transformed




def solveminimize(target_keypoints, image, init_pose=None, init_shape=None,IsLeft=True):
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="left" if IsLeft else "right",
        center_idx=None,
        mano_assets_root="manotorch/assets/mano",
        flat_hand_mean=False,
    )

    # print("The side is",mano_layer.side)
    # Initialization of MANO parameters
    BS = 1 #Batch Size
    if(init_shape==None):
        betas_shape = torch.zeros(BS, 10, requires_grad=True)
    else:
        betas_shape=init_shape
    if(init_pose==None):
        root_pose = torch.tensor([[0, 0 , 0]]).repeat(BS, 1)
        finger_pose = torch.zeros(BS, 45)
        hand_pose = torch.cat([root_pose, finger_pose], dim=1)
        hand_pose.requires_grad_(True)
    else:
        hand_pose=init_pose


    mano_output: MANOOutput = mano_layer(hand_pose, betas_shape)
    mano_keypoints = mano_output.joints[0].detach().numpy()

    # Compute the scaling factor
    mano_palm_size = np.linalg.norm(mano_keypoints[0] - mano_keypoints[5])  # Example: thumb to index root
    target_palm_size = np.linalg.norm(target_keypoints[0] - target_keypoints[5])
    scaling_factor = mano_palm_size / target_palm_size

    # Scale target keypoints
    target_keypoints_scaled = target_keypoints * scaling_factor

    mTc = rigitTrans(palm(target_keypoints_scaled),palm(mano_keypoints))
    print(mTc)

    target_keypoints_trans = np.dot(
        mTc, np.vstack((target_keypoints_scaled.T, np.ones(len(target_keypoints_scaled))))
    )
    target_keypoints_trans = target_keypoints_trans[:3].T


    rr.init("Keypoints2ManoHand", spawn=True)  # Initialize Rerun viewer
    rr.log("real_NOT_translated", rr.Points3D(target_keypoints, colors=(0, 255, 255),labels=[str(i) for i in range(21)], radii=0.001))#show MP real keypoints
    rr.log("real_translated", rr.Points3D(target_keypoints_trans, colors=(0, 0, 255),labels=[str(i) for i in range(21)], radii=0.001))#show MP real keypoints
    rr.log("real_SCALED", rr.Points3D(target_keypoints_scaled, colors=(255, 255, 255),labels=[str(i) for i in range(21)], radii=0.001))#show MP real keypoints
    rr.log("real2D", rr.Points2D(target_keypoints[:, :2], colors=(0, 0, 255),labels=[str(i) for i in range(21)], radii=0.001))#show MP real keypoints
    rr.log("image", rr.Image(image))

    
    rr.log("manozeroKP", rr.Points3D(mano_keypoints, colors=(255, 0, 0), labels=[str(i) for i in range(21)], radii=0.001))

    verts = mano_layer(hand_pose, betas_shape).verts
    faces = mano_layer.th_faces
    V = verts[0].detach().numpy()
    F = faces.numpy()
    # Log the hand mesh to the Rerun viewer
    rr.log("manozeroMesh",rr.Mesh3D(vertex_positions=V,triangle_indices=F,vertex_colors=(255, 0, 0,5)))


    # Combine initial pose and shape into a single parameter vector
    initial_guess = np.concatenate([
        hand_pose.detach().numpy().flatten(),
        betas_shape.detach().numpy().flatten()
    ])

    anatomy_loss = AnatomyConstraintLossEE()
    anatomy_loss.setup()
    axis_layer = AxisLayerFK(mano_assets_root="manotorch/assets/mano")

    def objective_functionMSEAnatomy(params, mano_layer, target_keypoints, progress=True):
        """
        Objective function for optimization: sum of squared residuals.
        """
        hand_pose_np = params[:48].reshape((1, -1))
        betas_np = params[48:].reshape((1, -1))

        # Convert to tensors
        hand_pose_tensor = torch.tensor(hand_pose_np, dtype=torch.float32, requires_grad=True)
        betas_tensor = torch.tensor(betas_np, dtype=torch.float32, requires_grad=True)

        # Forward pass through MANO
        mano_output: MANOOutput = mano_layer(hand_pose_tensor, betas_tensor)
        J = mano_output.joints[0]

        if progress:
            # Log current hand mesh and keypoints
            verts = mano_output.verts[0].detach().numpy()
            faces = mano_layer.th_faces.numpy()
            rr.log("estimated_hand", rr.Mesh3D(vertex_positions=verts, triangle_indices=faces))
            rr.log("estimated_keypoints", rr.Points3D(J.detach().numpy(), colors=(255, 0, 0)))

        # Calculate losses
            mse_loss = torch.nn.functional.mse_loss(J, torch.tensor(target_keypoints, dtype=torch.float32))
            T_g_p = mano_output.transforms_abs  # (B, 16, 4, 4)
            T_g_a, R, ee = axis_layer(T_g_p)  # ee (B, 16, 3)
            anatomy_constraint = anatomy_loss(ee) 


            # Weighted combination of losses
            total_loss = 1 * mse_loss #+ 0.01* anatomy_constraint

             # Compute gradients
            total_loss.backward()

            # Combine gradients for pose and shape
            grads = np.concatenate([
                hand_pose_tensor.grad.numpy().flatten(),
                betas_tensor.grad.numpy().flatten()
            ])

            # Log current losses
            print("mse_loss", mse_loss.item())
            # print("anatomy_loss", anatomy_constraint.item())
            # print("total_loss", total_loss.item())
            
            return total_loss.item(), grads


    # Optimization using scipy minimize
    result = minimize(
        fun = lambda params: objective_functionMSEAnatomy(params, mano_layer, target_keypoints_trans),
        x0=initial_guess,
        jac=True,  # Indicate that the gradient is provided
        method="L-BFGS-B",
        # method="BFGS",
        options={'disp': True, 'maxiter': 500}  # Optional: display optimization progress
    )

    # Find root_pose
    # Extract rotation and translation
    rotation_matrix = mTc[:3, :3]
    translation_vector = mTc[:3, 3]

    # # Convert rotation matrix to axis and angle
    axis, angle = transforms3d.axangles.mat2axangle(rotation_matrix)

    # # Compact representation: axis * angle
    axis_angle = (axis * angle*-1).tolist() # This is your 3-value Axis-Angle representation

    # Extract optimized pose and shape
    optimized_params = result.x
    #the root pose here is probably zero but we use that to later do the reverse transformation and find the correct rotation and translation in the same time
    optimized_hand_pose = torch.tensor(optimized_params[:48].reshape((1, -1)), dtype=torch.float32)
    optimized_betas = torch.tensor(optimized_params[48:].reshape((1, -1)), dtype=torch.float32)

    # set root pose for the correct mano parameters that will not fit to the keypoints due to the translation but have the correct pose and rotation
    optimized_params[:3] = axis_angle
    correct_root_optimized_hand_pose = torch.tensor(optimized_params[:48].reshape((1, -1)), dtype=torch.float32)

    # Log the estimated keypoints to Rerun
    Jnum = mano_layer(optimized_hand_pose, optimized_betas).joints[0]
    Jnum = Jnum.detach().numpy()
    rr.log("estimatedfinalKP", rr.Points3D(Jnum, colors=(255, 0, 0), labels=[str(i) for i in range(21)], radii=0.001))

    verts = mano_layer(optimized_hand_pose, optimized_betas).verts
    faces = mano_layer.th_faces
    V = verts[0].detach().numpy()
    F = faces.numpy()
    # Log the hand mesh to the Rerun viewer
    rr.log("estimatedfinalMesh",rr.Mesh3D(vertex_positions=V,triangle_indices=F,vertex_colors=(255, 0, 0,5)))

    
    reversed_mano_keypoints = reverse_rigid_transformation(Jnum, mTc)
    reversed_mano_keypoints = (1 / scaling_factor)*(reversed_mano_keypoints)
    rr.log("mano2D", rr.Points2D(reversed_mano_keypoints[:, :2], colors=(0, 255, 0),labels=[str(i) for i in range(21)], radii=0.001))#show mano final keypoints                        

    # Project the estimated mesh onto the image


    # Reverse the rigid transformation
    reversed_vertices = reverse_rigid_transformation(V, mTc)
    # Reverse scaling and translation
    reversed_vertices = (1 / scaling_factor) * (reversed_vertices)


    # TESTS
    # Compute face normals
    face_normals = np.cross(
        V[F[:, 1]] - V[F[:, 0]],  # Edge 1 of each triangle
        V[F[:, 2]] - V[F[:, 0]]   # Edge 2 of each triangle
    )
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)  # Normalize

    # Initialize vertex normals to zero
    vertex_normals = np.zeros_like(V)

    # Accumulate normals for each vertex based on the associated faces
    for i, face in enumerate(F):
        vertex_normals[face[0]] += face_normals[i]
        vertex_normals[face[1]] += face_normals[i]
        vertex_normals[face[2]] += face_normals[i]

    # Normalize the vertex normals
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)


    rr.log("estimatedfinalMeshBIG",rr.Mesh3D(vertex_positions=reversed_vertices,triangle_indices=F,vertex_colors=(195, 130, 90),vertex_normals=vertex_normals))

    # Log the projected 2D points to Rerun
    rr.log("mesh_projected_2D", rr.Points2D(
        reversed_vertices[:, :2],  # Take only x and y coordinates
        colors=(255, 0, 0),        # Green color for mesh points
        radii=0.1                # Set appropriate radius for visibility
    ))

    projected_2D_vertices = np.array(reversed_vertices[:, :2])


    # Collect all edges from the triangles
    all_edges = []  # List to store line segments (pairs of points)
    for triangle in F:  # `F` contains the triangle indices
        # Get the 3 vertices of the triangle (each as 2D)
        pts = projected_2D_vertices[triangle]
        # Create pairs of points (edges) for lines
        for i in range(3):
            p1 = pts[i]
            p2 = pts[(i + 1) % 3]  # Wrap around to close the triangle
            all_edges.append([p1.tolist(), p2.tolist()])  # Ensure points are in list format

    # Log all edges in a single call
    rr.log(
        "mesh_projected_2D_edges",
        rr.LineStrips2D(
            all_edges,  # List of line segments
            colors=[[200, 200, 255] for _ in all_edges],  # Color for each line
            radii=[0.5 for _ in all_edges],  # Thickness for each line
            # labels=[f"Edge {i}" for i in range(len(all_edges))],  # Optional: label each edge
        )
    )

    
    return mano_layer(correct_root_optimized_hand_pose, optimized_betas), correct_root_optimized_hand_pose, optimized_betas



def solveminimizeFAST(target_keypoints, image, init_pose=None, init_shape=None,IsLeft=True):
    mano_layer = ManoLayer(
        rot_mode="axisang",
        use_pca=False,
        side="left" if IsLeft else "right",
        center_idx=None,
        mano_assets_root="manotorch/assets/mano",
        flat_hand_mean=False,
    )

    # print("The side is",mano_layer.side)
    # Initialization of MANO parameters
    BS = 1 #Batch Size
    
    betas_shape = torch.zeros(BS, 10, requires_grad=True)
    root_pose = torch.tensor([[0, 0 , 0]]).repeat(BS, 1)
    finger_pose = torch.zeros(BS, 45)
    hand_pose = torch.cat([root_pose, finger_pose], dim=1)
    hand_pose.requires_grad_(True)
   

    mano_output: MANOOutput = mano_layer(hand_pose, betas_shape)
    mano_keypoints = mano_output.joints[0].detach().numpy()

    # Compute the scaling factor
    mano_palm_size = np.linalg.norm(mano_keypoints[0] - mano_keypoints[5])  # Example: thumb to index root
    target_palm_size = np.linalg.norm(target_keypoints[0] - target_keypoints[5])
    scaling_factor = mano_palm_size / target_palm_size

    # Scale target keypoints
    target_keypoints_scaled = target_keypoints * scaling_factor

    mTc = rigitTrans(palm(target_keypoints_scaled),palm(mano_keypoints))
    print(mTc)

    target_keypoints_trans = np.dot(
        mTc, np.vstack((target_keypoints_scaled.T, np.ones(len(target_keypoints_scaled))))
    )
    target_keypoints_trans = target_keypoints_trans[:3].T




    # Combine initial pose and shape into a single parameter vector
    initial_guess = np.concatenate([
        hand_pose.detach().numpy().flatten(),
        betas_shape.detach().numpy().flatten()
    ])

    def objective_functionMSEAnatomy(params, mano_layer, target_keypoints, progress=True):
        """
        Objective function for optimization: sum of squared residuals.
        """
        hand_pose_np = params[:48].reshape((1, -1))
        betas_np = params[48:].reshape((1, -1))

        # Convert to tensors
        hand_pose_tensor = torch.tensor(hand_pose_np, dtype=torch.float32, requires_grad=True)
        betas_tensor = torch.tensor(betas_np, dtype=torch.float32, requires_grad=True)

        # Forward pass through MANO
        mano_output: MANOOutput = mano_layer(hand_pose_tensor, betas_tensor)
        J = mano_output.joints[0]

        # Calculate losses
        mse_loss = torch.nn.functional.mse_loss(J, torch.tensor(target_keypoints, dtype=torch.float32))
            
        # Weighted combination of losses
        total_loss = mse_loss #+ 0.01* anatomy_constraint

        # Compute gradients
        total_loss.backward()

        #  Combine gradients for pose and shape
        grads = np.concatenate([
            hand_pose_tensor.grad.numpy().flatten(),
            betas_tensor.grad.numpy().flatten()
        ])

        
        return total_loss.item(), grads


    # Optimization using scipy minimize
    result = minimize(
        fun = lambda params: objective_functionMSEAnatomy(params, mano_layer, target_keypoints_trans),
        x0=initial_guess,
        jac=True,  # Indicate that the gradient is provided
        method="L-BFGS-B",
        # method="BFGS",
        options={'disp': True, 'maxiter': 500}  # Optional: display optimization progress
    )

    # Find root_pose
    # Extract rotation and translation
    rotation_matrix = mTc[:3, :3]
    translation_vector = mTc[:3, 3]

    # # Convert rotation matrix to axis and angle
    axis, angle = transforms3d.axangles.mat2axangle(rotation_matrix)

    # # Compact representation: axis * angle
    axis_angle = (axis * angle*-1).tolist() # This is your 3-value Axis-Angle representation

    # Extract optimized pose and shape
    optimized_params = result.x
    #the root pose here is probably zero but we use that to later do the reverse transformation and find the correct rotation and translation in the same time
    optimized_hand_pose = torch.tensor(optimized_params[:48].reshape((1, -1)), dtype=torch.float32)
    optimized_betas = torch.tensor(optimized_params[48:].reshape((1, -1)), dtype=torch.float32)

    # set root pose for the correct mano parameters that will not fit to the keypoints due to the translation but have the correct pose and rotation
    optimized_params[:3] = axis_angle
    correct_root_optimized_hand_pose = torch.tensor(optimized_params[:48].reshape((1, -1)), dtype=torch.float32)

    
    return mano_layer(correct_root_optimized_hand_pose, optimized_betas), correct_root_optimized_hand_pose, optimized_betas
