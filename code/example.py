from solver import *
from armatures import *
from models import *
import numpy as np
import config
from vedo import *

np.random.seed(20160923)
# np.random.seed(1596562)
pose_glb = np.zeros([1, 3]) # global rotation

n_pose = 12 # number of pose pca coefficients, in mano the maximum is 45
n_shape = 10 # number of shape pca coefficients
pose_pca = np.random.normal(size=n_pose)
shape = np.random.normal(size=n_shape)
# pose_pca = np.zeros(n_pose)
# shape = np.zeros(n_shape)
mesh = KinematicModel(config.MANO_MODEL_PATH, MANOArmature, scale=10)

wrapper = KinematicPCAWrapper(mesh, n_pose=n_pose)
solver = Solver(verbose=True)

_, keypoints = \
  mesh.set_params(pose_pca=pose_pca, pose_glb=pose_glb, shape=shape)
print("Given keypoints are:",keypoints)
params_est = solver.solve(wrapper, keypoints)

shape_est, pose_pca_est, pose_glb_est = wrapper.decode(params_est)

print('----------------------------------------------------------------------')
print('ground truth parameters')
print('pose pca coefficients:', pose_pca)
print('pose global rotation:', pose_glb)
print('shape: pca coefficients:', shape)

print('----------------------------------------------------------------------')
print('estimated parameters')
print('pose pca coefficients:', pose_pca_est)
print('pose global rotation:', pose_glb_est)
print('shape: pca coefficients:', shape_est)

mesh.set_params(pose_pca=pose_pca)
mesh.save_obj('./gt.obj')
mesh.set_params(pose_pca=pose_pca_est)
mesh.save_obj('./est.obj')
hand3d = Mesh("est.obj",)
hand3d.show(axes=1)
print('ground truth and estimated meshes are saved into gt.obj and est.obj')
