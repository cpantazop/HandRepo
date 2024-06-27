# it must be assured that parent joint appears before child joint

class MANOArmature:
  # number of movable joints
  n_joints = 16

  # indices of extended keypoints
  keypoints_ext = [333, 444, 672, 555, 744]

  n_keypoints = n_joints + len(keypoints_ext)

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    # extended
    'I3', 'M3', 'L3', 'R3', 'T3' #20
  ]

# class MANOArmatureNEW:
#   # number of movable joints
#   n_joints = 16

#   # indices of extended keypoints
#   keypoints_ext = [333, 444, 672, 555, 744]

#   n_keypoints = n_joints + len(keypoints_ext)

#   labels = [
#     'W', #0
#     'T0', 'T1', 'T2','T3', #4
#     'I0', 'I1', 'I2','I3', #8
#     'M0', 'M1', 'M2','M3', #12
#     'R0', 'R1', 'R2','R3', #16
#     'L0', 'L1', 'L2','L3' #20
    
#   ]