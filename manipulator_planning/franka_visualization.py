


import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

q_data = loadmat('frank_joint_traj.mat')['q_traj']

robot = rtb.models.DH.Panda()

q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165  ])*np.pi/180
q_max = np.array([ 165,   101,  165,  1.0,    165, 214, 165  ])*np.pi/180

q_pickup = np.random.uniform(q_min, q_max, (7, ))

# q_traj = np.linspace(q_min, q_max, 100)
# print(np.shape(q_traj))
# kk


# qt = rtb.jtraj(robot.qz, q_pickup, 50)
# qt = rtb.tools.trajectory.qplot(q_traj)
# print(qt[0])

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.view_init(elev=106., azim=39)
robot.plot(q_data[0:100], limits = [-1.0, 1.0, -1.0, 1.0, 0, 1.0])

print(robot.jacob0(robot.qz))


# print(robot)

