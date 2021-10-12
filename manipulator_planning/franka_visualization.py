


import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3



robot = rtb.models.DH.Panda()

q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165  ])*np.pi/180
q_max = np.array([ 165,   101,  165,  1.0,    165, 214, 165  ])*np.pi/180

q_pickup = np.random.uniform(q_min, q_max, (7, ))



qt = rtb.jtraj(robot.qz, q_pickup, 50)
robot.plot(qt.q, movie='planar.gif')

print(robot.jacob0(robot.qz))


# print(robot)

