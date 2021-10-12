

import numpy as np
import roboticstoolbox as rtb

from spatialmath import SE3
from roboticstoolbox.backends.Swift import Swift 

# robot = rtb.models.DH.Planar3()

# robot = rtb.models.DH.Panda()
robot = rtb.models.URDF.Panda()



# T = SE3(0.7, 0.2, 0.1) * SE3.OA([0, 1, 0], [0, 0, -1])
# sol = robot.ikine_LM(T)         # solve IK
# print(sol)

# q_pickup = sol.q
# print(robot.fkine(q_pickup))    # FK shows that desired end-effector pose was achieved


# q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165  ])*np.pi/180
# q_max = np.array([ 165,   101,  165,  1.0,    165, 214, 165  ])*np.pi/180

# q_pickup = np.random.uniform(q_min, q_max, (7, ))

# q_min = -(np.pi/4)*np.ones(3)
# q_max = (np.pi/3)*np.ones(3)


# # q_pickup = np.random.uniform(q_min, q_max, (3, ))

# # print(robot.qz)




# qt = rtb.jtraj(robot.qz, q_pickup, 50)
# # robot.plot(qt.q, movie='planar.gif')
# robot.plot(robot.qz, backend="swift")



# print(robot)

