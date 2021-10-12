
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3



robot = rtb.models.DH.Planar3()

q_min = -(np.pi/4)*np.ones(3)
q_max = (np.pi/3)*np.ones(3)


q_pickup = np.random.uniform(q_min, q_max, (3, ))



qt = rtb.jtraj(robot.qz, q_pickup, 50)
robot.plot(qt.q, movie='planar.gif')
robot = rtb.models.DH.Planar3()



print(robot.jacob0(robot.qz))


