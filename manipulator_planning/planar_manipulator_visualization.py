
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3



robot = rtb.models.DH.Planar3()

q_min = -(np.pi/4)*np.ones(3)
q_max = (np.pi/3)*np.ones(3)


q_pickup = np.random.uniform(q_min, q_max, (3, ))

q_pickup = np.array([-0.88477546, -2.06223881, -0.80570723])

q_init = np.hstack(( 0.1, -0.6, 0.0   ))


qt = rtb.jtraj(q_init, q_pickup, 50)
robot.plot(qt.q, movie='planar.gif')
robot = rtb.models.DH.Planar3()



print(robot.jacob0(robot.qz))


