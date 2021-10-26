

############# franka_IK

import jax.numpy as jnp
import numpy as np
from jax import jit, grad

import matplotlib.pyplot as plt
import time
# import scipy.io
from mpl_toolkits.mplot3d import Axes3D
import  scipy.io 

################ franka_Ik.py

def compute_fk(q):
    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    q_5 = q[4]
    q_6 = q[5]
    q_7 = q[6]

    x = -0.107*(((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1))*jnp.cos(q_5) + (jnp.sin(q_1)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1)*jnp.cos(q_2))*jnp.sin(q_5))*jnp.sin(q_6) - 0.088*(((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1))*jnp.cos(q_5) + (jnp.sin(q_1)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1)*jnp.cos(q_2))*jnp.sin(q_5))*jnp.cos(q_6) + 0.088*((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4))*jnp.sin(q_6) - 0.107*((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4))*jnp.cos(q_6) + 0.384*(jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + 0.0825*(jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - 0.0825*jnp.sin(q_1)*jnp.sin(q_3) - 0.0825*jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1) + 0.384*jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4) + 0.316*jnp.sin(q_2)*jnp.cos(q_1) + 0.0825*jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3)

    y = 0.107*(((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) + jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4))*jnp.cos(q_5) - (jnp.sin(q_1)*jnp.sin(q_3)*jnp.cos(q_2) - jnp.cos(q_1)*jnp.cos(q_3))*jnp.sin(q_5))*jnp.sin(q_6) + 0.088*(((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) + jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4))*jnp.cos(q_5) - (jnp.sin(q_1)*jnp.sin(q_3)*jnp.cos(q_2) - jnp.cos(q_1)*jnp.cos(q_3))*jnp.sin(q_5))*jnp.cos(q_6) - 0.088*((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4))*jnp.sin(q_6) + 0.107*((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4))*jnp.cos(q_6) - 0.384*(jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - 0.0825*(jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) - 0.0825*jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4) + 0.384*jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4) + 0.316*jnp.sin(q_1)*jnp.sin(q_2) + 0.0825*jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + 0.0825*jnp.sin(q_3)*jnp.cos(q_1)

    z = -0.107*((jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - jnp.sin(q_4)*jnp.cos(q_2))*jnp.cos(q_5) - jnp.sin(q_2)*jnp.sin(q_3)*jnp.sin(q_5))*jnp.sin(q_6) - 0.088*((jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - jnp.sin(q_4)*jnp.cos(q_2))*jnp.cos(q_5) - jnp.sin(q_2)*jnp.sin(q_3)*jnp.sin(q_5))*jnp.cos(q_6) + 0.088*(jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + jnp.cos(q_2)*jnp.cos(q_4))*jnp.sin(q_6) - 0.107*(jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + jnp.cos(q_2)*jnp.cos(q_4))*jnp.cos(q_6) + 0.384*jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + 0.0825*jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - 0.0825*jnp.sin(q_2)*jnp.cos(q_3) - 0.0825*jnp.sin(q_4)*jnp.cos(q_2) + 0.384*jnp.cos(q_2)*jnp.cos(q_4) + 0.316*jnp.cos(q_2) + 0.33

    cq1 = jnp.cos(q_1)
    cq2 = jnp.cos(q_2)
    cq3 = jnp.cos(q_3)
    cq4 = jnp.cos(q_4)
    cq5 = jnp.cos(q_5)
    cq6 = jnp.cos(q_6)
    cq7 = jnp.cos(q_7)

    sq1 = jnp.sin(q_1)
    sq2 = jnp.sin(q_2)
    sq3 = jnp.sin(q_3)
    sq4 = jnp.sin(q_4)
    sq5 = jnp.sin(q_5)
    sq6 = jnp.sin(q_6)
    sq7 = jnp.sin(q_7)


    r_32 = -cq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2)) - sq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4))
    r_33 = -cq6*(cq2*cq4 + cq3*sq2*sq4) + sq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5)
    r_31 = cq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4)) - sq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2))
    r_21 = cq7*(cq6*(cq5*(cq4*(cq1*sq3 + cq2*cq3*sq1) + sq1*sq2*sq4) + sq5*(cq1*cq3 - cq2*sq1*sq3)) + sq6*(cq4*sq1*sq2 - sq4*(cq1*sq3 + cq2*cq3*sq1))) - sq7*(cq5*(cq1*cq3 - cq2*sq1*sq3) - sq5*(cq4*(cq1*sq3 + cq2*cq3*sq1) + sq1*sq2*sq4))
    r_11 = cq7*(cq6*(cq5*(cq1*sq2*sq4 + cq4*(cq1*cq2*cq3 - sq1*sq3)) - sq5*(cq1*cq2*sq3 + cq3*sq1)) + sq6*(cq1*cq4*sq2 - sq4*(cq1*cq2*cq3 - sq1*sq3))) + sq7*(cq5*(cq1*cq2*sq3 + cq3*sq1) + sq5*(cq1*sq2*sq4 + cq4*(cq1*cq2*cq3 - sq1*sq3)))

    roll = jnp.arctan2(r_32, r_33)
    pitch = -jnp.arcsin(r_31)
    yaw =  jnp.arctan2(r_21, r_11)

    return x, y, z, roll, pitch, yaw




###### c_g takes in q/theta and gives u squared error
def compute_potential_cost(q):

    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    q_5 = q[4]
    q_6 = q[5]
    q_7 = q[6]

    x = -0.107*(((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1))*jnp.cos(q_5) + (jnp.sin(q_1)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1)*jnp.cos(q_2))*jnp.sin(q_5))*jnp.sin(q_6) - 0.088*(((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1))*jnp.cos(q_5) + (jnp.sin(q_1)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1)*jnp.cos(q_2))*jnp.sin(q_5))*jnp.cos(q_6) + 0.088*((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4))*jnp.sin(q_6) - 0.107*((jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4))*jnp.cos(q_6) + 0.384*(jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.sin(q_4) + 0.0825*(jnp.sin(q_1)*jnp.sin(q_3) - jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3))*jnp.cos(q_4) - 0.0825*jnp.sin(q_1)*jnp.sin(q_3) - 0.0825*jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_1) + 0.384*jnp.sin(q_2)*jnp.cos(q_1)*jnp.cos(q_4) + 0.316*jnp.sin(q_2)*jnp.cos(q_1) + 0.0825*jnp.cos(q_1)*jnp.cos(q_2)*jnp.cos(q_3)

    y = 0.107*(((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) + jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4))*jnp.cos(q_5) - (jnp.sin(q_1)*jnp.sin(q_3)*jnp.cos(q_2) - jnp.cos(q_1)*jnp.cos(q_3))*jnp.sin(q_5))*jnp.sin(q_6) + 0.088*(((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) + jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4))*jnp.cos(q_5) - (jnp.sin(q_1)*jnp.sin(q_3)*jnp.cos(q_2) - jnp.cos(q_1)*jnp.cos(q_3))*jnp.sin(q_5))*jnp.cos(q_6) - 0.088*((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4))*jnp.sin(q_6) + 0.107*((jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4))*jnp.cos(q_6) - 0.384*(jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.sin(q_4) - 0.0825*(jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + jnp.sin(q_3)*jnp.cos(q_1))*jnp.cos(q_4) - 0.0825*jnp.sin(q_1)*jnp.sin(q_2)*jnp.sin(q_4) + 0.384*jnp.sin(q_1)*jnp.sin(q_2)*jnp.cos(q_4) + 0.316*jnp.sin(q_1)*jnp.sin(q_2) + 0.0825*jnp.sin(q_1)*jnp.cos(q_2)*jnp.cos(q_3) + 0.0825*jnp.sin(q_3)*jnp.cos(q_1)

    z = -0.107*((jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - jnp.sin(q_4)*jnp.cos(q_2))*jnp.cos(q_5) - jnp.sin(q_2)*jnp.sin(q_3)*jnp.sin(q_5))*jnp.sin(q_6) - 0.088*((jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - jnp.sin(q_4)*jnp.cos(q_2))*jnp.cos(q_5) - jnp.sin(q_2)*jnp.sin(q_3)*jnp.sin(q_5))*jnp.cos(q_6) + 0.088*(jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + jnp.cos(q_2)*jnp.cos(q_4))*jnp.sin(q_6) - 0.107*(jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + jnp.cos(q_2)*jnp.cos(q_4))*jnp.cos(q_6) + 0.384*jnp.sin(q_2)*jnp.sin(q_4)*jnp.cos(q_3) + 0.0825*jnp.sin(q_2)*jnp.cos(q_3)*jnp.cos(q_4) - 0.0825*jnp.sin(q_2)*jnp.cos(q_3) - 0.0825*jnp.sin(q_4)*jnp.cos(q_2) + 0.384*jnp.cos(q_2)*jnp.cos(q_4) + 0.316*jnp.cos(q_2) + 0.33

    cq1 = jnp.cos(q_1)
    cq2 = jnp.cos(q_2)
    cq3 = jnp.cos(q_3)
    cq4 = jnp.cos(q_4)
    cq5 = jnp.cos(q_5)
    cq6 = jnp.cos(q_6)
    cq7 = jnp.cos(q_7)

    sq1 = jnp.sin(q_1)
    sq2 = jnp.sin(q_2)
    sq3 = jnp.sin(q_3)
    sq4 = jnp.sin(q_4)
    sq5 = jnp.sin(q_5)
    sq6 = jnp.sin(q_6)
    sq7 = jnp.sin(q_7)


    r_32 = -cq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2)) - sq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4))
    r_33 = -cq6*(cq2*cq4 + cq3*sq2*sq4) + sq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5)
    r_31 = cq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4)) - sq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2))
    r_21 = cq7*(cq6*(cq5*(cq4*(cq1*sq3 + cq2*cq3*sq1) + sq1*sq2*sq4) + sq5*(cq1*cq3 - cq2*sq1*sq3)) + sq6*(cq4*sq1*sq2 - sq4*(cq1*sq3 + cq2*cq3*sq1))) - sq7*(cq5*(cq1*cq3 - cq2*sq1*sq3) - sq5*(cq4*(cq1*sq3 + cq2*cq3*sq1) + sq1*sq2*sq4))
    r_11 = cq7*(cq6*(cq5*(cq1*sq2*sq4 + cq4*(cq1*cq2*cq3 - sq1*sq3)) - sq5*(cq1*cq2*sq3 + cq3*sq1)) + sq6*(cq1*cq4*sq2 - sq4*(cq1*cq2*cq3 - sq1*sq3))) + sq7*(cq5*(cq1*cq2*sq3 + cq3*sq1) + sq5*(cq1*sq2*sq4 + cq4*(cq1*cq2*cq3 - sq1*sq3)))

    roll = jnp.arctan2(r_32, r_33)
    pitch = -jnp.arcsin(r_31)
    yaw =  jnp.arctan2(r_21, r_11)

    # jnp.sum( (jnp.arctan2(r_32, r_33)-roll_des)**2+(-jnp.arcsin(r_31)-pitch_des)**2    )

    cost = (x-x_f)**2+(y-y_f)**2+(z-z_f)**2+(roll-roll_f)**2+(pitch-pitch_f)**2+(yaw-yaw_f)**2
    # cost = (x-x_f)**2+(y-y_f)**2+(z-z_f)**2


    return cost


maxiter = 3000
num_dof = 7

q_min = np.array([-165, -100, -165, -165, -165, -1.0, -165  ])*np.pi/180
q_max = np.array([ 165,   101,  165,  1.0,    165, 214, 165  ])*np.pi/180

q_init = (q_min+q_max)/2.0

x_init, y_init, z_init, roll_init, pitch_init, yaw_init = compute_fk(q_init)

var = 80*np.pi/180.0

q_check = np.random.uniform(q_init-var, q_init+var, (7,))

# q_test = np.random.uniform(q_min, q_max, (7,))
q_test = np.clip(q_check, q_min, q_max)

grad_fun_jit = jit(grad(compute_potential_cost))
compute_fk_jit = jit(compute_fk)
compute_cost_jit = jit(compute_potential_cost)

x_f, y_f, z_f, roll_f, pitch_f, yaw_f = compute_fk(jnp.asarray(q_test))

roll_f = 0.0
pitch_f = 0.0
yaw_f = 0.0
# print(x_f, y_f, z_f, roll_f, pitch_f)

q = jnp.asarray((q_min+q_max)/2.0)
# q = jnp.zeros(num_dof)
q_traj = np.ones((maxiter, num_dof))
x_traj = np.ones((maxiter, 3))
orient_traj = np.ones((maxiter, 3))
cost_track = np.ones(maxiter)
delt = 0.005

for i in range(0, maxiter):
    grad_vec = grad_fun_jit(q)
    q = q-delt*grad_vec
    q = jnp.clip(q, jnp.asarray(q_min),  jnp.asarray(q_max))
    x_traj[i, 0], x_traj[i, 1], x_traj[i, 2], orient_traj[i,0], orient_traj[i,1], orient_traj[i,2]  = compute_fk_jit(q)
    cost_track[i] = compute_cost_jit(q)
    q_traj[i] = q



scipy.io.savemat('frank_joint_traj.mat', {'q_traj': q_traj }) ########## matrix of y position of the vehicle



plt.figure(1)
plt.plot(cost_track)

fig = plt.figure(2)
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_traj[:,0], x_traj[:,1], x_traj[:,2], 'om', markersize = 7.0)
ax.plot(x_f*np.ones(1), y_f*np.ones(1), z_f*np.ones(1), 'ok', markersize = 10.0)
ax.plot(x_init*np.ones(1), y_init*np.ones(1), z_init*np.ones(1), 'ob', markersize = 10.0)
ax.set_xlim3d(-1.0, 1.0)
ax.set_ylim3d(-1.0, 1.0)
ax.set_zlim3d(0, 1.2)


plt.show()
