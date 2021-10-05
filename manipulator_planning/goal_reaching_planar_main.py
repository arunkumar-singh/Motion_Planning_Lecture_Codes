
####### goal_reaching_planar

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax import jit, grad
from jax.config import config; config.update("jax_enable_x64", True)


def compute_goal_potential( q  ):
    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]

    x_e = l_1*jnp.cos(q_1)+l_2*jnp.cos(q_1+q_2)+l_3*jnp.cos(q_1+q_2+q_3)
    y_e = l_1*jnp.sin(q_1)+l_2*jnp.sin(q_1+q_2)+l_3*jnp.sin(q_1+q_2+q_3)

    goal_potential = (x_e-x_f)**2+(y_e-y_f)**2

    return goal_potential

def compute_fk(q):
    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]

    x_e = l_1*jnp.cos(q_1)+l_2*jnp.cos(q_1+q_2)+l_3*jnp.cos(q_1+q_2+q_3)
    y_e = l_1*jnp.sin(q_1)+l_2*jnp.sin(q_1+q_2)+l_3*jnp.sin(q_1+q_2+q_3)

    return x_e, y_e

l_1 = 1.5
l_2 = 1.5
l_3 = 1.5

# x_f = 4.0
# y_f = 3.5


### goal_reaching_planar.py

# x_f = np.random.uniform(-4.0, 4.0)
# y_f = np.random.uniform(-4.0, 4.0)

x_f = -0.45
y_f =  -0.59


maxiter = 1000
x_traj = np.ones(maxiter)
y_traj = np.ones(maxiter)

# q_init = jnp.zeros(3)
# q_init = jnp.ones(3)*jnp.pi/2
q_init = jnp.hstack((0.1, 0.0, 0.0   ))
# q_init = jnp.asarray(np.random.uniform(-2.1, 2.1 , 3   )   )
q_traj = np.zeros((maxiter, 3))

man_x_init = np.hstack((0.0, l_1*np.cos(q_init[0]), l_1*np.cos(q_init[0])+l_2*np.cos(q_init[0]+q_init[1]), l_1*np.cos(q_init[0])+l_2*np.cos(q_init[0]+q_init[1])+l_3*np.cos(q_init[0]+q_init[1]+q_init[2])   ))
man_y_init = np.hstack((0.0, l_1*np.sin(q_init[0]), l_1*np.sin(q_init[0])+l_2*np.sin(q_init[0]+q_init[1]), l_1*np.sin(q_init[0])+l_2*np.sin(q_init[0]+q_init[1])+l_3*np.sin(q_init[0]+q_init[1]+q_init[2])   ))


grad_fun = jit(grad(compute_goal_potential))
fk_fun = jit(compute_fk)
goal_potential_fun = jit(compute_goal_potential)
cost_track = np.ones(maxiter-1)
x_traj[0], y_traj[0] = fk_fun(q_init)

delt = 0.001
for i in range(1, maxiter):
    grad_vec = grad_fun(q_init)
    q_init = q_init-delt*grad_vec
    x_traj[i], y_traj[i] = fk_fun(q_init)
    cost_track[i-1] = goal_potential_fun(q_init)
    q_traj[i] = q_init




print(man_x_init, man_y_init, q_init[0])

q_fin = q_traj[-1]

man_x_fin = np.hstack((0.0, l_1*np.cos(q_fin[0]), l_1*np.cos(q_fin[0])+l_2*np.cos(q_fin[0]+q_fin[1]), l_1*np.cos(q_fin[0])+l_2*np.cos(q_fin[0]+q_fin[1])+l_3*np.cos(q_fin[0]+q_fin[1]+q_fin[2])   ))
man_y_fin = np.hstack((0.0, l_1*np.sin(q_fin[0]), l_1*np.sin(q_fin[0])+l_2*np.sin(q_fin[0]+q_fin[1]), l_1*np.sin(q_fin[0])+l_2*np.sin(q_fin[0]+q_fin[1])+l_3*np.sin(q_fin[0]+q_fin[1]+q_fin[2])   ))




plt.figure(1)
plt.plot(x_traj, y_traj, '-r', linewidth = 3.0)
plt.plot(x_f*np.ones(1), y_f*np.ones(1), 'og', markersize = 10.0)
plt.plot(x_traj[0]*np.ones(1), y_traj[0]*np.ones(1), 'om', markersize = 10.0)
plt.plot(man_x_init, man_y_init, '-ob', linewidth = 3.0, markersize = 8.0)
plt.plot(man_x_fin, man_y_fin, '-ok', linewidth = 3.0, markersize = 8.0)

plt.axis('equal')


plt.figure(2)
plt.plot(cost_track)
plt.show()
