




import jax.numpy as jnp
import numpy as np
from jax import jit, grad

import matplotlib.pyplot as plt
import time
import scipy.io

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



def compute_potential( p  ):
    x = p[0]
    y = p[1]

    # goal_potential = 0.5*((x-x_g)**2+(y-y_g)**2) ### c_g

    goal_potential = 0.5*jnp.sqrt((x-x_g)**2+(y-y_g)**2)
    dist_obs = jnp.sqrt((x-x_o)**2+(y-y_o)**2)


    # obstacle_potential = eta*((1/dist_obs)-(1/d_o))**2

    obstacle_potential = jnp.maximum(0, -dist_obs+d_o   )**2
    # obstacle_potential = 1/(1+jnp.exp(-dist_obs+d_o))



    total_potential = goal_potential+10*obstacle_potential

    return total_potential


def compute_potential_obsfree(p):
    x = p[0]
    y = p[1]

    goal_potential = 0.5*((x-x_g)**2+(y-y_g)**2) ## c_g

    return goal_potential




x_init = 0.0
y_init = 0.0

x_g = 6.0
y_g = 6.0


x_o = 3.1
y_o = 3.0

eta = 1000.0

# eta = 10.0
# eta = 0.001


d_o = 2.7

obs_rad = 1.5

potential_grad = jit(grad(compute_potential))

potential_grad_obsfree = jit(grad(compute_potential_obsfree))


# grad_x = potential_grad(jnp.hstack(( x_init, y_init )))



maxiter = 6600

x_traj = np.ones(maxiter)*x_init
y_traj = np.ones(maxiter)*y_init

v = 0.1

delt = 0.1

for i in range(1, maxiter):

    start = time.time()
    dist_obs = jnp.sqrt((x_init-x_o)**2+(y_init-y_o)**2)
    if(dist_obs<=d_o):

        grad_x = potential_grad(  jnp.hstack(( x_init, y_init  ))      )


    if(dist_obs>d_o):

        grad_x = potential_grad_obsfree(  jnp.hstack(( x_init, y_init  ))      )


    vx, vy = -v*grad_x


    x_init = x_init+vx*delt
    y_init = y_init+vy*delt

    # print(vx, vy)

    x_traj[i] = x_init
    y_traj[i] = y_init

    # eta = jnp.max( jnp.hstack(( 1/dist_obs, 500.0)))


    print(time.time()-start)


# scipy.io.savemat('potential_path_pf.mat', {'x': x_traj[0:i], 'y': y_traj[0:i]   }) ########### matrix for x position of the vehicle
# scipy.io.savemat('y_pf.mat', {'y': y_traj[0:i]}) ########## matrix of y position of the vehicle


th = np.linspace(0, 2*np.pi, 100)
x_obs = x_o+obs_rad*np.cos(th)
y_obs = y_o+obs_rad*np.sin(th)


############################################################################# Visualization over cost surface

###### path_planning_pf.py


num_samples = 100


x_workspace = np.linspace(-2.0, 6.0, num_samples)
y_workspace = np.linspace(-2.0, 6.0, num_samples)

x_grid, y_grid = np.meshgrid( x_workspace, y_workspace)

# goal_potential = 0.5*((x_grid-x_g)**2+(y_grid-y_g)**2)
goal_potential = 0.5*jnp.sqrt((x-x_g)**2+(y-y_g)**2)



# d_0 = 1.5

dist_obs = np.sqrt((x_grid-x_o)**2+(y_grid-y_o)**2)

# dist_obs = sqrt((x_grid-x_o)**2+(y_grid-y_o)**2)


# eta = 0.03
# eta = 30.0
# obstacle_potential = maximum( zeros(( num_samples, num_samples )),  0.5*eta*((1/dist_obs)-(1/d_0))**2 )

# obstacle_potential = 0.5*eta*((1/dist_obs)-(1/d_0))**2 

# obstacle_potential = np.minimum(eta*((1/dist_obs)-(1/d_o))**2, 40.0*np.ones((num_samples, num_samples  ))  ) # c_o

obstacle_potential = np.maximum(0, -dist_obs+d_o   )

# obstacle_potential = 10/(1+np.exp(-dist_obs+d_o))

# obstacle_potential = eta*np.maximum( np.zeros(( num_samples, num_samples  )), dist_obs         )


combined_potential = obstacle_potential+goal_potential



plt.figure(1)
plt.plot(x_traj, y_traj)
plt.plot(x_g*np.ones(1), y_g*np.ones(1), 'og', markersize = 3.0)
plt.plot(x_obs, y_obs, '-k' )
plt.axis('equal')


z_traj = 0.5*((x_traj-x_g)**2+(y_traj-y_g)**2)





fig_3 = plt.figure(3)
ax_3 = fig_3.gca(projection='3d')
# ax_3.set_zlim(0.0, 3.01)
ax_3.set_xlim(-2.0, 6.01)
ax_3.set_ylim(-2.0, 6.01)

ax_3.plot_surface(x_grid, y_grid, combined_potential, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha = 0.5)


ax_3.plot(x_traj, y_traj, z_traj, '-k', linewidth = 3.0)
ax_3.plot(x_traj[0]*np.ones(1), y_traj[0]*np.ones(1), z_traj[0]*np.ones(1), 'og', markersize = 10.0)

ax_3.plot(x_traj[-1]*np.ones(1), y_traj[-1]*np.ones(1), z_traj[-1]*np.ones(1), 'om', markersize = 10.0)





plt.show()
