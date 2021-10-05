

import numpy as  np
import jax.numpy as jnp
import matplotlib.pyplot as plt 
from jax import grad, jit

from scipy.io import loadmat

path_data = loadmat('potential_path_pf.mat')
x_pf = path_data['x'][0]
y_pf = path_data['y'][0]



def compute_potential( p  ):
    x = p[0]
    y = p[1]

    goal_potential = 0.5*((x-x_g)**2+(y-y_g)**2) ### c_g

    dist_obs = jnp.sqrt((x-x_o)**2+(y-y_o)**2)

    obstacle_potential = eta*((1/dist_obs)-(1/d_o))**2

    # obstacle_potential = jnp.min(jnp.hstack(( eta*((1/dist_obs)-(1/d_o))**2, 40.0) ) ) # c_o
    # obstacle_potential = jnp.max(0.0, -dist_obs+d_o   )

    # smooth_potential = (x-2*x_traj[i-1]-x_traj[i-2])**2+(y-2*y_traj[i-1]-y_traj[i-2])**2

    total_potential = goal_potential+obstacle_potential

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

num_samples = 30

potential_grad = jit(grad(compute_potential))
potential_grad_obsfree = jit(grad(compute_potential_obsfree))




x_workspace = np.linspace(-6.0, 6.0, num_samples)
y_workspace = np.linspace(-6.0, 6.0, num_samples)

x_grid, y_grid = np.meshgrid( x_workspace, y_workspace)

x_flow = np.zeros(( num_samples, num_samples  ))

y_flow = np.zeros(( num_samples, num_samples  ))

for i in range(0, num_samples):
	for j in range(0, num_samples):

		dist_obs = np.sqrt((x_grid[i, j]-x_o)**2+(y_grid[i, j]-y_o)**2)
		if(dist_obs>d_o):

			x_flow[i, j ], y_flow[i, j] = potential_grad_obsfree(jnp.hstack(( x_grid[i, j], y_grid[i, j])) )

		if(dist_obs<d_o):	
			
			x_flow[i, j ], y_flow[i, j] = potential_grad(jnp.hstack(( x_grid[i, j], y_grid[i, j])) )

		# x_flow[i, j] = -x_flow[i, j]/jnp.sqrt( x_flow[i, j]**2+y_flow[i, j]**2  )
		# y_flow[i, j] = -y_flow[i, j]/jnp.sqrt( x_flow[i, j]**2+y_flow[i, j]**2  )

		x_flow[i, j] = -x_flow[i, j]
		y_flow[i, j] = -y_flow[i, j]
		




th = np.linspace(0, 2*np.pi, 100)
x_obs = x_o+d_o*np.cos(th)
y_obs = y_o+d_o*np.sin(th)

step = 1
scale = 1

fig, ax = plt.subplots(figsize=(20, 20))
ax.streamplot(x_grid, y_grid, x_flow, y_flow, linewidth = 2.0)
plt.plot(x_obs, y_obs, '-k', linewidth = 3.0)
# plt.plot(x_g*np.ones(1), y_g*np.ones(1), 'om', markersize = 10)
plt.plot(x_pf[-1]*np.ones(1), y_pf[-1]*np.ones(1), 'om', markersize = 10)
plt.plot(x_pf[0]*np.ones(1), y_pf[0]*np.ones(1), 'og', markersize = 10)

plt.plot(x_pf, y_pf, '-r', linewidth = 2.0)
plt.axis('square')

plt.show()

