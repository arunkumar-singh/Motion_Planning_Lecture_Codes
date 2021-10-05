

import numpy as  np
import jax.numpy as jnp
import matplotlib.pyplot as plt 
from jax import grad, jit

from scipy.io import loadmat
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


path_data = loadmat('potential_path_pf.mat')
x_pf = path_data['x'][0]
y_pf = path_data['y'][0]



def compute_potential( p  ):
    
	x = p[0]
	y = p[1]

	goal_potential = 0.5*((x-x_g)**2+(y-y_g)**2) ### c_g
	# goal_potential = 0.5*jnp.sqrt((x-x_g)**2+(y-y_g)**2)

	dist_obs = jnp.sqrt((x-x_o)**2+(y-y_o)**2)

	


	obstacle_potential = eta*((1/dist_obs)-(1/d_o))**2

	# obstacle_potential = jnp.min(jnp.hstack(( eta*((1/dist_obs)-(1/d_o))**2, 40.0) ) ) # c_o


	# obstacle_potential = jnp.maximum(0.0, -dist_obs+d_o   )
	# smooth_potential = (x-2*x_traj[i-1]-x_traj[i-2])**2+(y-2*y_traj[i-1]-y_traj[i-2])**2

	total_potential = goal_potential+obstacle_potential

	return total_potential


def compute_potential_clipped( p  ):
    
	x = p[0]
	y = p[1]

	goal_potential = 0.5*((x-x_g)**2+(y-y_g)**2) ### c_g
	# goal_potential = 0.5*jnp.sqrt((x-x_g)**2+(y-y_g)**2)


	dist_obs = jnp.sqrt((x-x_o)**2+(y-y_o)**2)

	


	# obstacle_potential = eta*((1/dist_obs)-(1/d_o))**2

	obstacle_potential = jnp.min(jnp.hstack(( eta*((1/dist_obs)-(1/d_o))**2, 50.0) ) ) # c_o


	# obstacle_potential = jnp.maximum(0.0, -dist_obs+d_o   )
	# smooth_potential = (x-2*x_traj[i-1]-x_traj[i-2])**2+(y-2*y_traj[i-1]-y_traj[i-2])**2

	total_potential = goal_potential+obstacle_potential

	return total_potential



def compute_potential_obsfree(p):
    x = p[0]
    y = p[1]

    goal_potential = 0.5*((x-x_g)**2+(y-y_g)**2) ## c_g
    # goal_potential = 0.5*jnp.sqrt((x-x_g)**2+(y-y_g)**2)


    return goal_potential


x_init = 0.0
y_init = 0.0

x_g = 6.0
y_g = 6.0


x_o_1 = 3.1
y_o_1 = 3.0

x_o_2 = 3
y_o_2 = -2.0

x_o_3 = -1.0
y_o_3 = -2.0

x_o_4 = -2.0
y_o_4 = 3.0



x_o_vec = np.hstack(( x_o_1, x_o_2, x_o_3, x_o_4   ))
y_o_vec = np.hstack(( y_o_1, y_o_2, y_o_3, y_o_4   ))


eta = 10**3

# eta = 10.0
# eta = 0.001

d_o = 1.7

num_samples = 30

potential_grad = jit(grad(compute_potential))
potential_grad_obsfree = jit(grad(compute_potential_obsfree))
potential = jit(compute_potential_clipped)
potential_obsfree = jit(compute_potential_obsfree)


potential_matrix = np.zeros(( num_samples, num_samples  ))


x_workspace = np.linspace(-6.1, 6.1, num_samples)
y_workspace = np.linspace(-6.1, 6.1, num_samples)

x_grid, y_grid = np.meshgrid( x_workspace, y_workspace)

x_flow = np.zeros(( num_samples, num_samples  ))

y_flow = np.zeros(( num_samples, num_samples  ))

for i in range(0, num_samples):
	for j in range(0, num_samples):



		# dist_obs = np.sqrt((x_grid[i, j]-x_o)**2+(y_grid[i, j]-y_o)**2)

		dist_obs = np.sqrt((x_grid[i, j]-x_o_vec)**2+(y_grid[i, j]-y_o_vec)**2)

		min_idx = np.argmin(dist_obs)


		# print(dist_obs[min_idx])
		# kk



		if(dist_obs[min_idx]>d_o):
			x_o = x_o_vec[min_idx]
			y_o = y_o_vec[min_idx]


			x_flow[i, j ], y_flow[i, j] = potential_grad_obsfree(jnp.hstack(( x_grid[i, j], y_grid[i, j])) )
			potential_matrix[i, j] = potential_obsfree(jnp.hstack(( x_grid[i, j], y_grid[i, j])))

		if(dist_obs[min_idx]<=d_o):	
			x_o = x_o_vec[min_idx]
			y_o = y_o_vec[min_idx]
			
			x_flow[i, j ], y_flow[i, j] = potential_grad(jnp.hstack(( x_grid[i, j], y_grid[i, j])) )
			
			potential_matrix[i, j] = potential(jnp.hstack(( x_grid[i, j], y_grid[i, j])))

		x_flow[i, j] = -x_flow[i, j]/jnp.sqrt( x_flow[i, j]**2+y_flow[i, j]**2  )
		y_flow[i, j] = -y_flow[i, j]/jnp.sqrt( x_flow[i, j]**2+y_flow[i, j]**2  )

		# x_flow[i, j] = -x_flow[i, j]
		# y_flow[i, j] = -y_flow[i, j]


th = np.linspace(0, 2*np.pi, 100)
x_obs = x_o+d_o*np.cos(th)
y_obs = y_o+d_o*np.sin(th)

x_obs_2 = x_o_2+d_o*np.cos(th)
y_obs_2 = y_o_2+d_o*np.sin(th)

x_obs_3 = x_o_3+d_o*np.cos(th)
y_obs_3 = y_o_3+d_o*np.sin(th)


x_obs_4 = x_o_4+d_o*np.cos(th)
y_obs_4 = y_o_4+d_o*np.sin(th)


step = 1
scale = 1

fig, ax = plt.subplots(figsize=(20, 20))
ax.streamplot(x_grid, y_grid, x_flow, y_flow, linewidth = 2.0)
plt.plot(x_obs, y_obs, '-k', linewidth = 3.0)
plt.plot(x_obs_2, y_obs_2, '-k', linewidth = 3.0)
plt.plot(x_obs_3, y_obs_3, '-k', linewidth = 3.0)
plt.plot(x_obs_4, y_obs_4, '-k', linewidth = 3.0)




plt.plot(x_g*np.ones(1), y_g*np.ones(1), 'om', markersize = 10)
# plt.plot(x_pf[-1]*np.ones(1), y_pf[-1]*np.ones(1), 'om', markersize = 10)
# plt.plot(x_pf[0]*np.ones(1), y_pf[0]*np.ones(1), 'og', markersize = 10)

# plt.plot(x_pf, y_pf, '-r', linewidth = 2.0)
plt.axis('square')




# plt.figure(2)
# im = plt.imshow(potential_matrix, cmap=plt.cm.RdBu, extent=(-7, 7, 7, -7), interpolation='bilinear')
# c= plt.pcolormesh(X, Y, cost_matrix_first_obs)
# plt.colorbar(im)
# # plt.plot(x_traj, y_traj, '-k', linewidth = 3.0)
# plt.plot(x_g*np.ones(1), y_g*np.ones(1), 'om', markersize = 20.0)
# # plt.plot(x_f*ones(1), y_f*ones(1), 'om', markersize = 10.0)

figure_2 = plt.figure(2)
ax_2 = plt.axes()

# sns.heatmap(cost_matrix_second_obs, xticklabels=False, yticklabels=False)
c= plt.pcolormesh(x_grid, y_grid, potential_matrix)
# c= plt.pcolormesh(X, Y, cost_matrix_second_obs, cmap="RdBu")
ax_2.axis([-6.1,6.1, -6.1, 6.1])
figure_2.colorbar(c, ax=ax_2)  
plt.plot(x_g*np.ones(1), y_g*np.ones(1), 'om', markersize = 20.0)


fig_3 = plt.figure(3)
ax_3 = fig_3.gca(projection='3d')
# ax_3.set_zlim(0.0, 3.01)
ax_3.set_xlim(-6.1, 6.1)
ax_3.set_ylim(-6.1, 6.1)

ax_3.plot_surface(x_grid, y_grid, potential_matrix, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


plt.show()

