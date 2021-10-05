

from numpy import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# goal_reaching_cost_visualization.py

num_samples = 100

x_0 = -6.0
y_0 = -6.0

x_f = 3.0
y_f = 3.0


x_o = 1.0
y_o = 1.0


x_workspace = linspace(-6.0, 6.0, num_samples)
y_workspace = linspace(-6.0, 6.0, num_samples)

x_grid, y_grid = meshgrid( x_workspace, y_workspace)

goal_potential = 0.5*((x_grid-x_f)**2+(y_grid-y_f)**2)

d_0 = 1.5

dist_obs = -sqrt((x_grid-x_o)**2+(y_grid-y_o)**2)+d_0

# dist_obs = sqrt((x_grid-x_o)**2+(y_grid-y_o)**2)


# eta = 0.03
eta = 30.0
# obstacle_potential = maximum( zeros(( num_samples, num_samples )),  0.5*eta*((1/dist_obs)-(1/d_0))**2 )

# obstacle_potential = 0.5*eta*((1/dist_obs)-(1/d_0))**2 


obstacle_potential = eta*maximum( zeros(( num_samples, num_samples  )), dist_obs         )


combined_potential = obstacle_potential+goal_potential

x_traj = linspace(x_0, x_f, num_samples)
y_traj = linspace(x_0, y_f, num_samples)

z_traj = 0.5*((x_traj-x_f)**2+(y_traj-y_f)**2)


fig = plt.figure(1)
ax = fig.gca(projection='3d')
# ax.set_zlim(0.00, 30.01)
surf = ax.plot_surface(x_grid, y_grid, goal_potential, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha = 0.2)

ax.plot(x_traj, y_traj, z_traj, '-k', linewidth = 3.0)
ax.plot(x_traj[0]*ones(1), y_traj[0]*ones(1), z_traj[0]*ones(1), 'og', markersize = 10.0)
ax.plot(x_traj[-1]*ones(1), y_traj[-1]*ones(1), z_traj[-1]*ones(1), 'om', markersize = 10.0)

plt.xlabel('x')
plt.ylabel('y')




# plt.figure(2)
# im = plt.imshow(goal_potential, cmap=plt.cm.RdBu, extent=(-6, 6, 6, -6), interpolation='bilinear')
# plt.colorbar(im)
# plt.plot(x_traj, y_traj, '-k', linewidth = 3.0)
# plt.plot(x_0*ones(1), y_0*ones(1), 'og', markersize = 10.0)
# plt.plot(x_f*ones(1), y_f*ones(1), 'om', markersize = 10.0)


figure_2 = plt.figure(2)
ax_2 = plt.axes()

# sns.heatmap(cost_matrix_second_obs, xticklabels=False, yticklabels=False)
c= plt.pcolormesh(x_grid, y_grid, goal_potential)
# c= plt.pcolormesh(X, Y, cost_matrix_second_obs, cmap="RdBu")
ax_2.axis([-6,6, -6, 6])
figure_2.colorbar(c, ax=ax_2)  
plt.plot(x_f*ones(1), y_f*ones(1), 'om', markersize = 20.0)
plt.plot(x_0*ones(1), y_0*ones(1), 'og', markersize = 10.0)


# plt.title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')



fig_3 = plt.figure(3)
ax_3 = fig_3.gca(projection='3d')
# ax_3.set_zlim(0.0, 3.01)
ax_3.set_xlim(-2.0, 2.01)
ax_3.set_ylim(-2.0, 2.01)

ax_3.plot_surface(x_grid, y_grid, obstacle_potential, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


fig_4 = plt.figure(4)
ax_4 = fig_4.gca(projection='3d')
# ax_3.set_zlim(0.0, 3.01)
ax_4.set_ylim(-6.0, 6.01)
ax_4.set_xlim(-6.0, 6.01)

ax_4.plot_surface(x_grid, y_grid, combined_potential, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, alpha = 0.6)

ax_4.plot(x_traj, y_traj, z_traj, '-k', linewidth = 3.0)
ax_4.plot(x_traj, y_traj, z_traj, '-k', linewidth = 3.0)
ax_4.plot(x_traj[0]*ones(1), y_traj[0]*ones(1), z_traj[0]*ones(1), 'og', markersize = 10.0)
ax_4.plot(x_traj[-1]*ones(1), y_traj[-1]*ones(1), z_traj[-1]*ones(1), 'om', markersize = 10.0)

plt.xlabel('x')
plt.ylabel('y')

plt.figure(5)
im = plt.imshow(combined_potential, cmap=plt.cm.RdBu, extent=(-6, 6, 6, -6), interpolation='bilinear')
plt.colorbar(im)
plt.plot(x_traj, y_traj, '-k', linewidth = 3.0)
plt.plot(x_0*ones(1), y_0*ones(1), 'og', linewidth = 3.0)
plt.plot(x_f*ones(1), y_f*ones(1), 'om', linewidth = 3.0)


figure_5 = plt.figure(5)
ax_5 = plt.axes()

# sns.heatmap(cost_matrix_second_obs, xticklabels=False, yticklabels=False)
c= plt.pcolormesh(x_grid, y_grid, combined_potential)
# c= plt.pcolormesh(X, Y, cost_matrix_second_obs, cmap="RdBu")
ax_5.axis([-6,6, -6, 6])
figure_5.colorbar(c, ax=ax_2)  
plt.plot(x_f*ones(1), y_f*ones(1), 'om', markersize = 20.0)
plt.plot(x_0*ones(1), y_0*ones(1), 'og', markersize = 10.0)

plt.figure(6)
plt.contour(combined_potential, linspace(combined_potential.min(),combined_potential.max(),25)) 

plt.show()
