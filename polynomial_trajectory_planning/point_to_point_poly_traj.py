

import numpy as np
import matplotlib.pyplot as plt

x_0 = 0.0
y_0 = 0.0
xdot_0 = 8.0
ydot_0 = 0.0

xddot_0 = 0.0
yddot_0 = 0.0

x_f = np.hstack(( 100.0, 100.0, 100.0, 100.0, 80, 80.0, 80.0, 80.0, 80.0, 80   ))
y_f = np.hstack(( 5.0, 2.0, 0.0, -2.0, -5.0,  5.0, 2.0, 0.0, -2.0, -5.0    ))

xdot_f = 8.0*np.ones(np.shape(x_f)[0])
ydot_f = 0.0*np.ones(np.shape(x_f)[0])


xddot_f = 0.0*np.ones(np.shape(x_f)[0])
yddot_f = 0.0*np.ones(np.shape(x_f)[0])


t_o = 0.0
t_f = 10.0
delt = 0.1

num_samples = int(t_f/delt)

t_vec = np.linspace(t_o, t_f, num_samples).reshape(num_samples, 1)


A_x = np.hstack(( 1.0*np.ones((num_samples,1)), t_vec, t_vec**2   ))
A_xdot = np.hstack(( np.zeros((num_samples,1)), np.ones(( num_samples, 1)), 2*t_vec   ))
A_xddot = np.hstack(( np.zeros((num_samples,1)), np.zeros((num_samples,1)), np.ones(( num_samples, 1 ))   ))

P_x = np.hstack(( t_vec**3, t_vec**4, t_vec**5))
P_xdot = np.hstack(( 3*t_vec**2, 4*t_vec**3, 5*t_vec**4   ))
P_xddot = np.hstack((6*t_vec, 12*t_vec**2, 20*t_vec**3 ))

sol_x = np.ones((np.shape(x_f)[0], 3  ))
sol_y = np.ones((np.shape(x_f)[0], 3  ))

x_traj = np.ones((np.shape(x_f)[0], num_samples  ))
y_traj = np.ones((np.shape(x_f)[0], num_samples  ))

for i in range(0, np.shape(x_f)[0]):

	c_x_tilda = np.hstack(( x_0, xdot_0, xddot_0   ))
	c_y_tilda = np.hstack(( y_0, ydot_0, yddot_0   ))

	s_x = np.hstack(( x_f[i], xdot_f[i], xddot_f[i]   ))
	s_y = np.hstack(( y_f[i], ydot_f[i], yddot_f[i]  ))

	G_x = np.vstack(( A_x[-1], A_xdot[-1], A_xddot[-1]  ))
	H_x = np.vstack(( P_x[-1], P_xdot[-1], P_xddot[-1]  ))
	
	G_y = G_x
	H_y = H_x



	sol_x[i] = np.linalg.solve(H_x, s_x-np.dot(G_x, c_x_tilda))
	sol_y[i] = np.linalg.solve(H_y, s_y-np.dot(G_y, c_y_tilda))
	x_traj[i] = np.dot(A_x, c_x_tilda)+np.dot(P_x, sol_x[i])
	y_traj[i] = np.dot(A_x, c_y_tilda)+np.dot(P_x, sol_y[i])


h_car = 4.8/2.0
w_car = 2.4/2.0



x_ego_vehicle_init = np.array([ x_0+h_car, x_0+h_car, x_0-h_car, x_0-h_car, x_0+h_car    ])
y_ego_vehicle_init =  np.array([ y_0-w_car, y_0+w_car, y_0+w_car, y_0-w_car, y_0-w_car    ])        


plt.figure(1)
plt.plot(x_traj.T, y_traj.T, linewidth = 3.0)
plt.plot(x_ego_vehicle_init, y_ego_vehicle_init, '-k', linewidth = 3.0)
plt.axis('equal')
plt.show()












