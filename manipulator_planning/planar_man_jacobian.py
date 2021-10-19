


import numpy as  np
import matplotlib.pyplot as plt
import time

###planar_man_jacobian.py


def jac_planar(q):

    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]

    jac_theta_row_1 = np.hstack(( -l_1*np.sin(q_1)-l_2*np.sin(q_1+q_2)-l_3*np.sin(q_1+q_2+q_3), -l_2*np.sin(q_1+q_2)-l_3*np.sin(q_1+q_2+q_3),  -l_3*np.sin(q_1+q_2+q_3)   ))
    jac_theta_row_2 = np.hstack(( l_1*np.cos(q_1)+l_2*np.cos(q_1+q_2)+l_3*np.cos(q_1+q_2+q_3), l_2*np.cos(q_1+q_2)+l_3*np.cos(q_1+q_2+q_3),  l_3*np.cos(q_1+q_2+q_3)   ))
    jac_theta_row_3 = np.hstack(( 1.0, 1.0,  1.0   ))

    jac_theta = np.vstack(( jac_theta_row_1, jac_theta_row_2    ))

    return jac_theta



def fk_fun(q):
    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]

    x_e = l_1*np.cos(q_1)+l_2*np.cos(q_1+q_2)+l_3*np.cos(q_1+q_2+q_3)
    y_e = l_1*np.sin(q_1)+l_2*np.sin(q_1+q_2)+l_3*np.sin(q_1+q_2+q_3)

    return x_e, y_e, q_1+q_2+q_3


l_1 = 1.5
l_2 = 1.5
l_3 = 1.5

x_f = 4.0
y_f = 3.5


k = 1.0

x_f = np.random.uniform(-4.0, 4.0)
y_f = np.random.uniform(-4.0, 4.0)
gamma_f = 0.0

x_f = -0.45
y_f =  -0.59




maxiter = 600
x_traj = np.ones(maxiter)
y_traj = np.ones(maxiter)
gamma_traj = np.ones(maxiter)

q_init = np.hstack((0.1, 0.1, 0.1   ))
# q_init = np.hstack(( 0.1, 0.1, 0.1   ))
q_traj = np.zeros((maxiter, 3))

man_x_init = np.hstack((0.0, l_1*np.cos(q_init[0]), l_1*np.cos(q_init[0])+l_2*np.cos(q_init[0]+q_init[1]), l_1*np.cos(q_init[0])+l_2*np.cos(q_init[0]+q_init[1])+l_3*np.cos(q_init[0]+q_init[1]+q_init[2])   ))
man_y_init = np.hstack((0.0, l_1*np.sin(q_init[0]), l_1*np.sin(q_init[0])+l_2*np.sin(q_init[0]+q_init[1]), l_1*np.sin(q_init[0])+l_2*np.sin(q_init[0]+q_init[1])+l_3*np.sin(q_init[0]+q_init[1]+q_init[2])   ))
x_traj[0], y_traj[0], gamma_traj[0] = fk_fun(q_init)
cost_track = np.ones(maxiter-1)

delt = 0.01

print(x_f, y_f)


for i in range(1, maxiter):

    xdot = k*(x_f-x_traj[i-1])
    ydot = k*(y_f-y_traj[i-1])
    gammadot = k*(gamma_f-gamma_traj[i-1])
    jac = jac_planar(q_init)
    qdot = np.dot(np.linalg.pinv(jac), np.hstack(( xdot, ydot   )) )

    q_init = q_init+qdot*delt
    q_init = np.clip(q_init, -2.1*np.ones(3),  2.1*np.ones(3))
    x_traj[i], y_traj[i], gamma_traj[i] = fk_fun(q_init)
    cost_track[i-1] = np.sqrt( (x_f-x_traj[i])**2+(y_f-y_traj[i])**2              )
    q_traj[i] = q_init


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
