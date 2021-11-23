



from numpy import *

import matplotlib.pyplot as plt
from scipy.io import loadmat

import cvxopt
from cvxopt import solvers

############## constrained_poly_traj_eq.py




x_traj_temp = loadmat('x_traj_desired_quad.mat')
x_d = x_traj_temp['x_traj'].squeeze()

y_traj_temp = loadmat('y_traj_desired_quad.mat')
y_d = y_traj_temp['y_traj'].squeeze()



x_to = 0.0
y_to = 0.0

x_t1 = 2.0
y_t1 = 25.0

x_t2 = 6.0
y_t2 = 35.0


x_tf = x_d[-1]
y_tf = y_d[-1]

xdot_to = 0.0
xddot_to = 0.0

xdot_tf = 0.0
xddot_tf = 0.0

ydot_to = 0.0
yddot_to = 0.0

ydot_tf = 0.0
yddot_tf = 0.0

delt = 0.02
t_o = 0.0
t_f = 120.0
delt = 0.04

num_samples = int(t_f/delt)

t_vec = linspace(t_o, t_f, num_samples).reshape(num_samples, 1)

############################## A and P matrix definition

A_x = hstack(( 1.0*ones((num_samples,1)), t_vec, t_vec**2   ))
A_xdot = hstack(( zeros((num_samples,1)), ones(( num_samples, 1)), 2*t_vec   ))
A_xddot = hstack(( zeros((num_samples,1)), zeros((num_samples,1)), ones(( num_samples, 1 ))   ))

P_x = hstack(( t_vec**3, t_vec**4, t_vec**5,  t_vec**6, t_vec**7    ))
P_xdot = hstack(( 3*t_vec**2, 4*t_vec**3, 5*t_vec**4, 6*t_vec**5, 7*t_vec**6   ))
P_xddot = hstack((6*t_vec, 12*t_vec**2, 20*t_vec**3, 30*t_vec**4,  42*t_vec**5  ))


A_y = hstack(( 1.0*ones((num_samples,1)), t_vec, t_vec**2   ))
A_ydot = hstack(( zeros((num_samples,1)), ones(( num_samples, 1)), 2*t_vec   ))
A_yddot = hstack(( zeros((num_samples,1)), zeros((num_samples,1)), ones(( num_samples, 1 ))   ))

P_y = hstack(( t_vec**3, t_vec**4, t_vec**5,  t_vec**6, t_vec**7    ))
P_ydot = hstack(( 3*t_vec**2, 4*t_vec**3, 5*t_vec**4, 6*t_vec**5, 7*t_vec**6   ))
P_yddot = hstack((6*t_vec, 12*t_vec**2, 20*t_vec**3, 30*t_vec**4,  42*t_vec**5  ))


Gx_end = vstack(( A_x[-1], A_xdot[-1], A_xddot[-1] ))
Hx_end = vstack(( P_x[-1], P_xdot[-1], P_xddot[-1] ))

Gy_end = vstack(( A_y[-1], A_ydot[-1], A_yddot[-1] ))
Hy_end = vstack(( P_y[-1], P_ydot[-1], P_yddot[-1] ))


sx_end = hstack(( x_tf, xdot_tf, xddot_tf   ))
sy_end = hstack(( y_tf, ydot_tf, yddot_tf   ))

c_x_tilda = hstack(( x_to, xdot_to, xddot_to   ))
c_y_tilda = hstack(( y_to, ydot_to, yddot_to   ))

w_1 = 100.0
w_2 = 1.0

# x_d = linspace(x_to, x_tf*10, num_samples)
# y_d = 0.5*y_tf*ones(num_samples)

Qx_tr = w_1*dot(P_x.T, P_x)
qx_tr = -w_1*dot(P_x.T, x_d-dot(A_x, c_x_tilda))

Qx_acc = w_2*dot(P_xddot.T, P_xddot)
qx_acc = w_2*dot(P_xddot.T, dot(A_xddot, c_x_tilda))

Qx = Qx_tr+Qx_acc
qx = qx_tr+qx_acc

# sol_x = linalg.solve(-Qx, qx)

Qy_tr = w_1*dot(P_y.T, P_y)
qy_tr = -w_1*dot(P_y.T, y_d-dot(A_y, c_y_tilda))

Qy_acc = w_2*dot(P_yddot.T, P_yddot)
qy_acc = w_2*dot(P_yddot.T, dot(A_yddot, c_y_tilda))

Qy = Qy_tr+Qy_acc
qy = qy_tr+qy_acc

# sol_y = linalg.solve(-Qy, qy)


Mx_eq = Hx_end ############ the M matrix
nx_eq = sx_end-dot(Gx_end, c_x_tilda) ### n vector

My_eq = Hy_end
ny_eq = sy_end-dot(Gy_end, c_y_tilda)


sol_x = solvers.qp(cvxopt.matrix(Qx, tc = 'd'), cvxopt.matrix(qx, tc = 'd'), None, None, cvxopt.matrix(Mx_eq, tc = 'd'), cvxopt.matrix(nx_eq, tc = 'd'))
sol_x = array(sol_x['x']).squeeze()


sol_y = solvers.qp(cvxopt.matrix(Qy, tc = 'd'), cvxopt.matrix(qy, tc = 'd'), None, None, cvxopt.matrix(My_eq, tc = 'd'), cvxopt.matrix(ny_eq, tc = 'd'))
sol_y = array(sol_y['x']).squeeze()

############################ plotting the solution
x = dot(A_x, c_x_tilda)+dot(P_x, sol_x)
y = dot(A_y, c_y_tilda)+dot(P_y, sol_y)

xdot = dot(A_xdot, c_x_tilda)+dot(P_xdot, sol_x)

plt.figure(1)
plt.plot(x, y, '-', linewidth = 3.0)
plt.plot(x_d, y_d, '-k')

plt.figure(2)
plt.plot(xdot)

plt.show()
