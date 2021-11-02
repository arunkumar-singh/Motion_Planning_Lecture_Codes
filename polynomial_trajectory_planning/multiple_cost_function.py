



from numpy import *

import matplotlib.pyplot as plt


######### final_cost_poly_traj

x_to = 0.0
y_to = 0.0

x_t1 = 2.0
y_t1 = 25.0

x_t2 = 6.0
y_t2 = 35.0


x_tf = 10.0
y_tf = 30.0

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
t_f = 60.0
delt = 0.02

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

Gx_mid = vstack(( A_x[int(num_samples/3)-1], A_x[2*int(num_samples/3)-1] ))
Hx_mid = vstack(( P_x[int(num_samples/3)-1], P_x[2*int(num_samples/3)-1] ))


Gy_end = vstack(( A_y[-1], A_ydot[-1], A_yddot[-1] ))
Hy_end = vstack(( P_y[-1], P_ydot[-1], P_yddot[-1] ))

Gy_mid = vstack(( A_y[int(num_samples/3)-1], A_y[2*int(num_samples/3)-1] ))
Hy_mid = vstack(( P_y[int(num_samples/3)-1], P_y[2*int(num_samples/3)-1] ))



#
# G = vstack(( A_x[int(num_samples/3)-1], A_x[2*int(num_samples/3)-1], A_x[-1], A_xdot[-1], A_xddot[-1]  ))
# H = vstack(( P_x[int(num_samples/3)-1], P_x[2*int(num_samples/3)-1], P_x[-1], P_xdot[-1], P_xddot[-1]  ))

c_x_tilda = hstack(( x_to, xdot_to, xddot_to   ))
c_y_tilda = hstack(( y_to, ydot_to, yddot_to   ))

s_x_end = hstack(( x_tf, xdot_tf, xddot_tf   ))
s_y_end = hstack(( y_tf, ydot_tf, yddot_tf   ))


s_x_mid = hstack(( x_t1, x_t2  ))
s_y_mid = hstack(( y_t1, y_t2  ))


w_end = 1.0
w_mid = 1.0
w_acc = 1.0

Qx_end = w_end*dot(Hx_end.T, Hx_end)
qx_end = -w_end*dot(Hx_end.T, s_x_end-dot(Gx_end, c_x_tilda))

Qx_mid = w_mid*dot(Hx_mid.T, Hx_mid)
qx_mid = -w_mid*dot(Hx_mid.T, s_x_mid-dot(Gx_mid, c_x_tilda))

Qx_acc = w_acc*dot(P_xddot.T, P_xddot)
qx_acc = w_acc*dot(P_xddot.T, dot(A_xddot, c_x_tilda))

Qx = Qx_end+Qx_acc+Qx_mid
qx = qx_end+qx_acc+qx_mid

sol_x = linalg.solve(-Qx, qx) ###### -Qx^(-1)qx


################################

Qy_end = w_end*dot(Hy_end.T, Hy_end)
qy_end = -w_end*dot(Hy_end.T, s_y_end-dot(Gy_end, c_y_tilda))

Qy_mid = w_mid*dot(Hy_mid.T, Hy_mid)
qy_mid = -w_mid*dot(Hy_mid.T, s_y_mid-dot(Gy_mid, c_y_tilda))

Qy_acc = w_acc*dot(P_yddot.T, P_yddot)
qy_acc = w_acc*dot(P_yddot.T, dot(A_yddot, c_y_tilda))

Qy = Qy_end+Qy_acc+Qy_mid
qy = qy_end+qy_acc+qy_mid

sol_y = linalg.solve(-Qy, qy)

x = dot(A_x, c_x_tilda)+dot(P_x, sol_x)
y = dot(A_y, c_y_tilda)+dot(P_y, sol_y)

xdot = dot(A_xdot, c_x_tilda)+dot(P_xdot, sol_x)
ydot = dot(A_ydot, c_y_tilda)+dot(P_ydot, sol_y)

plt.figure(1)
plt.plot(x, y, '-', linewidth = 3.0)
plt.plot(x_to*ones(1), y_to*ones(1), 'om', markersize = 16.0)
plt.plot(x_tf*ones(1), y_tf*ones(1), 'og', markersize = 16.0)

plt.plot(x_t1*ones(1), y_t1*ones(1), 'ok', markersize = 16.0)
plt.plot(x_t2*ones(1), y_t2*ones(1), 'ok', markersize = 16.0)

plt.figure(2)

plt.plot(xdot, '-r', linewidth = 3.0)
plt.plot(ydot, '-b', linewidth = 3.0)




plt.show()