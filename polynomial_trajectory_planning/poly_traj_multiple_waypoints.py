

import numpy as np

import matplotlib.pyplot as plt 

import time



x_to = 0.0
y_to = 0.0

xdot_to = 0.0
ydot_to = 0.0

xddot_to = 0.0
yddot_to = 0.0


xdot_tf = 0.0
ydot_tf = 0.0

xddot_tf = 0.0
yddot_tf = 0.0



x_1 = 5.0
y_1 = 8.0

x_2 = 12.0
y_2 = -3.0


x = np.hstack(( x_to, x_1, x_3  ))
y = np.hstack(( y_to, y_1, y_3  ))



delt = 0.02
t_o = 0.0
t_f = 10.0
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




plt.plot(x, y)
plt.axis('equal')
plt.show()