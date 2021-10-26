

##################### poly_traj_linsolve

from numpy import *
import matplotlib.pyplot as plt


def compute_traj_singleaxis( s, G, H, c_tilda ):

	sol = linalg.solve(H, s-dot(G, c_tilda))

	return sol






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

############ t_vec = [t_1 t_2, t_3.....t_n]


# x_t = x_to+xdot_to*t_vec+(1/2)*xddot_to*t_vec**2+a_1*t_vec**3+a_2*t_vec**4+a_3*t_vec**5+a_4*t_vec**6+a_5*t_vec**7
# xdot_t = xdot_to+xddot_to*t_vec+3*a_1*t_vec**2+4*a_2*t_vec**3+5*a_3*t_vec**4+6*a_4*t_vec**5+7*a_5*t_vec**6
# xddot_t = xddot_to+6*a_1*t_vec+12*a_2*t_vec**2+20*a_3*t_vec**3+30*a_4*t_vec**4+42*a_5*t_vec**5
# xdddot_t = 6*a_1+24*a_2*t_vec+60*a_3*t**2+120*a_4*t_vec**3+210*a_5*t_vec**4


A_x = hstack(( 1.0*ones((num_samples,1)), t_vec, t_vec**2   ))
A_xdot = hstack(( zeros((num_samples,1)), ones(( num_samples, 1)), 2*t_vec   ))
A_xddot = hstack(( zeros((num_samples,1)), zeros((num_samples,1)), ones(( num_samples, 1 ))   ))

P_x = hstack(( t_vec**3, t_vec**4, t_vec**5,  t_vec**6, t_vec**7    ))
P_xdot = hstack(( 3*t_vec**2, 4*t_vec**3, 5*t_vec**4, 6*t_vec**5, 7*t_vec**6   ))
P_xddot = hstack((6*t_vec, 12*t_vec**2, 20*t_vec**3, 30*t_vec**4,  42*t_vec**5  ))


G = vstack(( A_x[int(num_samples/3)-1], A_x[2*int(num_samples/3)-1], A_x[-1], A_xdot[-1], A_xddot[-1]  ))
H = vstack(( P_x[int(num_samples/3)-1], P_x[2*int(num_samples/3)-1], P_x[-1], P_xdot[-1], P_xddot[-1]  ))



c_x_tilda = hstack(( x_to, xdot_to, xddot_to   ))
c_y_tilda = hstack(( y_to, ydot_to, yddot_to   ))

s_x = hstack(( x_t1, x_t2, x_tf, xdot_tf, xddot_tf   ))
s_y = hstack(( y_t1, y_t2, y_tf, ydot_tf, yddot_tf   ))

sol_x = compute_traj_singleaxis( s_x, G, H, c_x_tilda )
sol_y = compute_traj_singleaxis( s_y, G, H, c_y_tilda )

x_t = dot(A_x, c_x_tilda)+dot(P_x, sol_x)
y_t = dot(A_x, c_y_tilda)+dot(P_x, sol_y)

xdot_t = dot(A_xdot, c_x_tilda)+dot(P_xdot, sol_x)
ydot_t = dot(A_xdot, c_y_tilda)+dot(P_xdot, sol_y)

plt.figure(1)
plt.plot(x_t, y_t)
plt.plot(x_to, y_to, 'om', markersize = 10.0)
plt.plot(x_tf, y_tf, 'og', markersize = 10.0)
plt.plot(x_t1, y_t1, 'ok', markersize = 10.0)
plt.plot(x_t2, y_t2, 'ok', markersize = 10.0)

plt.figure(2)
plt.plot(xdot_t, '-r')
plt.plot(ydot_t, '-k')


plt.show()
