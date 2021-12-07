



from numpy import *
from scipy.linalg import block_diag
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """
    def eigsorted(cov):
        vals, vecs = linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]


    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = degrees(arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip



def robot_monte_carlo_sim(num_steps, vx, vy, delta_t):





	x_pos = x_init*np.ones(4*num_steps+1)
	y_pos = y_init*np.ones(4*num_steps+1)

	




	for i in range(1, 4*num_steps+1):

		eps_vx = np.random.normal(0, 0.05)
		eps_vy = np.random.normal(0, 0.05)




		x_pos[i] = x_pos[i-1]+(vx[i-1]+eps_vx)*delta_t 
		y_pos[i] = y_pos[i-1]+(vy[i-1]+eps_vx)*delta_t 


	return x_pos, y_pos	




############################################################################

num_steps = 100

x_init = 0.0
y_init = 0.0

vx_1 = 0.2*np.ones(num_steps)
vy_1 = 0.0*np.ones(num_steps)

vx_2 = 0.0*np.ones(num_steps)
vy_2 = 0.2*np.ones(num_steps)

vx_3 = -0.2*np.ones(num_steps)
vy_3 = 0.0*np.ones(num_steps)

vx_4 = 0.0*np.ones(num_steps)
vy_4 = -0.2*np.ones(num_steps)

vx = np.hstack(( vx_1, vx_2, vx_3, vx_4   ))
vy = np.hstack(( vy_1, vy_2, vy_3, vy_4   ))

delta_t = 0.1

num_state = 2

A_motion = np.identity(num_state)
B_motion = np.identity(num_state)*delta_t


num_sim = 100

x_sim = x_init*np.ones(( num_sim, 4*num_steps+1   ))
y_sim = y_init*np.ones(( num_sim, 4*num_steps+1   ))



for i in range(0, num_sim):

	x_sim[i], y_sim[i] = robot_monte_carlo_sim(num_steps, vx, vy, delta_t)







###########################################################################






# maxiter = 400
# num_steps = 100

# num_state = 2
# delt = 0.1

# x_init = 0.0
# y_init = 0.0

# vx_1 = 0.2*np.ones(num_steps)
# vy_1 = 0.0*np.ones(num_steps)

# vx_2 = 0.0*np.ones(num_steps)
# vy_2 = 0.2*np.ones(num_steps)

# vx_3 = -0.2*np.ones(num_steps)
# vy_3 = 0.0*np.ones(num_steps)

# vx_4 = 0.0*np.ones(num_steps)
# vy_4 = -0.2*np.ones(num_steps)

# v_x = np.hstack(( vx_1, vx_2, vx_3, vx_4   ))
# v_y = np.hstack(( vy_1, vy_2, vy_3, vy_4   ))

# x_beacon = 0.2
# y_beacon = 0.2

# A_motion = identity(num_state)
# B_motion = identity(num_state)*delt

# ################ This is what decides R_k
sigma_vx = 0.05
sigma_vy = 0.05

covariance_control = block_diag( sigma_vx**2, sigma_vy**2 ) #########

# mu_x_t_post = x_init*ones(maxiter+1)
# mu_y_t_post = y_init*ones(maxiter+1)
# cov_state_post = zeros(( maxiter+1, num_state**2  ))


mu_x_t_post_without_measurement = x_init*ones(4*num_steps+1)
mu_y_t_post_without_measurement = y_init*ones(4*num_steps+1)
cov_state_post_without_measurement = zeros(( 4*num_steps+1, num_state**2  ))


# C_observ = identity(num_state) ######## C_k
# Q_observ = block_diag( 0.001, 0.001    ) ########### Q_k




# ############## Kalman filter loop
# for i in range(0, maxiter):

#     ############### Eqn(25)
#     mu_x_t_prior = mu_x_t_post[i]+v_x[i]*delt ######## \overline{mu}_k-1 = A\mu_{k-1}+Bu_k, but since A is identity, it gets further simplified.
#     mu_y_t_prior = mu_y_t_post[i]+v_y[i]*delt

#     R_k = dot(B_motion, dot(covariance_control, B_motion.T)   )
#     cov_state_prior =(dot(A_motion,  dot(cov_state_post[i].reshape(num_state, num_state), A_motion.T  )  )+R_k)
#     ####untill here eqn(25)


#     if sqrt((mu_x_t_prior-x_beacon)**2+(mu_y_t_prior-y_beacon)**2 )>0.3:

#         Q_observ = Q_observ*100

#     #from here equation(26)
#     kalman_gain_temp_1 = dot( cov_state_prior, C_observ.T) ### 24a
#     kalman_gain_temp_2 = dot(C_observ.T, dot(cov_state_prior, C_observ.T) )+Q_observ ####### 24b

#     ########## P_k
#     kalman_gain = dot(kalman_gain_temp_1, linalg.inv(kalman_gain_temp_2)) #### 24c

#     cov_state_post[i+1] = dot( (identity(num_state) -dot(kalman_gain, C_observ)), cov_state_prior ).reshape(num_state**2)
#     z_measurement = dot(C_observ, hstack(( mu_x_t_prior, mu_y_t_prior  ))   )

#     mu_post = hstack((mu_x_t_prior, mu_y_t_prior))+dot(kalman_gain, (z_measurement-dot(C_observ, hstack((mu_x_t_prior, mu_y_t_prior)) )) )
#     mu_x_t_post[i+1] = mu_post[0]
#     mu_y_t_post[i+1] = mu_post[1]



for i in range(1, 4*num_steps+1):
	mu_x_t_prior = mu_x_t_post_without_measurement[i-1]+vx[i-1]*delta_t
	mu_y_t_prior = mu_y_t_post_without_measurement[i-1]+vy[i-1]*delta_t

	cov_state_prior =(dot(A_motion,  dot(cov_state_post_without_measurement[i-1].reshape(num_state, num_state), A_motion.T  )  )+dot(B_motion, dot(covariance_control, B_motion.T)   ) )

	cov_state_post_without_measurement[i] = cov_state_prior.reshape(num_state**2)

	mu_x_t_post_without_measurement[i] = mu_x_t_prior
	mu_y_t_post_without_measurement[i] = mu_y_t_prior





fig, ax = plt.subplots(1, 1)

for i in range(0, 4*num_steps+1):
	# plot_cov_ellipse(cov_state_post[i].reshape(num_state, num_state), hstack(( mu_x_t_post[i], mu_y_t_post[i]  )), nstd=2, ax=None, edgecolor = 'k', fill = 0)
	plot_cov_ellipse(cov_state_post_without_measurement[i].reshape(num_state, num_state), hstack(( mu_x_t_post_without_measurement[i], mu_y_t_post_without_measurement[i]  )), nstd=2, ax=None, edgecolor = 'r', fill = 0)


ax.plot(mu_x_t_post_without_measurement, mu_y_t_post_without_measurement)
ax.plot(x_sim.T, y_sim.T)

# ax.plot(x_beacon, y_beacon, 'om', markersize = 5.0)
# ax.plot(x_t[0], y_t[0], 'og', markersize = 2 )
# plot_cov_ellipse(cov_state_post, hstack(( x_t[i], y_t[i]  )), nstd=2, ax=None, edgecolor = 'k', linewidth = 3.0)
# plot_cov_ellipse(cov_state_post_2, hstack(( x_t[i], y_t[i]  )), nstd=2, ax=None, edgecolor = 'r', linewidth = 3.0)



# plot_cov_ellipse(cov_state[10].reshape(num_state, num_state), hstack(( x_t[10], y_t[10]  )), nstd=1, ax=None)

plt.axis('equal')

# plt.figure(2)
# plt.plot(sqrt((mu_x_t_post-x_beacon)**2+(mu_y_t_post-y_beacon)**2))

plt.show()
