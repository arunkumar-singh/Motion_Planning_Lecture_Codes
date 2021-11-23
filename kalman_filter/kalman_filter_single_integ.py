

from numpy import *
from scipy.linalg import block_diag
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.pyplot as plt



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





maxiter = 60

num_state = 2
delt = 0.1
v_x = 0.1
v_y = 0.2
x_init = 0.0
y_init = 0.0

x_beacon = 0.2
y_beacon = 0.2

A_motion = identity(num_state)
B_motion = identity(num_state)*delt

################ This is what decides R_k
sigma_vx = 0.05
sigma_vy = 0.02

covariance_control = block_diag( sigma_vx**2, sigma_vy**2 ) #########

mu_x_t_post = x_init*ones(maxiter+1)
mu_y_t_post = y_init*ones(maxiter+1)
cov_state_post = zeros(( maxiter+1, num_state**2  ))


mu_x_t_post_without_measurement = x_init*ones(maxiter+1)
mu_y_t_post_without_measurement = y_init*ones(maxiter+1)
cov_state_post_without_measurement = zeros(( maxiter+1, num_state**2  ))


C_observ = identity(num_state) ######## C_k
Q_observ = block_diag( 0.001, 0.001    ) ########### Q_k




############## Kalman filter loop
for i in range(0, maxiter):

    ############### Eqn(25)
    mu_x_t_prior = mu_x_t_post[i]+v_x*delt ######## \overline{mu}_k-1 = A\mu_{k-1}+Bu_k, but since A is identity, it gets further simplified.
    mu_y_t_prior = mu_y_t_post[i]+v_y*delt

    R_k = dot(B_motion, dot(covariance_control, B_motion.T)   )
    cov_state_prior =(dot(A_motion,  dot(cov_state_post[i].reshape(num_state, num_state), A_motion.T  )  )+R_k)
    ####untill here eqn(25)


    if sqrt((mu_x_t_prior-x_beacon)**2+(mu_y_t_prior-y_beacon)**2 )>0.3:

        Q_observ = Q_observ*100

    #from here equation(26)
    kalman_gain_temp_1 = dot( cov_state_prior, C_observ.T) ### 24a
    kalman_gain_temp_2 = dot(C_observ.T, dot(cov_state_prior, C_observ.T) )+Q_observ ####### 24b

    ########## P_k
    kalman_gain = dot(kalman_gain_temp_1, linalg.inv(kalman_gain_temp_2)) #### 24c

    cov_state_post[i+1] = dot( (identity(num_state) -dot(kalman_gain, C_observ)), cov_state_prior ).reshape(num_state**2)
    z_measurement = dot(C_observ, hstack(( mu_x_t_prior, mu_y_t_prior  ))   )

    mu_post = hstack((mu_x_t_prior, mu_y_t_prior))+dot(kalman_gain, (z_measurement-dot(C_observ, hstack((mu_x_t_prior, mu_y_t_prior)) )) )
    mu_x_t_post[i+1] = mu_post[0]
    mu_y_t_post[i+1] = mu_post[1]



for i in range(0, maxiter):
	mu_x_t_prior = mu_x_t_post_without_measurement[i]+v_x*delt
	mu_y_t_prior = mu_y_t_post_without_measurement[i]+v_y*delt

	cov_state_prior =(dot(A_motion,  dot(cov_state_post_without_measurement[i].reshape(num_state, num_state), A_motion.T  )  )+dot(B_motion, dot(covariance_control, B_motion.T)   ) )

	cov_state_post_without_measurement[i+1] = cov_state_prior.reshape(num_state**2)

	mu_x_t_post_without_measurement[i+1] = mu_x_t_prior
	mu_y_t_post_without_measurement[i+1] = mu_y_t_prior





fig, ax = plt.subplots(1, 1)

for i in range(0, maxiter+1):
	plot_cov_ellipse(cov_state_post[i].reshape(num_state, num_state), hstack(( mu_x_t_post[i], mu_y_t_post[i]  )), nstd=2, ax=None, edgecolor = 'k', fill = 0)
	plot_cov_ellipse(cov_state_post_without_measurement[i].reshape(num_state, num_state), hstack(( mu_x_t_post_without_measurement[i], mu_y_t_post_without_measurement[i]  )), nstd=2, ax=None, edgecolor = 'r', fill = 0)


ax.plot(mu_x_t_post, mu_y_t_post)
ax.plot(x_beacon, y_beacon, 'om', markersize = 5.0)
# ax.plot(x_t[0], y_t[0], 'og', markersize = 2 )
# plot_cov_ellipse(cov_state_post, hstack(( x_t[i], y_t[i]  )), nstd=2, ax=None, edgecolor = 'k', linewidth = 3.0)
# plot_cov_ellipse(cov_state_post_2, hstack(( x_t[i], y_t[i]  )), nstd=2, ax=None, edgecolor = 'r', linewidth = 3.0)



# plot_cov_ellipse(cov_state[10].reshape(num_state, num_state), hstack(( x_t[10], y_t[10]  )), nstd=1, ax=None)

plt.axis('equal')

plt.figure(2)
plt.plot(sqrt((mu_x_t_post-x_beacon)**2+(mu_y_t_post-y_beacon)**2))

plt.show()
