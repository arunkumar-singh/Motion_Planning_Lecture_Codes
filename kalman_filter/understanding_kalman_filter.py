

from numpy import *


import matplotlib.pyplot  as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms



def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov_1 = cov(x, y)
    pearson = cov_1[0, 1]/sqrt(cov_1[0, 0] * cov_1[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = sqrt(1 + pearson)
    ell_radius_y = sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = sqrt(cov_1[0, 0]) * n_std
    mean_x = mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = sqrt(cov_1[1, 1]) * n_std
    mean_y = mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def get_correlated_dataset(n, dependency, mu, scale):
    latent = random.randn(n, 2)
    dependent = latent.dot(dependency)
    scaled = dependent * scale
    scaled_with_offset = scaled + mu
    # return x and y of the new, correlated dataset
    return scaled_with_offset[:, 0], scaled_with_offset[:, 1]




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












x_prev = 1.0
y_prev = 1.0

vx_k = 0.5
vy_k = 0.3

delt = 0.4

epsilon_vx = 0.1
epsilon_vy = 0.1


####################### in determinstic case


########################### Position after time Delta t




x_next_det = x_prev+delt*vx_k
y_next_det = y_prev+delt*vy_k


################################ in Stochastic case


x_next_stochastic = x_prev+delt*(vx_k+random.normal(0, epsilon_vx))
y_next_stochastic = y_prev+delt*(vy_k+random.normal(0, epsilon_vx))

print(x_next_det, y_next_det)
print(x_next_stochastic, y_next_stochastic)


num_sim = 10000

x_next_stochastic_vec = ones(num_sim)
y_next_stochastic_vec = ones(num_sim)


for i in range(0, num_sim):

    x_next_stochastic_vec[i] = x_prev+delt*(vx_k+random.normal(0, epsilon_vx))
    y_next_stochastic_vec[i] = y_prev+delt*(vy_k+random.normal(0, epsilon_vx))


fig, ax = plt.subplots(1, 1)


# plt.figure(1)
ax.plot(x_next_stochastic_vec, y_next_stochastic_vec, 'om')
ax.plot(x_prev*ones(1), y_prev*ones(1), 'og', markersize = 20.0)
ax.plot(x_next_det*ones(1), y_next_det*ones(1), 'ok', markersize = 20.0)
confidence_ellipse(x_next_stochastic_vec, y_next_stochastic_vec, ax, edgecolor='red', linewidth = 3.0)


plt.show()
