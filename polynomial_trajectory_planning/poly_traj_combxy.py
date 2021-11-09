

import numpy as np

import matplotlib.pyplot as plt 

import time



x_1 = 0.0
y_1 = 0.0

x_2 = 5.0
y_2 = 0.0

x_3 = 5.0
y_3 = 6.0

x_4 = 4.0
y_4 = 6.0

x = np.hstack(( x_1, x_2, x_3, x_4   ))
y = np.hstack(( y_1, y_2, y_3, y_4  ))

plt.plot(x, y)
plt.axis('equal')