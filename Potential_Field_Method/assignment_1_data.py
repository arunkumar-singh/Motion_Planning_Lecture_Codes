

import numpy as np



################## Data for Assignment 1 Part 1

################## obstacle coordinates


x_o_1 = 3.1
y_o_1 = 3.0

x_o_2 = 3
y_o_2 = -2.0

x_o_3 = -1.0
y_o_3 = -2.0

x_o_4 = -2.0
y_o_4 = 3.0


############################### Start and goal postions

x_init = np.hstack(( 0.0, 6.0, 0.34, -6.0, -5.0    ))
y_init = np.hstack(( 0.0, -6.0, 0.69, 6.0, -0.35  ))

x_g = 6.0*np.ones(5)
y_g = 6.0*np.ones(5)





############################################ Data for Assignment 2

x_init = 0.0
y_init = 0.0
z_init = 0.0

x_g = 6.0
y_g = 6.0
z_g = 6.0

########### obstacle coordinates
x_o = 3.1
y_o = 3.0
z_o = 3.0







