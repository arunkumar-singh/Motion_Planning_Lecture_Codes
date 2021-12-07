
import numpy as np

import matplotlib.pyplot as plt 




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


x_pos = x_init*np.ones(4*num_steps+1)
y_pos = y_init*np.ones(4*num_steps+1)

delta_t = 0.1



for i in range(1, 4*num_steps+1):



	x_pos[i] = x_pos[i-1]+vx[i-1]*delta_t 
	y_pos[i] = y_pos[i-1]+vy[i-1]*delta_t 




plt.figure(1)
plt.plot(x_pos, y_pos, 'ok', markersize = 3.0)
plt.plot(x_pos[0], y_pos[0], 'om', markersize = 18.0)
plt.plot(x_pos[-1], y_pos[-1], 'og', markersize = 13.0)

plt.show()