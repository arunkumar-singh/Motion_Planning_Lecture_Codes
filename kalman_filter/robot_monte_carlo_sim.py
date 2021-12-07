
import numpy as np
import matplotlib.pyplot as plt 



def robot_monte_carlo_sim(num_steps, vx, vy, delta_t):





	x_pos = x_init*np.ones(4*num_steps+1)
	y_pos = y_init*np.ones(4*num_steps+1)

	




	for i in range(1, 4*num_steps+1):

		eps_vx = np.random.normal(0, 0.05)
		eps_vy = np.random.normal(0, 0.05)




		x_pos[i] = x_pos[i-1]+(vx[i-1]+eps_vx)*delta_t 
		y_pos[i] = y_pos[i-1]+(vy[i-1]+eps_vx)*delta_t 


	return x_pos, y_pos	





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

num_sim = 100

x_sim = x_init*np.ones(( num_sim, 4*num_steps+1   ))
y_sim = y_init*np.ones(( num_sim, 4*num_steps+1   ))



for i in range(0, num_sim):

	x_sim[i], y_sim[i] = robot_monte_carlo_sim(num_steps, vx, vy, delta_t)



plt.figure(1)
plt.plot(x_sim.T, y_sim.T)
plt.show()	