

# fk_franka.py

import numpy as np

def fk_franka(q):

    q_1 = q[0]
    q_2 = q[1]
    q_3 = q[2]
    q_4 = q[3]
    q_5 = q[4]
    q_6 = q[5]
    q_7 = q[6]

    x = -0.107*(((np.sin(q_1)*np.sin(q_3) - np.cos(q_1)*np.cos(q_2)*np.cos(q_3))*np.cos(q_4) - np.sin(q_2)*np.sin(q_4)*np.cos(q_1))*np.cos(q_5) + (np.sin(q_1)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1)*np.cos(q_2))*np.sin(q_5))*np.sin(q_6) - 0.088*(((np.sin(q_1)*np.sin(q_3) - np.cos(q_1)*np.cos(q_2)*np.cos(q_3))*np.cos(q_4) - np.sin(q_2)*np.sin(q_4)*np.cos(q_1))*np.cos(q_5) + (np.sin(q_1)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1)*np.cos(q_2))*np.sin(q_5))*np.cos(q_6) + 0.088*((np.sin(q_1)*np.sin(q_3) - np.cos(q_1)*np.cos(q_2)*np.cos(q_3))*np.sin(q_4) + np.sin(q_2)*np.cos(q_1)*np.cos(q_4))*np.sin(q_6) - 0.107*((np.sin(q_1)*np.sin(q_3) - np.cos(q_1)*np.cos(q_2)*np.cos(q_3))*np.sin(q_4) + np.sin(q_2)*np.cos(q_1)*np.cos(q_4))*np.cos(q_6) + 0.384*(np.sin(q_1)*np.sin(q_3) - np.cos(q_1)*np.cos(q_2)*np.cos(q_3))*np.sin(q_4) + 0.0825*(np.sin(q_1)*np.sin(q_3) - np.cos(q_1)*np.cos(q_2)*np.cos(q_3))*np.cos(q_4) - 0.0825*np.sin(q_1)*np.sin(q_3) - 0.0825*np.sin(q_2)*np.sin(q_4)*np.cos(q_1) + 0.384*np.sin(q_2)*np.cos(q_1)*np.cos(q_4) + 0.316*np.sin(q_2)*np.cos(q_1) + 0.0825*np.cos(q_1)*np.cos(q_2)*np.cos(q_3)

    y = 0.107*(((np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1))*np.cos(q_4) + np.sin(q_1)*np.sin(q_2)*np.sin(q_4))*np.cos(q_5) - (np.sin(q_1)*np.sin(q_3)*np.cos(q_2) - np.cos(q_1)*np.cos(q_3))*np.sin(q_5))*np.sin(q_6) + 0.088*(((np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1))*np.cos(q_4) + np.sin(q_1)*np.sin(q_2)*np.sin(q_4))*np.cos(q_5) - (np.sin(q_1)*np.sin(q_3)*np.cos(q_2) - np.cos(q_1)*np.cos(q_3))*np.sin(q_5))*np.cos(q_6) - 0.088*((np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1))*np.sin(q_4) - np.sin(q_1)*np.sin(q_2)*np.cos(q_4))*np.sin(q_6) + 0.107*((np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1))*np.sin(q_4) - np.sin(q_1)*np.sin(q_2)*np.cos(q_4))*np.cos(q_6) - 0.384*(np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1))*np.sin(q_4) - 0.0825*(np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + np.sin(q_3)*np.cos(q_1))*np.cos(q_4) - 0.0825*np.sin(q_1)*np.sin(q_2)*np.sin(q_4) + 0.384*np.sin(q_1)*np.sin(q_2)*np.cos(q_4) + 0.316*np.sin(q_1)*np.sin(q_2) + 0.0825*np.sin(q_1)*np.cos(q_2)*np.cos(q_3) + 0.0825*np.sin(q_3)*np.cos(q_1)

    z = -0.107*((np.sin(q_2)*np.cos(q_3)*np.cos(q_4) - np.sin(q_4)*np.cos(q_2))*np.cos(q_5) - np.sin(q_2)*np.sin(q_3)*np.sin(q_5))*np.sin(q_6) - 0.088*((np.sin(q_2)*np.cos(q_3)*np.cos(q_4) - np.sin(q_4)*np.cos(q_2))*np.cos(q_5) - np.sin(q_2)*np.sin(q_3)*np.sin(q_5))*np.cos(q_6) + 0.088*(np.sin(q_2)*np.sin(q_4)*np.cos(q_3) + np.cos(q_2)*np.cos(q_4))*np.sin(q_6) - 0.107*(np.sin(q_2)*np.sin(q_4)*np.cos(q_3) + np.cos(q_2)*np.cos(q_4))*np.cos(q_6) + 0.384*np.sin(q_2)*np.sin(q_4)*np.cos(q_3) + 0.0825*np.sin(q_2)*np.cos(q_3)*np.cos(q_4) - 0.0825*np.sin(q_2)*np.cos(q_3) - 0.0825*np.sin(q_4)*np.cos(q_2) + 0.384*np.cos(q_2)*np.cos(q_4) + 0.316*np.cos(q_2) + 0.33

    r_32 = -cq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2)) - sq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4))
    r_33 = -cq6*(cq2*cq4 + cq3*sq2*sq4) + sq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5)
    r_31 = cq7*(cq6*(cq5*(cq2*sq4 - cq3*cq4*sq2) + sq2*sq3*sq5) + sq6*(cq2*cq4 + cq3*sq2*sq4)) - sq7*(cq5*sq2*sq3 - sq5*(cq2*sq4 - cq3*cq4*sq2))
    r_21 = cq7*(cq6*(cq5*(cq4*(cq1*sq3 + cq2*cq3*sq1) + sq1*sq2*sq4) + sq5*(cq1*cq3 - cq2*sq1*sq3)) + sq6*(cq4*sq1*sq2 - sq4*(cq1*sq3 + cq2*cq3*sq1))) - sq7*(cq5*(cq1*cq3 - cq2*sq1*sq3) - sq5*(cq4*(cq1*sq3 + cq2*cq3*sq1) + sq1*sq2*sq4))
    r_11 = cq7*(cq6*(cq5*(cq1*sq2*sq4 + cq4*(cq1*cq2*cq3 - sq1*sq3)) - sq5*(cq1*cq2*sq3 + cq3*sq1)) + sq6*(cq1*cq4*sq2 - sq4*(cq1*cq2*cq3 - sq1*sq3))) + sq7*(cq5*(cq1*cq2*sq3 + cq3*sq1) + sq5*(cq1*sq2*sq4 + cq4*(cq1*cq2*cq3 - sq1*sq3)))

    roll = np.arctan2(r_32, r_33)

    pitch =  -np.arcsin(r_31)

    yaw =  np.arctan2(r_21, r_11)


    return x, y, z, roll, pitch, yaw
