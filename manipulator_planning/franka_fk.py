



from sympy import *
import numpy as np

alpha1 = 0.0
alpha2 = -pi/2
alpha3 = pi/2
alpha4 = pi/2
alpha5 = -pi/2
alpha6 = pi/2
alpha7 = pi/2
alpha8 = 0.0

d1 = 0.330
d2 = 0.0
d3 = 0.3160
d4 = 0.0
d5 = 0.3840
d6 = 0.0
d7 = 0.0
	# d8 = 0.1070

a1 = 0.0
a2 = 0.0
a3 = 0.0
a4 = 0.0825
a5 = -0.0825
a6 = 0.0
a7 = 0.0880
a8 = 0.0
q8 = 0.0
d8 = 0.1070

cq1, cq2, cq3, cq4, cq5, cq6, cq7 = symbols('cq1, cq2, cq3, cq4, cq5, cq6, cq7')
sq1, sq2, sq3, sq4, sq5, sq6, sq7 = symbols('sq1, sq2, sq3, sq4, sq5, sq6, sq7')

# cq1, cq2, cq3, cq4, cq5, cq6, cq7 = symbols('cq1, cq2, cq3, cq4, cq5, cq6, cq7')
# sq1, sq2, sq3, sq4, sq5, sq6, sq7 = symbols('sq1, sq2, sq3, sq4, sq5, sq6, sq7')



T1 = np.array([ [cq1, -sq1, 0.0, a1  ], [ sq1*cos(alpha1), cq1*cos(alpha1), -sin(alpha1), -d1*sin(alpha1)  ], [ sq1*sin(alpha1), cq1*sin(alpha1), cos(alpha1), d1*cos(alpha1)   ], [0.0, 0.0, 0.0, 1.0 ]   ])
T2 = np.array([ [cq2, -sq2, 0.0, a2  ], [ sq2*cos(alpha2), cq2*cos(alpha2), -sin(alpha2), -d2*sin(alpha2)  ], [ sq2*sin(alpha2), cq2*sin(alpha2), cos(alpha2), d2*cos(alpha2)   ], [0.0, 0.0, 0.0, 1.0 ]   ])
T3 = np.array([ [cq3, -sq3, 0.0, a3  ], [ sq3*cos(alpha3), cq3*cos(alpha3), -sin(alpha3), -d3*sin(alpha3)  ], [ sq3*sin(alpha3), cq3*sin(alpha3), cos(alpha3), d3*cos(alpha3)   ], [0.0, 0.0, 0.0, 1.0 ]   ])
T4 = np.array([ [cq4, -sq4, 0.0, a4  ], [ sq4*cos(alpha4), cq4*cos(alpha4), -sin(alpha4), -d4*sin(alpha4)  ], [ sq4*sin(alpha4), cq4*sin(alpha4), cos(alpha4), d4*cos(alpha4)   ], [0.0, 0.0, 0.0, 1.0 ]   ])
T5 = np.array([ [cq5, -sq5, 0.0, a5  ], [ sq5*cos(alpha5), cq5*cos(alpha5), -sin(alpha5), -d5*sin(alpha5)  ], [ sq5*sin(alpha5), cq5*sin(alpha5), cos(alpha5), d5*cos(alpha5)   ], [0.0, 0.0, 0.0, 1.0 ]   ])
T6 = np.array([ [cq6, -sq6, 0.0, a6  ], [ sq6*cos(alpha6), cq6*cos(alpha6), -sin(alpha6), -d6*sin(alpha6)  ], [ sq6*sin(alpha6), cq6*sin(alpha6), cos(alpha6), d6*cos(alpha6)   ], [0.0, 0.0, 0.0, 1.0 ]   ])
T7 = np.array([ [cq7, -sq7, 0.0, a7  ], [ sq7*cos(alpha7), cq7*cos(alpha7), -sin(alpha7), -d7*sin(alpha7)  ], [ sq7*sin(alpha7), cq7*sin(alpha7), cos(alpha7), d7*cos(alpha7)   ], [0.0, 0.0, 0.0, 1.0 ]   ])
T8 = np.array([ [cos(q8), -sin(q8), 0.0, a8  ], [ sin(q8)*cos(alpha8), cos(q8)*cos(alpha8), -sin(alpha8), -d8*sin(alpha8)  ], [ sin(q8)*sin(alpha8), cos(q8)*sin(alpha8), cos(alpha8), d8*cos(alpha8)   ], [0.0, 0.0, 0.0, 1.0 ]   ])

T_fin = T1.dot(T2).dot(T3).dot(T4).dot(T5).dot(T6).dot(T7).dot(T8)
