import numpy as np

# inputs are :
# mr and ml (velocities of each motor)
# p1 [xi, yi, teta i] (initial position)

length = 1  # distance between wheels
d = 0.1  # delta (increment of t)

pi = np.array([0, 0, 0])

mr = -10
ml = 10

if mr == ml:  # no rotation
    vc = (mr+ml)/2
    po = pi + np.array([vc*d, vc*d, 0])

else:  # rotation equations
    r = 0.5*(ml + mr)/(mr - ml)  # radius of rotation
    w = (mr - ml)/length  # angular velocity

    icc = np.array([pi[0]-r*np.sin(pi[2]), pi[1]+r*np.cos(pi[2])])  # instantaneous center of curvature

    rotmax = np.array([  # rotation matrix
        [np.cos(w*d), -np.sin(w*d), 0],
        [np.sin(w*d), np.cos(w*d), 0],
        [0, 0, 1]
        ])

    secmax= np.array([  # second matrix (multiplies rotation one)
                        [pi[0]-icc[0]],
                        [pi[1]-icc[1]],
                        [pi[2]]
                        ])

    po = np.dot(rotmax, secmax) + np.array([  # output position [x, y, teta]
            [icc[0]],
            [icc[1]],
            [w*d]
            ])

po[2] = po[2]/(2*np.pi)*360  # to change from radians to degrees

print("po= \n", po)