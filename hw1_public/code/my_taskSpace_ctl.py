# CTL is the name of the controller.
# DT is the time step.
# Q and QD are respectively the position and the velocity of the joints.
# GRAVITY is the gravity vector g(q).
# CORIOLIS is the Coriolis force vector c(q, qd).
# M is the mass matrix M(q).
# J is the Jacobian.
# CART are the coordinates of the current position.
# DESCART are the coordinates of the desired position.

import numpy as np
from math import pi

def my_taskSpace_ctl(ctl, dt, q, qd, gravity, coriolis, M, J, cart, desCart, resting_pos=None):
    KP = np.diag([65, 35])
    KD = np.diag([12, 8])
    gamma = 0.6
    dFact = 1e-6

    if ctl == 'JacTrans':
        qd_des = gamma * J.T * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity
    elif ctl == 'JacPseudo':
        qd_des = gamma * J.T * np.linalg.pinv(J * J.T) * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity
    elif ctl == 'JacDPseudo':
        qd_des = J.T * np.linalg.pinv(J * J.T + dFact * np.eye(2)) * (desCart - cart)
        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP,KD])) * np.vstack([error,errord]) + coriolis + gravity
    elif ctl == 'JacNullSpace':
        #TODO Implement your controller here
        qd_0 = np.matmul(KP, resting_pos - q)   # vgl. robotics Foliensatz Folie 102
        xd_des = (desCart - cart)       # vgl. Folie 99, Was soll den eta sein? Hier geht es super mit eta = 1

        # soll hier der damping coefficient rein und gleich überall? vgl. Seite 101/102
        qd_des = np.matmul(J.T * np.linalg.pinv(J * J.T + dFact * np.eye(2)), xd_des) + \
                 np.matmul(np.eye(2) - np.matmul(J.T * np.linalg.pinv(J * J.T + dFact * np.eye(2)), J), qd_0)

        error = q + qd_des * dt - q
        errord = qd_des - qd
        u = M * np.vstack(np.hstack([KP, KD])) * np.vstack([error, errord]) + coriolis + gravity
        print(q)
        print(cart)

    return u
