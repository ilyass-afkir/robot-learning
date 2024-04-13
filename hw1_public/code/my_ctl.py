# CTL is the name of the controller.
# Q_HISTORY is a matrix containing all the past position of the robot. Each row of this matrix is [q_1, ... q_i], where
# i is the number of the joints. Please do not use / change these values!
# Q and QD are the current position and velocity, respectively. Use them for control!
# Q_DES, QD_DES, QDD_DES are the desired position, velocity and acceleration, respectively.
# GRAVITY is the gravity vector g(q). This is already precomputed and provided to you!
# CORIOLIS is the Coriolis force vector c(q, qd). This is already precomputed and provided to you!
# M is the mass matrix M(q). This is already precomputed and provided to you!

import numpy as np

def my_ctl(ctl, q, qd, q_des, qd_des, qdd_des, q_hist, q_deshist, gravity, coriolis, M):
    #TODO set the parameters here

    K_P = np.array([[65, 0], [0, 35]])
    K_D = np.array([[12, 0], [0, 8]])
    K_I = np.array([[0.1, 0], [0, 0.1]])

    dt = 0.002

    if ctl == 'P':
        #TODO Implement your controller here
        u = np.matmul(K_P, (q_des - q).reshape((2, 1)))
    elif ctl == 'PD':
        #TODO Implement your controller here
        u = np.matmul(K_P, (q_des - q).reshape((2, 1))) + np.matmul(K_D, (q_des - q).reshape((2, 1)))
    elif ctl == 'PID':
        #TODO Implement your controller here
        # Wahrscheinlich einfach die Matrixeinträge von Q_history abziehen von einander und dann 1/numberOfRows fürs Integral
        integral = np.sum((q_hist - q_deshist)*dt, axis=0).reshape((2, 1))
        u = np.matmul(K_P, (q_des - q).reshape((2, 1))) + np.matmul(K_D, (q_des - q).reshape((2, 1))) + np.matmul(K_I, integral)
    elif ctl == 'PD_Grav':
        #TODO Implement your controller here
        u = np.matmul(K_P, (q_des - q).reshape((2, 1))) + np.matmul(K_D, (q_des - q).reshape((2, 1))) + np.asarray(gravity).reshape((2, 1))
    elif ctl == 'ModelBased':
        #TODO Implement your controller here
        qdd_ref = qdd_des.reshape((2, 1)) + np.matmul(K_D, (qd_des - qd).reshape((2, 1))) + np.matmul(K_P, (q_des - q).reshape((2, 1)))
        u = np.matmul(M, qdd_ref) + np.asarray(coriolis).reshape((2, 1)) + np.asarray(gravity).reshape((2, 1))
    elif ctl == 'PD_high_gains':
        u = np.matmul(5*K_P, (q_des - q).reshape((2, 1))) + np.matmul(5*K_D, (q_des - q).reshape((2, 1)))  #TODO Implement your controller here (only needed in 1.2 D)
    return u

