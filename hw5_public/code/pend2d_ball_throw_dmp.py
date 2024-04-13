from math import cos, pi, sin

import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(precision=3, linewidth=100000)


class Pend2dBallThrowDMP:
    def __init__(self):
        """
        This function is called automatically every time the class is instantiated.
        It sets up all the objects that will be used in the class, such as variables and functions.

        :param self: Refer to the object instance itself
        """
        self.num_basis = 5
        self.num_traj_steps = 100
        self.dim_joints = 2
        self.dt = 0.01
        self.lengths = np.ones(self.dim_joints)
        self.masses = np.ones(self.dim_joints)
        self.init_state = np.array([-pi, 0.0, 0.0, 0.0])
        self.ball_goal = [2.0, 1.0]
        self.release_step = 50
        self.kp = 1400.0

    def get_desired_traj_dmp(self, theta):
        """
        The get_desired_traj_dmp function computes the desired trajectory of the robot's joints by using a DMP to
        generate a set of weights for each joint. The function takes in an array of DMP weights and returns an array
        of desired joint positions, velocities, accelerations, and time steps. The first two elements in the output are
        the initial position and velocity respectively. The rest is computed using Euler's method.

        :param self: Access the variables and methods of the class in which a method is defined
        :param theta: Weight the basis functions
        :return: The desired trajectory for the dmps
        """
        alphaz = 8.0 / 3.0
        alpha = 25.0
        beta = alpha / 4.0
        tau = 1.0
        Ts = 1.0
        g = self.init_state[::2]

        C = np.exp(-alphaz * np.arange(self.num_basis) / (self.num_basis - 1) * Ts)
        H = 0.5 / (0.65 * np.diff(C) ** 2)
        H = np.append(H, H[-1])

        q = np.zeros((self.num_traj_steps, 2 * self.dim_joints))
        q[0, :] = self.init_state
        x = np.ones(self.num_traj_steps)

        for i in range(self.num_traj_steps - 1):
            psi = np.exp(-H * (x[i] - C) ** 2)
            f = np.dot(theta.T, psi) * x[i] / np.sum(psi)
            qdd_des = (
                alpha * (beta * (g - q[i, ::2]) - (q[i, 1::2] / tau)) + f.T
            ) * tau ** 2
            q[i + 1, 1::2] = q[i, 1::2] + qdd_des * self.dt
            q[i + 1, ::2] = q[i, ::2] + q[i + 1, 1::2] * self.dt
            xd = -alphaz * x[i] * tau
            x[i + 1] = x[i] + xd * self.dt

        return q

    def transition_function(self, x, action):
        """
        The transition_function function takes in a state and an action, and returns the next state.
        The function assumes that the first half of the input vector is position states, while the second half is velocity states.
        The function also assumes that each velocity corresponds to a mass (the masses are stored in self.masses).


        :param self: Access variables that belongs to the class
        :param x: Store the position of the masses
        :param action: Determine the force applied to the cart
        :return: The new state of the system
        """
        xnew = np.zeros(x.shape)
        xnew[1::2] = x[1::2] + (action / self.masses) * self.dt
        xnew[::2] = x[::2] + xnew[1::2] * self.dt
        return xnew

    def get_forward_kinematics(self, theta):
        """
        The get_forward_kinematics function takes a list of joint angles and returns the end effector position.

        :param self: Access the variables that are defined in the class
        :param theta: Represent the angles of each joint
        :return: The forward kinematics of the robot
        """
        y = np.zeros((theta.shape[0], 2))[0]
        for i in range(self.dim_joints):
            y += (
                np.array([sin(np.sum(theta[: i + 1])), cos(np.sum(theta[: i + 1]))])
                * self.lengths[i]
            )
        return y

    def get_jacobian(self, theta):
        """
        The get_jacobian function returns the Jacobian matrix for a given set of joint angles.
        The Jacobian is defined as: J = [dx/dtheta_i, dy/dtheta_i] where (dx,dy) is the x and y
        component of the end effector's position relative to some reference point. The i refers to
        the index of each joint angle in that list.

        :param self: Access the class attributes
        :param theta: Calculate the position of each joint
        :return: Two values, the jacobian matrix and the end effector position
        """
        si = self.get_forward_kinematics(theta)
        J = np.zeros((2, self.dim_joints))
        for j in range(self.dim_joints):
            pj = np.array([0.0, 0.0])
            for i in range(j):
                pj += (
                    np.array([sin(sum(theta[: i + 1])), cos(sum(theta[: i + 1]))])
                    * self.lengths[i]
                )
            pj = -(si - pj)
            J[np.ix_([0, 1], [j])] = np.mat([-pj[1], pj[0]]).T
        return [J, si]

    def simulate_system(self, des_q):
        """
            The simulate_system function takes a desired trajectory of joint angles as input, and returns the resulting
            trajectory of joint velocities, along with the corresponding cartesian coordinates. The function also returns
            the forward kinematics at each timestep.

            :param self: Access the variables and other properties of the class
            :param des_q: Define the desired trajectory of the system
            """
        K = np.zeros((self.dim_joints, 2 * self.dim_joints))
        K[:, ::2] = self.kp * np.eye(self.dim_joints)
        K[:, 1::2] = 2 * np.sqrt(self.kp) * np.eye(self.dim_joints)

        q = np.zeros((des_q.shape[0], 2 * self.dim_joints))
        q[0, :] = self.init_state

        b = np.zeros((des_q.shape[0], 2))
        bd = np.zeros((des_q.shape[0], 2))
        b[0, :] = self.get_forward_kinematics(q[0, :])

        u = np.zeros((des_q.shape[0], self.dim_joints))

        for i in range(des_q.shape[0] - 1):
            u[i, :] = np.dot(K, (des_q[i, :] - q[i, :]).T)
            q[i + 1, :] = self.transition_function(q[i, :], u[i, :])
            if i > self.release_step:
                bd[i + 1, :] = bd[i, :]
                bd[i + 1, 1] = bd[i + 1, 1] - 10 * self.dt
                b[i + 1, :] = b[i, :] + bd[i, :] * self.dt
            else:
                b[i + 1, :] = self.get_forward_kinematics(q[i + 1, ::2])
                bd[i + 1, :] = np.dot(
                    self.get_jacobian(q[i + 1, ::2])[0], q[i + 1, 1::2].T
                )

        return [q, u, b, bd]

    def get_reward(self, theta):
        """
        The get_reward function computes the reward for a given trajectory.
        The input is a vector of joint angles, and the output is the distance from
        the ball to the goal at that point in time. The function also returns whether
        the trajectory was successful or not.

        :param self: Access the variables and methods of the class in python
        :param theta: Represent the weights of each basis function
        :return: The cost of the trajectory
        """
        q_des = self.get_desired_traj_dmp(np.reshape(theta, (-1, self.num_basis)).T)
        data_traj = self.simulate_system(q_des)

        u_factor = -1e-4
        u_cost = u_factor * np.linalg.norm(data_traj[1]) ** 2
        dist_factor = -1e4
        b_diff = self.ball_goal - data_traj[2][-1, :]
        r_cost = np.dot(b_diff, b_diff) * dist_factor
        return u_cost + r_cost

    def get_joints_in_task_space(self, q):
        """
        The get_joints_in_task_space function returns the joint positions of the robot in task space.
        The function takes as input a list of joint angles q, and outputs two lists x_i containing
        the position of each link's endpoint in 2D task space. The first element in x_i is always [0, 0].


        :param self: Access the variables and methods of the class in which it is used
        :param q: Get the joint angles of the robot
        :return: The position of the two joints in the task space
        """
        x1 = np.array(self.lengths[0] * np.array([sin(q[0]), cos(q[0])]))
        x2 = x1 + np.array(
            self.lengths[1] * np.array([sin(q[2] + q[0]), cos(q[2] + q[0])])
        )
        return x1, x2

    def visualize(self, q, line):
        """
        The visualize function takes in a configuration vector q and a matplotlib line object.
        It then computes the forward kinematics of the arm, plots the resulting curve, and updates
        the line object to reflect this change.

        :param self: Access the class attributes
        :param q: Pass the configuration of the robot to the function
        :param line: Update the line in the animation
        :return: The joints in task space, which is the position of the end effector
        """
        lw = 4.0
        fs = 26
        mp1, mp2 = self.get_joints_in_task_space(q)
        thisx = [0, mp1[0], mp2[0]]
        thisy = [0, mp1[1], mp2[1]]
        line.set_data(thisx, thisy)
