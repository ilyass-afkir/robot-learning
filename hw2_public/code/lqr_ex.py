import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


class LQR:
    def __init__(self):
        """
        The __init__ function is called automatically when a new instance of the class is created.
        The __init__ function can take arguments, but self is always the first one.
        Self is just a reference to the instance of the class. It's automatically passed in when you instantiate an instance of the class.

        :param self: Refer to an instance of the class
        :return: None
        """
        # TODO set up all parameters here
        self.dim = None
        self.T = 50
        self.A = np.asarray([[1, 0.1], [0, 1]])
        self.B = np.asarray([0, 0.1]).reshape((2, 1))
        self.Sigma = np.asarray([[0.001, 0], [0, 0.001]])
        self.b = np.asarray([10, 0])
        self.K = np.asarray([5, 0.3]).reshape((1, 2))
        self.k = 0.3
        self.H = 1
        self.R = np.asarray([[0.01, 0], [0, 0.1]])
        self.r = np.asarray([25, 0]).reshape((2, 1))
        self.N = 20
        self.V = [None] * (self.T + 1)
        self.v = [None] * (self.T + 1)

    def run(self, policy, active_sdes=False):
        """
        The run function runs the LQR with the given properties.
        It returns a tuple of three elements: (state, action, reward).
        The state is a 3D array of shape (2, 51, N), where 2 is the number of states in each time step and N is the number
        of experiments. The action and reward are both 3D arrays with shapes (51,) and (51,) respectively.

        :param self: Access variables that belongs to the class
        :param policy: Compute the action
        :param active_sdes: mark to set sdes to zero or not. Adaption by me to be able to get results for sdes = 0 and sdes = r in one execution
        :param rand_seed=None: Ensure that the random seed is not set
        :return: [numpy arrays] The history of states (2, 51, 20), actions (1, 51, 20) and rewards (1, 51, 20) for each experiment
        """
        history_s = np.zeros(
            (2, self.T + 1, self.N)
        )  # state at t=0 up to state at t=50 for each experiment
        history_r = np.zeros(
            (1, self.T + 1, self.N)
        )  # reward at t=0 up to reward at t=50 for each experiment
        history_a = np.zeros(
            (1, self.T + 1, self.N)
        )  # action at t=0 up to action at t=50 for each experiment
        # TODO: implement the LQR with the given properties [for task a and task b]
        """
        implement your code here
        """

        # N experiments
        for i in range(self.N):
            # init state s_0
            s = gaussian_noise(self.b, np.eye(2)).reshape((2, 1))

            #print("Experiment: " + str(i))

            # each experiment for up to T days
            for j in range(self.T+1):
            #for j in [50, 40, 30, 20, 10, 0]:

                #print("Iteration: " + str(j))

                # get action depending on current policy
                if str(policy.__name__) == "action_function":
                    reward_r = np.asarray([15, 0]).reshape((2, 1)) if (j <= 14) else self.r
                    a = policy(self.K, s, self.k)

                elif str(policy.__name__) == "P_action_function":
                    reward_r = np.asarray([15, 0]).reshape((2, 1)) if (j <= 14) else np.asarray([25, 0]).reshape((2, 1))

                    # change sdes either to reward_r or 0 based on the newly introduced parameter
                    sdes = reward_r if active_sdes else np.zeros((2, 1))
                    a = policy(self.K, s, self.k, sdes)

                elif str(policy.__name__) == "optimal_policy":
                    reward_r = np.asarray([15, 0]).reshape((2, 1)) if (j <= 14) else self.r
                    a = policy(self, s, j)

                else:
                    raise NameError("Unknown policy function!")

                # get noise
                w = gaussian_noise(self.b, self.Sigma).reshape((2, 1))

                # get reward
                reward_R = np.asarray([[10000, 0], [0, 0.01]]) if (j in [14, 40]) else self.R
                reward = reward_function(s, reward_r, reward_R, a, self.H, j, self.T)

                # save in history
                history_s[:, j, i] = s.reshape((2,))
                history_a[:, j, i] = a
                history_r[:, j, i] = reward

                # get new state
                s = np.matmul(self.A, s) + np.matmul(self.B, a) + w

        return (history_s, history_a, history_r)

    def learn(self):
        """
        The learn function returns a policy function which takes in an input state and time,
        and returns the action to take. The policy is computed using the optimal LQR algorithm.

        :param self: Access variables that belongs to the class
        :return: A function that takes as input a state and time
        """
        # TODO: implement the optimal LQR [for task c]
        """
        implement your code here
        """

        # returns M_t
        def M(self, t:int):
            V_t_plus = V(self, t+1)
            return self.B * 1.0/(self.H + np.transpose(self.B) @ V_t_plus @ self.B) @ np.transpose(self.B) @ V_t_plus @ self.A

        # returns V_t
        def V(self, t:int):
            # look up in table if already calculated
            if t < len(self.V) and self.V[t] is not None:
                return self.V[t]
            # else calculate it
            else:
                reward_R = np.asarray([[10000, 0], [0, 0.01]]) if (t in [14, 40]) else self.R

                if t >= self.T:
                    return_value = reward_R
                else:
                    return_value = reward_R + np.transpose(self.A - M(self, t)) @ V(self, t+1) @ self.A

                if t <= self.T:
                    self.V[t] = return_value

                return return_value

        # returns v_t
        def v(self, t:int):
            # look up in table if already calculated
            if t < len(self.v) and self.v[t] is not None:
                return self.v[t]
            # else calculate it
            else:
                reward_R = np.asarray([[10000, 0], [0, 0.01]]) if (t in [14, 40]) else self.R
                reward_r = np.asarray([15, 0]).reshape((2, 1)) if (t <= 14) else self.r

                if t >= self.T:
                    return_value = reward_R @ reward_r
                else:
                    return_value = reward_R @ reward_r + np.transpose(self.A - M(self, t)) @ (v(self, t+1) - V(self, t+1) @ self.b.reshape((2, 1)))

                if t <= self.T:
                    self.v[t] = return_value

                return return_value

        # returns K_t
        def K(self, t):
            V_t_plus = V(self, t+1)
            return - 1.0/(self.H + np.transpose(self.B) @ V_t_plus @ self.B) @ np.transpose(self.B) @ V_t_plus @ self.A

        # returns k_t
        def k(self, t):
            V_t_plus = V(self, t+1)
            return - 1.0/(self.H + np.transpose(self.B) @ V_t_plus @ self.B) @ np.transpose(self.B) @ (V_t_plus @ self.b.reshape((2, 1)) - v(self, t+1))

        # optimal policy function
        def optimal_policy(self, state: np.ndarray, t:int):
            return K(self, t) @ state + k(self, t)

        return optimal_policy


def run_tasks(obj):
    """
    The run_tasks function runs the LQR, P controller and optimal policy.
    It plots the results for each task in a separate figure.

    :param obj: Pass the object to the run_tasks function
    :return: The mean and std of the cumulative reward over all experiments
    """
    # TODO: [task a] Run LQR controller and plot states and action and print the cumulative reward
    """
    implement your code here
    """

    # execute system with no policy and save its history
    no_policy_history = obj.run(action_function)
    plot_states_and_actions(no_policy_history)
    # mean and std for 2a)
    """
    mean, std = calculate_reward_mean_and_deviation(no_policy_history[2])
    print(mean)
    print(std)
    """

    # TODO: [task b] Run P controller (s_des = r) and (s_des = 0) and plot the results
    p_policy_history_0 = obj.run(P_action_function)
    p_policy_history_r = obj.run(P_action_function, active_sdes=True)
    plot_p_states(p_policy_history_0, p_policy_history_r)

    """
    mean, std = calculate_reward_mean_and_deviation(p_policy_history_r[2])
    print(mean)
    print(std)
    """

    # TODO: [task c] Learn optimal policy and use it and plot the results
    # Learn optimal policy and use it
    """
    implement your code here
    """
    opt_pol = obj.learn()
    optimal_policy_history = obj.run(opt_pol)

    plot_states_in_comparison(no_policy_history[0][0, :, :], p_policy_history_r[0][0, :, :], optimal_policy_history[0][0, :, :], "Values of state 1")
    plot_states_in_comparison(no_policy_history[0][1, :, :], p_policy_history_r[0][1, :, :], optimal_policy_history[0][1, :, :], "Values of state 2")

    # calculate mean and std for optimal policy
    mean, std = calculate_reward_mean_and_deviation(optimal_policy_history[2])
    print(mean)
    print(std)

    return mean, std


# returns the next state, given the current
def next_state_function(A, s, B, a, w):
    return np.matmul(A, s) + np.matmul(B, a) + w


# returns the value of the input signal a
def action_function(K, s: np.ndarray, k):
    return - np.matmul(K, s) + k


# returns the value of the input signal a similar to a P controller
def P_action_function(K, s: np.ndarray, k, sdes: np.ndarray):
    return np.matmul(K, sdes - s) + k


# returns the value of the reward function
def reward_function(s, r, R, a, H, T, T_max):
    base_reward = - np.matmul(np.transpose(s - r), np.matmul(R, s - r))
    return base_reward - a * H * a if (T != T_max) else base_reward


# returns a random variable distributed like N(b,Σ)
def gaussian_noise(b, sigma):
    return np.random.multivariate_normal(b, sigma)


# plots the actions and states for each time step t
def plot_states_and_actions(history: np.ndarray):
    # get states
    state_values1 = history[0][0, :, :]
    state_values2 = history[0][1, :, :]
    a, b = state_values1.shape

    # calculate mean states
    mean_states1 = np.asarray([(1.0/b) * sum(state_values1[t, :]) for t in range(a)])
    mean_states2 = np.asarray([(1.0/b) * sum(state_values2[t, :]) for t in range(a)])

    # calculate standard deviation
    std1 = np.asarray([np.sqrt((1.0/(b-1)) * sum(np.square(state_values1[t, :] - mean_states1[t]))) for t in range(a)])
    std2 = np.asarray([np.sqrt((1.0/(b-1)) * sum(np.square(state_values2[t, :] - mean_states2[t]))) for t in range(a)])

    # get actions
    action_values = history[1][0, :, :]
    a, b = action_values.shape

    # calculate mean actions
    mean_actions = np.asarray([(1.0/b) * sum(action_values[t, :]) for t in range(a)])

    # calculate standard deviation of the actions
    action_std = np.asarray([np.sqrt((1.0/(b-1)) * sum(np.square(action_values[t, :] - mean_actions[t]))) for t in range(a)])

    # plot means
    plt.plot(range(a), mean_states1)
    plt.plot(range(a), mean_states2)
    plt.plot(range(a), mean_actions)

    # plot 95 percentile for states
    plt.fill_between(range(a), mean_states1 + 2*std1, mean_states1 - 2*std1, alpha=0.3)
    plt.fill_between(range(a), mean_states2 + 2*std2, mean_states2 - 2*std2, alpha=0.3)
    plt.fill_between(range(a), mean_actions + 2*action_std, mean_actions - 2*action_std, alpha=0.3)

    # add titles etc.
    plt.xlabel("time horizon $t$")
    plt.ylabel("state/action value")
    plt.title("State and action values over time")
    plt.legend(["mean $s_1$", "mean $s_2$", "mean $a$"], loc="best")

    plt.show()


# plots the actions and states for each time step t
def plot_p_states(history_0: np.ndarray, history_r: np.ndarray):
    # get states
    state_values_0 = history_0[0][0, :, :]
    state_values_r = history_r[0][0, :, :]
    a, b = state_values_0.shape

    # calculate mean states
    mean_states_0 = np.asarray([(1.0/b) * sum(state_values_0[t, :]) for t in range(a)])
    mean_states_r = np.asarray([(1.0/b) * sum(state_values_r[t, :]) for t in range(a)])

    # calculate standard deviation
    std_0 = np.asarray([np.sqrt((1.0/(b-1)) * sum(np.square(state_values_0[t, :] - mean_states_0[t]))) for t in range(a)])
    std_r = np.asarray([np.sqrt((1.0/(b-1)) * sum(np.square(state_values_r[t, :] - mean_states_r[t]))) for t in range(a)])

    # plot state mean
    plt.plot(range(a), mean_states_0)
    plt.plot(range(a), mean_states_r)

    # plot 95 percentile for states
    plt.fill_between(range(a), mean_states_0 + 2*std_0, mean_states_0 - 2*std_0, alpha=0.3)
    plt.fill_between(range(a), mean_states_r + 2*std_r, mean_states_r - 2*std_r, alpha=0.3)

    # add titles etc.
    plt.xlabel("time horizon $t$")
    plt.ylabel("state value $s_1$")
    plt.title("State values of $s_1$ over time")
    plt.legend(["mean $s_1$ with $s^{des} = 0$", "mean $s_1$ with $s^{des} = r$"], loc="best")

    plt.show()


# plots the two states of all 3 controllers as needed by 2c
def plot_states_in_comparison(no_policy_states: np.ndarray, p_policy_states: np.ndarray, optimal_policy_states: np.ndarray, title: str):
    # get dimensions
    t, e = no_policy_states.shape

    # get means
    no_policy_means = np.asarray([(1.0/e) * sum(no_policy_states[i, :]) for i in range(t)])
    p_policy_means = np.asarray([(1.0/e) * sum(p_policy_states[i, :]) for i in range(t)])
    optimal_policy_means = np.asarray([(1.0/e) * sum(optimal_policy_states[i, :]) for i in range(t)])

    # get standard deviations
    no_policy_std = np.asarray([np.sqrt((1.0/(e-1)) * sum(np.square(no_policy_states[i, :] - no_policy_means[i]))) for i in range(t)])
    p_policy_std = np.asarray([np.sqrt((1.0/(e-1)) * sum(np.square(p_policy_states[i, :] - p_policy_means[i]))) for i in range(t)])
    optimal_policy_std = np.asarray([np.sqrt((1.0/(e-1)) * sum(np.square(optimal_policy_states[i, :] - optimal_policy_means[i]))) for i in range(t)])

    # plot means
    plt.plot(range(t), no_policy_means)
    plt.plot(range(t), p_policy_means)
    plt.plot(range(t), optimal_policy_means)

    # plot stds
    plt.fill_between(range(t), no_policy_means - 2*no_policy_std, no_policy_means + 2*no_policy_std, alpha=0.3)
    plt.fill_between(range(t), p_policy_means - 2*p_policy_std, p_policy_means + 2*p_policy_std, alpha=0.3)
    plt.fill_between(range(t), optimal_policy_means - 2*optimal_policy_std, optimal_policy_means + 2*optimal_policy_std, alpha=0.3)

    # add titles etc.
    plt.xlabel("time horizon $t$")
    plt.ylabel("state value")
    plt.title(title)
    plt.legend(["Controller a", "Controller b", "Controller c"], loc="best")

    plt.show()


# returns the mean and standard deviation of the cumulative rewards given the history of rewards
def calculate_reward_mean_and_deviation(rewards: np.ndarray):
    # get history dimension
    m, n, num_exp = rewards.shape

    # calculate cumulative rewards
    cumulative_rewards = np.asarray([sum(rewards[0, :, n]) for n in range(num_exp)])

    # get mean value of cumulative reward
    mean_cum_rew = (1/num_exp) * sum(cumulative_rewards)

    # get standard deviation
    st_dev = np.sqrt((1/(num_exp-1)) * sum(np.square(cumulative_rewards - mean_cum_rew)))

    return mean_cum_rew, st_dev


if __name__ == "__main__":
    LQR = LQR()
    run_tasks(LQR)
