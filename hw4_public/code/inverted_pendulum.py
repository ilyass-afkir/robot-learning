import numpy as np
from utils import MuJoCo, ObservationType, Policy, PlotRewards
from pathlib import Path
import matplotlib.pyplot as plt


class InvertedPendulum(MuJoCo):
    """
    Mujoco simulation of Inverted Pendulum environment.
    """
    def __init__(self, gamma=0.99, horizon=100):
        """
        The __init__ function is the constructor for a class. It is called when an instance of a class is created.
        The __init__ function can take arguments, but self is always the first one. This function serves to initialize
        the objects that are instances of this class, e.g., setting up variables and loading files.

        :param self: Reference the object in which it is called
        :param gamma=0.99: Define the discount factor
        :param horizon=100: Define the number of timesteps in one episode
        :return: The object itself

        """
        action_spec = ["linear_joint"]
        observation_spec = [("joint0", ObservationType.JOINT_POS),
                            ("joint0", ObservationType.JOINT_VEL),
                            ("joint1", ObservationType.JOINT_POS),
                            ("joint1", ObservationType.JOINT_VEL),
                            ]

        super().__init__(
            (Path(__file__).resolve().parent / "data" / "inverted_pendulum.xml").as_posix(),
            action_spec,
            observation_spec,
            gamma,
            horizon,
            n_substeps=4,
            additional_data_spec=[],
            collision_groups=[])

        self.init_pendulum_conf = np.array([0, 0, 0, 0])  # the pendulum always starts at up position

    def reward(self, state, action, next_state, absorbing=False): # TODO: investigate this function for Task 4.2 a
        """
        The reward function r(s,a) is a function of the state s and action a.
        It returns the reward for taking action a in state s.
        The reward is defined as:

        :param self: Access the attributes and methods of the class in python
        :param state: Determine the current position and velocity of the agent
        :param action: Penalize the agent for choosing actions that are too large
        :param next_state: Compute the reward
        :param absorbing=False: Tell the environment whether we are in a terminal state or not
        :return: A negative value of the distance between the current position and goal

        """

        x = state[0]
        y = 0.5 * np.cos(state[2])
        diff = np.sqrt((0.1 * x) ** 2 + (y - 0.5) ** 2)
        r = - diff - 1e-3 * np.linalg.norm(action)
        return r

    def is_absorbing(self, state):
        """
        The is_absorbing function is a helper function that checks if the state is absorbing.
        It returns True if it is an absorbing state, and False otherwise.

        :param self: Access the class attributes
        :param state: Check if the state is absorbing
        :return: False

        """
        return False

    def setup(self):
        """
        The setup function is called when the environment is first created.
        It initializes the environment, and sets up any variables that need to be reset each time a new episode starts.

        :param self: Access the class attributes
        :return: The initial state of the pendulum

        """
        self._data.qpos[:] = [self.init_pendulum_conf[0], self.init_pendulum_conf[2]]
        self._data.qvel[:] = [self.init_pendulum_conf[1], self.init_pendulum_conf[3]]

def test_execute_episode():
    """
    The test_execute_episode function tests the execute_episode function.
    It creates an inverted pendulum environment and a policy, then executes the episode using that policy.
    The render parameter is set to True so that we can see what happens in the episode.
    """
    mdp = InvertedPendulum(0.99, 100)
    theta = np.asarray([1, 1, 5, 1, 0.1]).reshape((5, 1)) #TODO: policy parameters, Task 4.2.c
    policy = make_policy(mdp, theta)
    execute_episode(policy, mdp, 100, 0.99, render=True)

def execute_episode(policy, mdp, num_steps, discount, render=False):
    """
    The execute_episode function executes a single episode of the environment.
    It takes in a policy, an MDP, and number of steps to execute as arguments.
    The function returns the discounted reward for that episode.
    The environment's step method is called, which advances the environment's internal state by one time step.
    The agent provides an action to the environment, and the environment returns a new observation, the reward earned
    by the agent, and a flags which here not important.

    :param policy: Get the action for a given state
    :param mdp: Get the current state, the reward and other information
    :param num_steps: Define the number of steps that are taken in an episode
    :param discount: Discount the future rewards
    :param render=False: environment rendering
    :return: The discounted reward
    """
    discounted_reward = 0  # discounted reward
    state = mdp.reset()  # rest the environment
    for step in range(num_steps):
        action = policy.get_action(state) # TODO: complete how to get an action, Task 4.2.c
        state, reward, _, _ = mdp.step(action)
        discounted_reward += reward * (discount ** step)

        if render and step % 2 == 0:
            mdp.render()

    return discounted_reward


def evaluation(mdp, theta, num_steps, discount=0.90): # TODO: complete this function for Task 4.2 d
    """
    The evaluation function takes in an MDP, a policy and the number of steps to be executed.
    It returns the total reward received by executing that policy for that number of steps.

    :param mdp: Get the agent
    :param theta: the samples
    :param num_steps: Determine the length of each episode
    :param discount=0.90: Discount the rewards of future states
    :return: The average reward over a specific number of episodes
    """
    # Task 2.4.d
    policy = make_policy(mdp, theta) # TODO
    reward = execute_episode(policy, mdp, num_steps, discount) # TODO
    return reward


def make_policy(mdp, theta):
    """
    The make_policy function creates a policy object.
    The policy is a mapping from state to action.

    :param mdp: Get the observation and action space
    :param theta: Define the samples
    :return: A policy with theta as its parameter
    """

    return Policy(theta, mdp.observation_space, mdp.action_space)


def experiment(params):     # TODO: complete this function for Task 4.2.e
    """
    CEM-Algorithm:
    The Cross-Entropy Method is a reinforcement learning algorithm that is used to find the optimal actions for a given problem.
    It works by first generating a random sample of actions and calculating the mean reward for each action. The actions with the highest mean reward are then selected and
    used as the starting point for the next iteration. This process is repeated for a specified number of iterations, and the final set of actions with the highest mean reward
    are considered the optimal actions.
    The experiment function takes in a dictionary of parameters.
    The function also has access to the *mdp* object (agent), which is an instance of InvertedPendulum.
    The experiment function does not have return value, but it creates a plot showing how the mean reward changes over iterations.
    Hints:
    - add additional variance of the form [max(1 - itr / waning_time, 0) * additional_std^2] (regularization) to the computed variance in the first steps (waning_time)
    - to get the best action use the best [batch size * fraction of samples] actions

    :param params: Pass the parameters to the experiment function
    """

    # set parameters
    discount = params["discount"]
    num_steps = params["num_steps"]
    iterations = params["num_iter"]
    batch_size = params["batch_size"]  # number of samples per batch
    fraction = params["fraction"]
    additional_std = params["additional_std"]
    waning_time = params["waning_time"]

    # mean rewards for plotting
    mean_rewards = []
    rewards = []

    # rendering
    render = params["render"]

    # create MDP environment
    mdp = InvertedPendulum(0.99, num_steps)

    # get dimensions for theta
    dim_theta = 5

    # initialize mean and standard deviation
    theta_mean = [1, 1, 5, 1, 0.1]    # dummy init
    theta_std = [2, 2, 4, 2, 1]     # also dummy init

    # loop of CEM algorithm iteration
    for iteration in range(iterations):

        # generate policy batch / parameters corresponding to a policy. Too long for ternary operator :(
        if iteration < waning_time:
            # extra variance
            thetas = np.asarray([[theta_mean[i] + np.random.normal(0.0, np.sqrt(theta_std[i]**2 + max(1 - (iterations / waning_time), 0) * additional_std**2)) for i in range(dim_theta)] for pol in range(batch_size)])
        else:
            # only theta_std
            thetas = np.asarray([[theta_mean[i] + np.random.normal(0.0, theta_std[i]) for i in range(dim_theta)] for pol in range(batch_size)])

        # create a dictionary out of parameters indices and their reward
        rewards_dict = dict(zip(range(batch_size), [evaluation(mdp, thetas[i], num_steps, discount) for i in range(batch_size)]))

        # get the best thetas by sorting dict by values in descending order
        sorted_dict = dict(sorted(rewards_dict.items(), key=lambda x: x[1], reverse=True)[0:int(fraction * batch_size)])
        sorted_key_list = list(sorted_dict.keys())
        best_thetas = np.asarray([thetas[i] for i in sorted_key_list])

        # calculate new means and std
        theta_mean = [np.mean([t[i] for t in best_thetas]) for i in range(dim_theta)]
        theta_std = [np.std([t[i] for t in best_thetas]) for i in range(dim_theta)]

        # save mean reward
        #mean_rewards.append(np.mean(list(rewards_dict.values())))
        rewards.append(list(rewards_dict.values()))

    return rewards


if __name__ == '__main__':
    # TODO: this function should be called for Task 4.2 c
    # test_execute_episode() # please uncomment this line after successful completion of Task 4.2c

    """
    The main function runs the experiment.
    """
    # Task 4.2.e
    params = {
        "num_steps": 100,       # the number of steps that are taken in an episode
        "discount": 0.9,        # discount factor for reward
        "num_iter": 30,         # number of iterations for CEM
        "batch_size": 30,       # the number of samples that will be processed together at one time
        "fraction": 0.2,        # fraction of samples (used with the batch to get the first best thetas)
        "additional_std": 10.0,  # TODO, # additional standard deviation used for creation the regularization matrix
        "waning_time": 20,     # TODO, # waning time specifies a duration in which we artificially increase the variance of the parameters through a regularization matrix
        "render": True,         # True or False to enable/disable rendering
        }

    # initialize for training
    num_run = 10
    rewards = []
    # run the experiment n times -> this is the call to the CEM method
    [rewards.append(experiment(params)) for i in range(num_run)]

    # TODO: implement plotting of the mean rewards for Task 4.2 f
    # Task 2.4.f
    # TODO: use "plot_mean_conf" (in utils) to plot the rewards
    # plot mean rewards
    plot = PlotRewards()
    ax = plt.gca()
    ax.set_xlabel("number of epochs")
    ax.set_ylabel("mean reward")
    plot.plot_mean_conf(data=np.asarray(rewards).reshape(num_run, -1), ax=ax, label="mean reward")
    plt.show()
