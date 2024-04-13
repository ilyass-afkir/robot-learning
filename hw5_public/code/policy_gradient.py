import time

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from pend2d_ball_throw_dmp import *


def learn(env, num_dim, num_samples, max_iter, num_trials, alpha_coeff):
    """
    The learn function takes as input the environment, number of dimensions,
    number of samples per iteration, maximum number of iterations to perform and
    the number of trials to run. It returns a numpy array containing the mean reward
    for each trial for each iteration.

    :param env: environment
    :param num_dim: Specify the number of dimensions in the parameter space
    :param num_samples: Determine the number of samples used in each iteration (episodes)
    :param max_iter: Define the number of iterations that will be performed
    :param num_trials: Define the number of times we run the same algorithm with different initializations
    :param alpha_coeff: Control the learning rate
    :return: The average reward obtained by the agent during each iteration
    """
    # TODO: implement your code here
    # Task 5.1 c, d, e, f

    # c) start
    mean_rewards = np.zeros((num_trials, max_iter))
    np.random.seed(int(datetime.now().timestamp()))

    for trial in range(num_trials):
        # init means and covariance for theta
        theta_means = np.zeros(num_dim)
        std = np.diag(5*np.ones(num_dim))       # stays constant
        variance = std @ std                    # \sigma⁻²
        inv_variance = np.linalg.inv(variance)  # equal to \sigma⁻²

        for iteration in range(max_iter):
            # create thetas from distribution
            thetas = np.random.multivariate_normal(theta_means, variance, num_samples)

            # get rewards
            current_rewards = np.asarray([env.get_reward(thetas[i, :]) for i in range(num_samples)])
            mean_rewards[trial, iteration] = np.mean(current_rewards)

            # get policy derivative for the means
            # np.linalg.inv(variance) * np.linalg.inv(variance) equals \sigma⁻²
            # d): incorporating subtraction of baseline: - np.mean(current_rewards)
            nabla_j_means = (1/num_samples) * np.sum([(inv_variance @ (thetas[i, :] - theta_means)) * (current_rewards[i] - np.mean(current_rewards)) for i in range(num_samples)], axis=0)

            # update theta means
            theta_means += alpha_coeff * nabla_j_means
    # c) end



    return mean_rewards


def plot_rewards(mean_reward):
    """
    The plot_rewards function plots the average reward for each iteration of the algorithm.
    It takes as input a list of mean rewards, one for each alpha value.

    :param mean_reward: Plot the average reward for each iteration

    """
    # TODO: implement your code here
    # Task 5.1 c, d, e, f

    # c) start

    trials, iterations = mean_reward.shape

    # get mean and stds
    agent_means = np.mean(mean_reward, axis=0)
    agent_stds = np.std(mean_reward, axis=0)

    plt.plot(range(iterations), agent_means)
    plt.fill_between(range(iterations), agent_means + 2 * agent_stds, agent_means - 2 * agent_stds, alpha=0.3)

    plt.xlabel("Iteration")
    plt.ylabel("Mean reward")
    plt.title("Mean rewards plotted over iterations")
    plt.savefig("mean_rewards.pdf")
    plt.show()

    # c) end

    pass


if __name__ == "__main__":
    """
    The main function of the script.
    
    to compute and plot mean rewards for each trial and call learn and plot functions
    """

    # create environment
    env = Pend2dBallThrowDMP()
    # TODO: set up parameters
    num_dim = 10
    num_samples = 50  # number of samples
    max_iter = 100  # number of iterations
    num_trials = 10  # number of trials
    alpha_coeff = 0.1  # learning rate

    # execute experiment
    mean_rewards = learn(env, num_dim, num_samples, max_iter, num_trials, alpha_coeff)

    """
    functions to save and read in reward matrices, as it takes ~5-10 minutes to learn the parameters
    with open("mean_rewards.txt", "w") as matrixfile:
        for row in mean_rewards:
            matrixfile.write(' '.join([str(a) for a in row]) + '\n')

    
    with open('mean_rewards.txt', 'r') as f:
        mean_rewards = np.asarray([[float(num) for num in line.split(' ')] for line in f])
    """

    # plot results
    plot_rewards(mean_rewards)
