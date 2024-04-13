import matplotlib.pyplot as plt
import numpy as np


def prepare_data(path):
    """
    The prepare_data function takes a path to the data file as argument.
    It returns four variables.

    :param path: Path to data file
    :return: X, y, data_x and data_y
    X is a numpy array of shape (n, 2) where n is the number of rows in the dataset.
    Each row consists of two values: x0 and xi for i = 1..n-2 (the first column is all 1s).
    The second column contains all values from 0 to 100 with increments of 0.01; this range covers
    the interval [0, 100] inclusively. The last column contains all zeros except
    """
    data = np.loadtxt(path)
    data_x = data[:, 0]
    data_y = data[:, 1]

    n = len(data_x)
    X = np.append(np.ones((n, 1)), data_x.reshape(n, 1), axis=1)
    y = data_y.reshape(n, 1)

    return X, y, data_x, data_y


def cost_function(X, y, w):
    """
    The cost_function function computes the mean squared error for a given dataset
    and model weights. 

    :param X: Training data matrix (additional col. with 1)
    :param y: Training data labels
    :param w: Model weights
    :return: Mean squared error between predicted and actual values of y
    """
    return (1/y.size) * sum(np.square(X @ w - y))


def gradient_descent(X, y):
    """
    The gradient_descent function computes the gradient of the cost function at each iteration and updates
    the weights w. The number of iterations and learning rate are set as parameters.
    The costs vector is used to verify that the algorithm is converging.

    :param X: Training data matrix (additional col. with 1)
    :param y: Training data labels
    :return: The optimized weights and the costs per iteration
    """

    m, n = X.shape
    w = np.zeros((n, 1))
    costs = np.zeros(1000)
    alpha = 0.01

    for i in range(1000):
        # calculate value of derivative
        derivatives = np.zeros(n)
        for j in range(n):
            derivatives[j] = (2/y.size) * np.dot((X @ w - y).flatten(), X[:, j])

        # save costs of current iteration
        costs[i] = cost_function(X, y, w)

        # update weights
        for j in range(n):
            w[j] = w[j] - alpha * derivatives[j]

    return w, costs


def linear_regression(w):
    """
    The linear_regression function plots the line of best fit for a given weight vector.
    It takes in a weight vector and returns two lists, x_i and y_i, which are used to plot the line.

    :param w: Weight vector
    :return: two lists: x_i and y_i
    """
    # TODO write your own solution here

    x_i = np.linspace(-3, 4)
    y_i = w[1] * x_i + w[0]

    return x_i, y_i


def plot(data_x, data_y, x_i=None, y_i=None, costs=None, mode=None):
    """
    The plot function takes in data_x, data_y, x_i and y_i.
    It plots the points (data_x[i], data_y[i]) for all i.
    If x and y are provided it also plots a line of best fit.

    :param data_x: Data points
    :param data_y: Data point labels (i.e. y coordinate)
    :param x_i=None: Plot optimized weight vector
    :param y_i=None: Plot optimized weight vector
    :param costs=None: Pass a list of costs
    :param mode=None: Decide what to plot
    :return: None
    """
    if mode == "costs":
        plt.plot(costs)
        plt.xlabel("iterations")
        plt.title("cost function")
    elif mode == "linear_regression":
        plt.plot(data_x, data_y, "bo")
        plt.plot(x_i, y_i, "r")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Linear Regression")
    else:
        plt.plot(data_x, data_y, "bo", label="data")
        plt.title("Data")
        plt.legend()

    plt.show()


if __name__ == "__main__":
    path = "data_ml/data_linear_regression.txt"
    X, y, data_x, data_y = prepare_data(path)
    plot(data_x, data_y)
    print("Data preparation done!")
    w, costs = gradient_descent(X, y)
    plot(data_x, data_y, costs=costs, mode="costs")
    print("Gradient Descent computation done!")
    x_i, y_i = linear_regression(w)
    plot(data_x, data_y, x_i, y_i, costs, mode="linear_regression")
    print("Linear Regression computation done!")
