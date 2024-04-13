import matplotlib.pyplot as plt
import numpy as np


def prepare_data(path_train_data, path_valid_data):
    """
    The prepare_data function loads the training and validation data from a file,
    and returns the x_train, y_train, x_valid, y_valid and x-axis values.


    :param path_train_data: Path to the training data file
    :param path_valid_data: Path to the validation data file
    :return: x_train, y_train, x_valid, y_valid, x_pred
    """
    train = np.loadtxt(path_train_data)
    valid = np.loadtxt(path_valid_data)

    x_train = train[0, :]
    y_train = train[1, :]
    x_valid = valid[0, :]
    y_valid = valid[1, :]
    x_pred = np.linspace(0.0, 6.0, 601)

    plt.figure(0)
    plt.plot(x_train, y_train, "o", label="Training Set")

    return x_train, y_train, x_valid, y_valid, x_pred


def compute_LLS(x_train, y_train, x_pred):
    """
    The compute_LLS function computes the least-squares fit of a polynomial
    of degree n to the data in x_train and y_train. It then plots this fit for
    the given values of x.

    :param x_train: Training data
    :param y_train: Training data labels
    :param x_pred: x values for which labels are predicted
    :plot: plot the predictions of the lls model
    """

    # Aufgabe c)
    plt.figure(1)
    plt.plot(x_train, y_train, "o", label="Training Set")

    for n in [2, 3, 9]:
        features = np.asarray([np.sin(2**i*x_train) for i in range(n)]).T
        features_pred = np.asarray([np.sin(2**i*x_pred) for i in range(n)]).T

        # create design matrix
        phi = np.full((len(x_train), n), features)
        phi_pred = np.full((len(x_pred), n), features_pred)

        # model fitting
        theta = np.linalg.pinv(phi) @ y_train

        # prediction
        y_pred = phi_pred @ theta

        # plot
        plt.plot(x_pred, y_pred, label="n = {}".format(n))

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Linear Least Squares Fit")
    plt.savefig("lls.pdf")




def train_and_select_model(x_train, y_train, x_valid, y_valid):
    """
    The train_and_select_model function trains a model with the Mean Absolute Error. It shows the best model
    based on the training set and validation set and the model along with some other useful
    information about how well it performed. The function takes as input four numpy arrays: x_train, y_train,
    x_valid, and y_valid.

    :param x_train: Training data
    :param y_train: Training data labels
    :param x_valid: Validation data
    :param y_valid: Validation data labels
    :plot: Plot MAE on Training and validation set
    """
    
    # Aufgabe d) + e)

    mae_train = []
    mae_valid = []
    for n in range(1, 10):
        features = np.asarray([np.sin(2**i*x_train) for i in range(n)]).T
        features_pred_val = np.asarray([np.sin(2**i*x_valid) for i in range(n)]).T

        # create design matrix
        phi = np.full((len(x_train), n), features)
        phi_pred_val = np.full((len(x_valid), n), features_pred_val)

        # model fitting
        theta = np.linalg.pinv(phi) @ y_train

        # prediction
        y_pred = phi @ theta
        y_pred_val = phi_pred_val @ theta

        # calculate MAE
        mae_train.append(np.mean(np.abs(y_train - y_pred)))
        mae_valid.append(np.mean(np.abs(y_valid - y_pred_val)))

    plt.figure(2)
    plt.plot(np.arange(1, 10),  mae_train, "r", marker = "o")
    plt.title("MAE for the training set")
    plt.xlabel('number of features')
    plt.ylabel('MAE') 
    plt.savefig("MAE_train.pdf")
    
    #Aufgabe e)
    # Plot  
    plt.figure(3)
    plt.plot(np.arange(1, 10),  mae_train, "b", marker = "o", label="training set")
    plt.plot(np.arange(1, 10),  mae_valid, "r", marker = "o", label="validation set")
    plt.legend()
    plt.title("Model Selection")
    plt.xlabel('number of features')
    plt.ylabel('MAE') 
    plt.savefig("MAE_valid.pdf")


def k_fold(x_train, y_train):
    """
    The k_fold function takes in the training data and splits it into k parts.
    It then iterates through each part as the validation set, and uses all other
    parts as training data. It returns a matrix where each row is
    the average MAE.

    :param x_train: Training data
    :param y_train: Training data labels
    :plot: Plot the results
    """

    # Aufgabe f)
    x_train_copy = x_train.copy()
    y_train_copy = y_train.copy()
    k = len(x_train)
    mae_valid = np.zeros((k, 9))
    for l in range(k):
        x_valid = x_train_copy[l]
        y_valid = y_train_copy[l]
        x_train = np.delete(x_train_copy, l)
        y_train = np.delete(y_train_copy, l)
        for n in range(1, 10):
            features = np.asarray([np.sin(2**i*x_train) for i in range(n)]).T
            features_pred_val = np.asarray([np.sin(2**i*x_valid) for i in range(n)]).T

            # create design matrix
            phi = np.full((len(x_train), n), features)
            phi_pred_val = np.full((len([x_valid]), n), features_pred_val)

            # model fitting
            theta = np.linalg.pinv(phi) @ y_train

            # prediction
            y_pred_val = phi_pred_val @ theta

            # calculate MAE
            mae_valid[l, n-1] = np.mean(np.abs(y_valid - y_pred_val))
       
    mean_mae = np.mean(mae_valid, axis=0)
    vars_mae = np.var(mae_valid, axis=0)
    plt.figure(4)
    plt.errorbar(np.arange(1, 10),  mean_mae, vars_mae, marker = "o")
    plt.title("cross validation")
    plt.xlabel('number of features')
    plt.ylabel('MAE')
    plt.savefig("cross_validation.pdf")


if __name__ == "__main__":
    path_train_data = "data_ml/training_data.txt"
    path_valid_data = "data_ml/validation_data.txt"
    x_train, y_train, x_valid, y_valid, x_lin = prepare_data(
        path_train_data, path_valid_data
    )
    compute_LLS(x_train, y_train, x_lin)
    train_and_select_model(x_train, y_train, x_valid, y_valid)
    k_fold(x_train, y_train)
