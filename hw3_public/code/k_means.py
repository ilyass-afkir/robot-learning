import matplotlib.pyplot as plt
import numpy as np


def k_means(data):
    """
    The k_means function performs k-means clustering on the data and returns
    the following:
        - data_ : The original data with predicted cluster labels appended to it.
        - predicted_clusters_ : The actual cluster labels of the clusters.
        - n_data_pt : Number of data points

    :param data: Data that is to be clustered
    :return: The data and the predicted clusters
    """
    max_iterations = 100
    n_clusters = 5
    n_data_pt = data.shape[0]
    n_features = data.shape[1]
    plot(data, n_data_pt)
    data_, predicted_clusters_ = fit(
        data, n_clusters, n_features, max_iterations, n_data_pt
    )

    return data_, predicted_clusters_, n_data_pt


def fit(data, n_clusters, n_features, max_iterations, n_data_pt):
    """
    The fit function takes in the data and number of clusters for K-means clustering.
    The function returns the predicted labels for each sample, as well as the centroids of each cluster.

    :param data: Data that is to be clustered
    :param n_clusters: Number of clusters to be used
    :param n_features: Number of features that will be used to fit the data
    :param max_iterations: Maximum number of iterations
    :param n_data_pt: Number of data points to be used
    :return: The data and the predicted clusters
    """

    # TODO write your own solution here
    predicted_clusters = np.zeros(n_data_pt)

    # init random cluster centers in range of the random data
    cluster_centers = np.random.rand(n_clusters, n_features)
    for i in range(n_features):
        data_min = np.nanmin(data[:, i])
        data_max = np.nanmax(data[:, i])
        cluster_centers[i, :] = data_min * np.ones((1, n_features)) + (data_max - data_min) * cluster_centers[i, :]

    old_cluster_centers = cluster_centers.copy()

    # iterate
    for i in range(max_iterations):

        # assign points to clusters
        for d in range(n_data_pt):
            distances = np.asarray([np.linalg.norm(data[d] - cluster_centers[j, :]) for j in range(n_clusters)])
            min_dist = np.nanmin(distances)
            for e in range(n_clusters):
                if distances[e] == min_dist:
                    predicted_clusters[d] = e

        # calculate new cluster centers
        for c in range(n_clusters):
            cluster_centers[c, :] = np.zeros(n_features)
            number_of_points = 0

            for p in range(n_data_pt):
                if predicted_clusters[p] == c:
                    cluster_centers[c, :] = cluster_centers[c, :] + data[p]
                    number_of_points += 1

            cluster_centers[c, :] = cluster_centers[c, :] * (1/float(number_of_points))

        # check if cluster centers changed
        if all(cluster_centers.flatten() == old_cluster_centers.flatten()):
            break
        else:
            old_cluster_centers = cluster_centers.copy()

    return data, predicted_clusters


def plot(data, n_data_pt, predicted_clusters=None):
    """
    The plot function takes as input the data, the number of examples in that data, and an optional argument
    predicted_clusters. If predicted_clusters is not provided then it defaults to a vector of zeros.

    :param data: Pass the data to be plotted
    :param n_data_pt: Number of data points to plot
    :param predicted_clusters=None: Plot the data points with their true labels
    :return: Nothing, it just plots the data
    """
    colors = ["royalblue", "orange", "limegreen", "red", "peru"]
    clusters_colors = []

    if predicted_clusters is None:
        predicted_clusters = np.zeros(n_data_pt)

    for i in range(len(predicted_clusters)):
        clusters_colors.append(colors[int(predicted_clusters[i])])

    plt.scatter(data[:, 0], data[:, 1], c=clusters_colors, s=20)
    plt.axis([-14, 10, -15, 6])
    plt.show()


if __name__ == "__main__":
    np.random.seed(10)
    data = np.loadtxt("data_ml/data_kmeans.txt")
    data_, predicted_clusters_, n_data_pt = k_means(data)
    plot(data_, n_data_pt, predicted_clusters_)
