from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, MeanShift, SpectralClustering
from minisom import MiniSom
import pandas as pd
from sklearn.cluster import estimate_bandwidth
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score


def summarise_clusters(data: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    summary = data.groupby(cluster_col).mean().T

    plt.figure(figsize=(10, min(1 + 0.5 * summary.shape[0], 12)))
    sns.heatmap(summary, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Cluster Feature Means")
    plt.ylabel("Features")
    plt.xlabel("Cluster")
    plt.tight_layout()
    plt.show()

    return summary

#######################################
############ HIERARQUICAL #############
#######################################

def hierarchical_clustering(data: pd.DataFrame, n_clusters: int = None) -> tuple[pd.DataFrame, AgglomerativeClustering]:
    data = data.copy()

    if n_clusters is None:
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)

    model.fit(data)
    data['cluster'] = model.labels_
    return data, model


def plot_dendrogram(model, **kwargs):

    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)

    for i, merge in enumerate(model.children_):
        current_count = 0

        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]

        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    dendrogram(linkage_matrix, **kwargs)


#######################################
############### KMeans ################
#######################################

def kmeans_clustering(data: pd.DataFrame, data_scaled: pd.DataFrame,  n_clusters: int) -> pd.DataFrame:

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    data['KMeans'] = kmeans.fit_predict(data_scaled)

    return data

def test_multiple_clusters(data_scaled: pd.DataFrame, k_range: range):
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        score = silhouette_score(data_scaled, labels)
        print(f"k = {k}, silhouette score = {score:.4f}")
        results.append((k, score))

    return results

#######################################
################ SOM ##################
#######################################

def som_clustering(data: pd.DataFrame, x: int = 10, y: int = 10, input_len: int = None, sigma: float = 1.0, learning_rate: float = 0.5, iterations: int = 100) -> pd.DataFrame:

    if input_len is None:
        input_len = data.shape[1]

    som = MiniSom(x, y, input_len, sigma=sigma, learning_rate=learning_rate)

    som.random_weights_init(data.values)

    som.train_random(data.values, iterations)

    data['cluster'] = [som.winner(row) for row in data.values]

    return data


#######################################
############### DBScan ################
#######################################

def dbscan_clustering(data, eps: float, min_samples: int) -> pd.DataFrame:

    model = DBSCAN(eps=eps, min_samples=min_samples)

    data['cluster'] = model.fit_predict(data)

    return data

#######################################
############# Mean Shift ##############
#######################################

def mean_shift_clustering(data: pd.DataFrame, quantile: float, n_samples: int) -> pd.DataFrame:

    bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=n_samples)

    model = MeanShift(bandwidth=bandwidth)

    data['cluster'] = model.fit_predict(data)

    return data

#######################################
######## Spectral Clustering ##########
#######################################

def spectral_clustering(data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:

    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)

    data['cluster'] = model.fit_predict(data)

    return data



'''
# Mean Shift
def check_multidimensional_outliers_mean_shift(customer_info: pd.DataFrame, nr_obs_small_clusters: int) -> None:
    """
    Identifies multidimensional outliers using the Mean Shift clustering algorithm and visualizes the results.

    Parameters:
    -----------
    customer_info : pd.DataFrame
        A pandas DataFrame containing numerical features to analyze for multidimensional outliers.
    
    nr_obs_small_clusters : int
        The threshold for the minimum number of observations in a cluster to be considered valid.
        Below this threshold, the cluster is considered small, therefore the point is an outlier.

    Returns:
    --------
    None
        The function displays a scatter plot with clusters and highlights potential outliers.
    """

    mean_shift = MeanShift()
    mean_shift.fit(customer_info)

    customer_info['cluster_mean_shift'] = mean_shift.labels_

    cluster_sizes = customer_info['cluster_mean_shift'].value_counts()
    small_clusters = cluster_sizes[cluster_sizes < nr_obs_small_clusters].index  # Threshold
    customer_info['is_outlier_mean_shift'] = customer_info['cluster_mean_shift'].isin(small_clusters)

    return customer_info

# Isolation Forest
def check_multidimensional_outliers_isolation_forest(customer_info: pd.DataFrame, nr_obs_small_clusters: int) -> None:
    """
    Identifies multidimensional outliers using the Isolation Forest algorithm and visualizes the results.

    Parameters:
    -----------
    customer_info : pd.DataFrame
        A pandas DataFrame containing numerical features to analyze for multidimensional outliers.
    
    nr_obs_small_clusters : int
        The threshold for the minimum number of observations in a cluster to be considered valid.
        Below this threshold, the cluster is considered small, therefore the point is an outlier.

    Returns:
    --------
    None
        The function displays a scatter plot with clusters and highlights potential outliers.
    """

    isolation_forest = IsolationForest(contamination=0.1)
    customer_info['is_outlier_isolation_forest'] = isolation_forest.fit_predict(customer_info) == -1

    return customer_info

# SOM
def check_multidimensional_outliers_som(data: pd.DataFrame, 
                                        x: int, 
                                        y: int, 
                                        input_len: int, 
                                        sigma: float = 1.0,
                                        learning_rate: float = 0.5, 
                                        random_seed: int = None,
                                        number_of_iterations: int = 1000) -> MiniSom:
    """
    Train a Self-Organizing Map (SOM) using the given data.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data to train the SOM.
        x (int): The number of rows in the SOM grid.
        y (int): The number of columns in the SOM grid.
        input_len (int): The number of features in the input data.
        sigma (float, optional): The spread of the neighborhood function. Default is 1.0.
        learning_rate (float, optional): The initial learning rate. Default is 0.5.
        random_seed (int, optional): The seed for random number generation. Default is None.
        number_of_iterations (int, optional): The number of iterations for training. Default is 1000.

    Returns:
        MiniSom: The trained SOM model.
    """
    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate, random_seed=random_seed)
    som.random_weights_init(data)
    som.train_random(data, num_iteration=number_of_iterations)

def som_mean_clusters(data, col):
    """
    Calculate the mean of a specified column grouped by SOM winner nodes.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        col (str): The column name for which the mean is calculated.

    Returns:
        pd.DataFrame: A DataFrame with the mean values of the specified column grouped by winner nodes.
    """
    grouped = data.groupby(['winner_node'], as_index=False)[col].mean().sort_values(by=[col]).round(2)
    return grouped

def visualize_data_points_grid(data, scaled_data, som_model, color_variable, color_dict):
    """
    Visualize data points on a SOM grid with a distance map in the background.

    Args:
        data (pd.DataFrame): The input DataFrame containing the data.
        scaled_data (np.ndarray): The scaled data used for SOM training.
        som_model (minisom.MiniSom): The trained SOM model.
        color_variable (str): The column name used for coloring data points.
        color_dict (dict): A dictionary mapping unique values in the color_variable to colors.

    Returns:
        None: Displays a scatter plot with the SOM grid.
    """
    target = data[color_variable]
    fig, ax = plt.subplots()

    # Get weights for SOM winners
    w_x, w_y = zip(*[som_model.winner(d) for d in scaled_data])
    w_x = np.array(w_x)
    w_y = np.array(w_y)

    # Plot distance map in the background
    plt.pcolor(som_model.distance_map().T, cmap='bone_r', alpha=.2)
    plt.colorbar()

    # Plot data points with random perturbation to avoid overlap
    for c in np.unique(target):
        idx_target = target == c
        plt.scatter(
            w_x[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
            w_y[idx_target] + .5 + (np.random.rand(np.sum(idx_target)) - .5) * .8,
            s=50, c=color_dict[c], label=c
        )

    ax.legend(bbox_to_anchor=(1.2, 1.05))
    plt.grid()
    plt.show()

def plot_feature_influence(trained_som, data):
    """
    Plot the influence of each feature on the SOM nodes.

    Args:
        trained_som (minisom.MiniSom): The trained SOM model.
        data (pd.DataFrame): The input DataFrame containing the data.

    Returns:
        None: Displays a grid of plots showing feature influence.
    """
    feature_names = data.columns
    W = trained_som.get_weights()

    plt.figure(figsize=(10, 10))
    for i, f in enumerate(feature_names):
        plt.subplot(5, 5, i + 1)
        plt.title(f)
        plt.pcolor(W[:, :, i].T, cmap='coolwarm')
        plt.xticks(np.arange(15 + 1))
        plt.yticks(np.arange(15 + 1))
    plt.tight_layout()
    plt.show()

def plot_most_important_variable(trained_som, features):
    """
    Plot the most important variable for each SOM unit.

    Args:
        trained_som (minisom.MiniSom): The trained SOM model.
        features (list): List of feature names used in SOM training.

    Returns:
        None: Displays a plot showing the most important variable for each SOM unit.
    """
    W = trained_som.get_weights()

    plt.figure(figsize=(8, 8))
    for i in np.arange(W.shape[0]):
        for j in np.arange(W.shape[1]):
            feature = np.argmax(W[i, j, :])
            plt.plot([i + .5], [j + .5], 'o', color='C' + str(feature),
                     marker='s', markersize=24)

    legend_elements = [
        Patch(facecolor='C' + str(i), edgecolor='w', label=f) for i, f in enumerate(features)
    ]

    plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, .95))
    plt.xlim([0, 15])
    plt.ylim([0, 15])
    plt.show()

def remove_outliers(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Removes outliers from the customer information DataFrame based on predefined conditions.

    Parameters:
    -----------
    customer_info : pd.DataFrame
        The input DataFrame containing customer information, including various columns.

    Returns:
    --------
    pd.DataFrame
        A filtered DataFrame containing only rows that satisfy the outlier conditions.
    """
    outlier_conditions = (
        (customer_info['kids_home'] <= 8) &
        (customer_info['teens_home'] <= 4) &
        (customer_info['number_complaints'] <= 4) &
        (customer_info['distinct_stores_visited'] <= 8) &
        (customer_info['lifetime_spend_groceries'] <= 100000) &
        (customer_info['lifetime_spend_electronics'] <= 20000) &
        (customer_info['lifetime_spend_vegetables'] <= 2600) &
        (customer_info['lifetime_spend_nonalcohol_drinks'] <= 1400) &
        (customer_info['lifetime_spend_alcohol_drinks'] <= 2800) &
        (customer_info['lifetime_spend_meat'] <= 2600) &
        (customer_info['lifetime_spend_fish'] <= 2800) &
        (customer_info['lifetime_spend_hygiene'] <= 2400) &
        (customer_info['lifetime_spend_videogames'] <= 1700) &
        (customer_info['lifetime_spend_petfood'] <= 850) &
        (customer_info['lifetime_total_distinct_products'] <= 600) &
        (customer_info['percentage_of_products_bought_promotion'] >= -0.5) &
        (customer_info['percentage_of_products_bought_promotion'] <= 1.5)
    )
    return customer_info[outlier_conditions]

# Elbow method

def elbow_method(X, k_range=range(1, 11), plot=True, random_state=42):
    """
    Performs the Elbow Method to help choose the optimal number of clusters for K-Means.
    
    Parameters:
    - X (np.ndarray or pd.DataFrame): Input data.
    - k_range (iterable): Range of K values to try (default: 1 to 10).
    - scale_data (bool): Whether to scale the data using StandardScaler (recommended).
    - plot (bool): Whether to display the elbow plot.
    - random_state (int): Random seed for reproducibility.
    
    Returns:
    - distortions (list): Sum of squared distances for each K.
    """
    
    distortions = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, 
                        n_init=10, 
                        max_iter=300, 
                        random_state=random_state,
                        algorithm='elkan')  # faster for dense data
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)  # Sum of squared distances to closest cluster center
    
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(list(k_range), distortions, marker='o')
        plt.xlabel('Number of clusters (K)')
        plt.ylabel('Inertia (Sum of Squared Distances)')
        plt.title('Elbow Method For Optimal K')
        plt.xticks(list(k_range))
        plt.grid(True)
        plt.show()
    
    return distortions

'''