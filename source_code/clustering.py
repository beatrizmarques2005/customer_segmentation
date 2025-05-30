from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, MeanShift, SpectralClustering
from minisom import MiniSom
import pandas as pd
from sklearn.cluster import estimate_bandwidth
from scipy.cluster.hierarchy import dendrogram
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.base import ClusterMixin
import umap
import plotly.graph_objects as go
from sklearn.decomposition import PCA

def summarise_clusters(data: pd.DataFrame, cluster_col: str, exclude_cols: list = None, scaled: bool = False) -> pd.DataFrame:
    """
    Summarize cluster feature means, excluding specified columns.

    Args:
        data (pd.DataFrame): Input data including cluster labels.
        cluster_col (str): Name of the column containing cluster labels.
        exclude_cols (list, optional): Columns to exclude from the summary.
        scaled (bool): If True, use colored heatmap; if False, just print the DataFrame.

    Returns:
        pd.DataFrame: Transposed mean values for each cluster.
    """
    if exclude_cols is None:
        exclude_cols = []
        
    cols = [col for col in data.columns if col not in exclude_cols + [cluster_col]]
    summary = data.groupby(cluster_col)[cols].mean(numeric_only=True).T

    if scaled:
        colorscale = "viridis"
        fig = go.Figure(
            data=go.Heatmap(
                z=summary.values,
                x=summary.columns.astype(str),
                y=summary.index,
                colorscale=colorscale,
                colorbar=dict(title="Mean"),
                showscale=True,
                text=np.round(summary.values, 2),
                texttemplate="%{text}"
            )
        )
        fig.update_layout(
            title="Cluster Feature Means",
            xaxis_title="Cluster",
            yaxis_title="Features",
            autosize=True,
            margin=dict(l=40, r=40, t=60, b=40),
            height=800
        )
        fig.show()
    else:
        print(f'Cluster Summary (not scaled data):')
        display(summary)

    return summary

def visualize_clusters(
    data: pd.DataFrame,
    cluster_col: str,
    n_neighbors: int = None,
    min_dist: float = None,
    random_state: int = 42,
    exclude_cols: list = None
) -> None:
    """
    Visualize clusters in 2D using UMAP for dimensionality reduction.

    Args:
        data (pd.DataFrame): Input data including features and cluster labels.
        cluster_col (str): Name of the column containing cluster labels.
        n_neighbors (int): UMAP n_neighbors parameter.
        min_dist (float): UMAP min_dist parameter.
        random_state (int): Random seed for reproducibility.
        exclude_cols (list, optional): Columns to exclude from UMAP embedding.
    """
    if exclude_cols is None:
        exclude_cols = []
    features = data.drop(columns=[cluster_col] + exclude_cols)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    embedding = reducer.fit_transform(features)

    fig = go.Figure()
    for cluster, group in data.groupby(cluster_col):
        idx = group.index
        fig.add_trace(go.Scatter(
            x=embedding[idx, 0],
            y=embedding[idx, 1],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(size=8),
            text=idx.astype(str)
        ))

    fig.update_layout(
        title=f"Clusters Visualization (UMAP 2D): {cluster_col}",
        xaxis_title="UMAP-1",
        yaxis_title="UMAP-2",
        legend_title="Cluster",
        template="plotly_white"
    )
    fig.show()


#######################################
############ HIERARQUICAL #############
#######################################

def hierarchical_clustering(data: pd.DataFrame, n_clusters: int = None, exclude_cols: list = None) -> tuple[pd.DataFrame, AgglomerativeClustering]:
    data = data.copy()

    features = data.drop(columns=[col for col in exclude_cols if col in data.columns])

    if n_clusters is None:
        model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    else:
        model = AgglomerativeClustering(n_clusters=n_clusters)

    model.fit(features)
    data['cluster'] = model.labels_
    return data, model


def plot_dendrogram(model, y_line: float, **kwargs):
    """
    Plots a dendrogram for a fitted AgglomerativeClustering model with larger size, more y-axis labels, and a grid.
    Ensures only one figure is plotted.
    """
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

    fig, ax = plt.subplots(figsize=(18, 8))  # Use subplots to avoid multiple figures
    dendrogram(linkage_matrix, ax=ax, **kwargs)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Sample Index or (Cluster Size)")
    ax.set_yticks(np.linspace(0, np.max(model.distances_), 20))  # More y-axis labels
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.axhline(y=y_line, color='black', linestyle='--')  # Add horizontal line at y=27
    plt.tight_layout()
    plt.show()


#######################################
############### KMeans ################
#######################################

def kmeans_clustering(data: pd.DataFrame, data_scaled: pd.DataFrame,  n_clusters: int) -> pd.DataFrame:

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    data['KMeans'] = kmeans.fit_predict(data_scaled)

    #silhouette_score = silhouette_score(data_scaled, kmeans.fit_predict(data_scaled))
    #print(f"KMeans Silhouette Score: {silhouette_score:.4f}")
    return data

def plot_elbow(data: pd.DataFrame, max_k: int = 10, exclude_cols: list = None) -> None:
    """
    Plot the Elbow Method using Plotly, with option to exclude columns from clustering.

    Args:
        data (pd.DataFrame): Input data.
        max_k (int): Maximum number of clusters to try.
        exclude_cols (list, optional): Columns to exclude from clustering.
    """
    if exclude_cols is None:
        exclude_cols = []
    features = data.drop(columns=exclude_cols) if exclude_cols else data

    inertia = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)

    fig = go.Figure(
        data=go.Scatter(
            x=list(range(1, max_k + 1)),
            y=inertia,
            mode='lines+markers',
            marker=dict(size=10, color='royalblue'),
            line=dict(width=2),
            hovertemplate='<b>K = %{x}</b><br>Inertia = %{y:.2f}<extra></extra>'
        )
    )

    fig.update_layout(
        title="Elbow Method for Optimal K (Plotly)",
        xaxis_title="Number of Clusters (K)",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        xaxis=dict(dtick=1),
        template="plotly_white",
        hovermode='x unified'
    )
    fig.show()

'''def plot_elbow_kmeans(data: pd.DataFrame) -> None:


    dispersion = []
    for k in range(1, 50):
        kmeans = KMeans(n_clusters=k, random_state=0).fit(data)
        dispersion.append(kmeans.inertia_)

    plt.plot(range(1, 50), dispersion, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Dispersion (inertia)')
    plt.show()'''

####################################################
################ Silhouette Score ##################
####################################################

def test_multiple_clusters(data_scaled: pd.DataFrame, k_range: range, ):
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data_scaled)
        avg_score = silhouette_score(data_scaled, labels)
        print(f"k = {k}, silhouette score = {avg_score:.4f}")
        results.append((k, avg_score))

    return results

def plot_silhouette(data_scaled, model: ClusterMixin, n_clusters: int, title: str = None):
    """
    Plots a silhouette plot for any clustering algorithm.
    
    Parameters:
    - X: Scaled input data (np.array or pd.DataFrame)
    - model: A clustering model
    - n_clusters: Number of clusters used in the model
    - title: Optional plot title
    """
    # Fit the model and get cluster labels
    #if hasattr(model, "fit_predict"):
        
    cluster_labels = model.fit_predict(data_scaled)
    #else:
    #    model.fit(X)
    #    cluster_labels = model.labels_

    silhouette_avg = silhouette_score(data_scaled, cluster_labels)
    print(f"n_clusters = {n_clusters}, average silhouette_score = {silhouette_avg:.4f}")
    sample_silhouette_values = silhouette_samples(data_scaled, cluster_labels)

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 6)

    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(data_scaled) + (n_clusters + 1) * 10])

    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_title(title or f"Silhouette Plot (n_clusters={n_clusters})")
    ax1.set_xlabel("Silhouette Coefficient Values")
    ax1.set_ylabel("Cluster Label")
    ax1.set_yticks([])
    ax1.set_xticks(np.linspace(-0.1, 1.0, 12))

    plt.tight_layout()
    plt.show()

def avg_silhouette_score(data: pd.DataFrame) -> float:
    kmeans = KMeans(n_clusters=10, random_state=42)
    silhouette_avg = silhouette_score(data, kmeans.fit_predict(data))
    return silhouette_avg


#######################################
################ SOM ##################
#######################################

def som(data_np: np.ndarray,
                                        data: pd.DataFrame, 
                                        x: int, 
                                        y: int, 
                                        input_len: int, 
                                        sigma: float = 0.5,
                                        learning_rate: float = 1,
                                        neighborhood_function: str ='gaussian', 
                                        random_seed: int = 42,
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
    som.train_batch(data_np, number_of_iterations)

    data['winner_node'] = ([som.winner(data_np[i]) for i in range(len(data_np))])
    return data

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

def som_lattice(data_np: np.ndarray,
                                        x: int, 
                                        y: int, 
                                        input_len: int, 
                                        sigma: float = 0.5,
                                        learning_rate: float = 1,
                                        neighborhood_function: str ='gaussian', 
                                        random_seed: int = 42,
                                        number_of_iterations: int = 1000):

    som = MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate, random_seed=random_seed)
    som.train_batch(data_np, number_of_iterations)
    plt.pcolor(som.distance_map().T, cmap='bone_r')
    plt.colorbar()
    

#######################################
############### DBScan ################
#######################################

def dbscan_clustering(data: pd.DataFrame, eps: float, min_samples: int, exclude_cols: list = None) -> pd.DataFrame:
    """
    Perform DBSCAN clustering, with optional exclusion of columns.

    Args:
        data (pd.DataFrame): Input data.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
        exclude_cols (list, optional): Columns to exclude from clustering.

    Returns:
        pd.DataFrame: DataFrame with a new 'cluster' column.
    """
    if exclude_cols is None:
        exclude_cols = []
    features = data.drop(columns=exclude_cols) if exclude_cols else data

    model = DBSCAN(eps=eps, min_samples=min_samples)
    data['cluster'] = model.fit_predict(features)

    return data

def plot_dbscan_cluster_count_vs_eps(data: pd.DataFrame, min_samples: int, eps_values: list, exclude_cols: list = None) -> None:
    """
    Plots the number of clusters found by DBSCAN as a function of eps, with axes switched.

    Args:
        data (pd.DataFrame): The input data for clustering.
        min_samples (int): The min_samples parameter for DBSCAN.
        eps_values (list): List of eps values to try.
        exclude_cols (list, optional): Columns to exclude from clustering.
    """
    if exclude_cols is None:
        exclude_cols = []
    features = data.drop(columns=exclude_cols) if exclude_cols else data

    n_clusters_list = []
    for eps in eps_values:
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(features)
        # Exclude noise label (-1) from cluster count
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clusters_list.append(n_clusters)

    fig = go.Figure(
        data=go.Scatter(
            x=n_clusters_list,
            y=eps_values,
            mode='lines+markers',
            marker=dict(size=8),
            hovertemplate='Clusters=%{x}<br>eps=%{y:.2f}<extra></extra>'
        )
    )
    fig.update_layout(
        title=f"DBSCAN: eps vs Number of Clusters (min_samples={min_samples})",
        xaxis_title="Number of Clusters",
        yaxis_title="eps",
        template="plotly_white"
    )
    fig.show()


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

def spectral_clustering(data_scaled: pd.DataFrame, data: pd.DataFrame, n_clusters: int) -> pd.DataFrame:

    model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)

    data['cluster'] = model.fit_predict(data_scaled)
    

    return data



'''
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