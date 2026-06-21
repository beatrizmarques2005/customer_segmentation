from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from minisom import MiniSom
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
import umap
import plotly.graph_objects as go
import plotly.express as px

def summarise_clusters(data: pd.DataFrame, cluster_col: str, exclude_cols: list = None, scaled: bool = False) -> pd.DataFrame:
    """
    Computes the average values of numerical variables for each cluster 
    and optionally visualizes them as a heatmap.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing feature columns and cluster assignments.
    cluster_col : str
        The name of the column in `data` that indicates cluster labels 
        (e.g., 'cluster').
    exclude_cols : list, optional
        A list of columns to exclude from the averaging. These typically \n        include identifiers or non-numeric fields. Defaults to None.
    scaled : bool, optional
        If True, displays a heatmap of the cluster-wise feature means. \n        If False, prints the resulting DataFrame without visualization. \n        Defaults to False.

    Returns
    -------
    summary : pandas.DataFrame
        A DataFrame where rows are features and columns are clusters. \n        Each cell contains the mean value of a feature within a cluster.
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
    Visualizes clusters in a 2D space using UMAP for dimensionality reduction.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing features and cluster assignments.
    cluster_col : str
        The name of the column in `data` that indicates cluster labels \n        (e.g., 'cluster').
    n_neighbors : int, optional
        The number of neighboring points used in local approximations of the manifold \n        structure. Passed to the UMAP algorithm. Defaults to None (uses UMAP default).
    min_dist : float, optional
        The minimum distance between embedded points. Controls how tightly UMAP \n        packs points together. Defaults to None (uses UMAP default).
    random_state : int, optional
        Seed for reproducibility of UMAP embeddings. Defaults to 42.
    exclude_cols : list, optional
        A list of column names to exclude from the UMAP embedding process \n        (e.g., IDs or non-feature columns). Defaults to None.

    Returns
    -------
    None
        Displays an interactive 2D scatter plot of clusters using Plotly.
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

def cluster_sizes(data: pd.DataFrame, cluster_col: str, show_plot: bool = True) -> pd.Series:
    """
    Computes and optionally visualizes the number of records in each cluster.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing cluster assignments.
    cluster_col : str
        The name of the column in `data` that indicates cluster labels \n        (e.g., 'cluster').
    show_plot : bool, optional
        If True, displays a bar chart showing the number of records per cluster. \n        If False, only returns the cluster size counts. Defaults to True.

    Returns
    -------
    sizes : pandas.Series
        A Series indexed by cluster label, containing the count of records in each cluster.
    """
    sizes = data[cluster_col].value_counts().sort_index()

    fig = go.Figure(
        data=go.Bar(
            x=sizes.index.astype(str),
            y=sizes.values,
            marker_color='skyblue'
        )
    )
    fig.update_layout(
        title='Cluster Sizes',
        xaxis_title='Cluster',
        yaxis_title='Number of IDs',
        template='plotly_white'
    )
    fig.show()

def map_visualization(
    data: pd.DataFrame,
    lat_col: str = 'latitude',
    lon_col: str = 'longitude',
    zoom: int = 3,
    height: int = 500,
    title: str = "Map Visualization"
) -> None:
    """
    Displays a map with points plotted based on latitude and longitude coordinates.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing geographic coordinates.
    lat_col : str, optional
        The name of the column in `data` that contains latitude values. \n        Defaults to 'latitude'.
    lon_col : str, optional
        The name of the column in `data` that contains longitude values. \n        Defaults to 'longitude'.
    zoom : int, optional
        Initial zoom level for the map view. Higher values zoom in further. \n        Defaults to 3.
    height : int, optional
        Height of the map figure in pixels. Defaults to 500.
    title : str, optional
        Title displayed above the map. Defaults to "Map Visualization".

    Returns
    -------
    None
        Displays an interactive map with points using Plotly and OpenStreetMap tiles.
    """
    fig = px.scatter_mapbox(
        data,
        lat=lat_col,
        lon=lon_col,
        zoom=zoom,
        height=height
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        title=title
    )
    fig.show()

def reassign_clusters(data_clusters: pd.DataFrame, data_cluster_notscaled: pd.DataFrame, dict_num_cluster: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reassigns cluster labels in both scaled and unscaled DataFrames based on a mapping dictionary.

    Parameters
    ----------
    data_clusters : pandas.DataFrame
        DataFrame containing scaled data with existing cluster assignments in a column named 'cluster'.
    data_cluster_notscaled : pandas.DataFrame
        DataFrame containing unscaled data with cluster assignments in a column named 'cluster'.
    dict_num_cluster : dict
        Dictionary mapping old cluster labels to new cluster labels \n        (e.g., {0: 2, 1: 0, 2: 1}).

    Returns
    -------
    data_clusters : pandas.DataFrame
        The scaled DataFrame with cluster labels reassigned according to `dict_num_cluster`.
    data_cluster_notscaled : pandas.DataFrame
        The unscaled DataFrame with cluster labels reassigned according to `dict_num_cluster`.
    """
    data_clusters = data_clusters.copy()
    data_cluster_notscaled = data_cluster_notscaled.copy()
    data_clusters['cluster'] = data_clusters['cluster'].replace(dict_num_cluster)
    data_cluster_notscaled['cluster'] = data_cluster_notscaled['cluster'].replace(dict_num_cluster)

    return data_clusters, data_cluster_notscaled

def assign_excluded_ids_to_clusters(
    excluded_ids_no_makro_notscaled: pd.DataFrame,
    clustering_notscaled: pd.DataFrame
) -> pd.DataFrame:
    """
    Assigns clusters to excluded data points based on the nearest cluster centroid
    using the not-scaled dataset.

    Parameters
    ----------
    excluded_ids_no_makro : pandas.DataFrame
        DataFrame containing the data points that were excluded from the original clustering \n        (e.g., without Makro), including all relevant feature columns.
    clustering_notscaled : pandas.DataFrame
        DataFrame containing the original clustering results (not scaled), including a 'cluster' column \n        and all relevant feature columns for centroid calculation.

    Returns
    -------
    excluded_ids_no_makro_with_cluster : pandas.DataFrame
        A copy of `excluded_ids_no_makro` with an added 'cluster' column, assigning each row \n        to the closest cluster based on feature space distance.
    """

    if excluded_ids_no_makro_notscaled.empty or clustering_notscaled.empty:
        raise ValueError("Input DataFrames cannot be empty.")

    if 'cluster' not in clustering_notscaled.columns:
        raise ValueError("'cluster' column is missing in clustering_notscaled.")

    exclude_cols = ['customer_id', 'has_loyalty_card', 'longitude', 'latitude', 'gender']

    feature_cols_for_distance = [col for col in clustering_notscaled.columns if col not in exclude_cols + ['cluster']]

    centroids = clustering_notscaled.groupby('cluster')[feature_cols_for_distance].mean().values

    excluded_ids_features = excluded_ids_no_makro_notscaled[feature_cols_for_distance].values

    distances = cdist(excluded_ids_features, centroids)
    closest_clusters = distances.argmin(axis=1)

    excluded_ids_no_makro_notscaled_with_cluster = excluded_ids_no_makro_notscaled.copy()
    excluded_ids_no_makro_notscaled_with_cluster['cluster'] = closest_clusters

    return excluded_ids_no_makro_notscaled_with_cluster


#######################################
############ HIERARCHICAL #############
#######################################

def hierarchical_clustering(data: pd.DataFrame, n_clusters: int = None, exclude_cols: list = None) -> tuple[pd.DataFrame, AgglomerativeClustering]:
    """
    Performs hierarchical agglomerative clustering on the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset containing features for clustering.
    n_clusters : int, optional
        The number of clusters to find. If None, the clustering is performed using \n        a distance threshold to create a full dendrogram (no preset number of clusters). \n        Defaults to None.
    exclude_cols : list, optional
        List of columns to exclude from clustering (e.g., identifiers or non-numeric columns). \n        Defaults to None.

    Returns
    -------
    data : pandas.DataFrame
        A copy of the input DataFrame with an added 'cluster' column containing cluster labels.
    model : sklearn.cluster.AgglomerativeClustering
        The fitted AgglomerativeClustering model instance.
    """
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
    Plots a dendrogram for a fitted AgglomerativeClustering model with enhanced formatting.

    Parameters
    ----------
    model : sklearn.cluster.AgglomerativeClustering
        A fitted AgglomerativeClustering model containing hierarchical clustering info.
    y_line : float
        The y-axis value at which to draw a horizontal reference line (e.g., to indicate cluster cutoff).
    **kwargs : dict
        Additional keyword arguments to pass to `scipy.cluster.hierarchy.dendrogram`.

    Returns
    -------
    None
        Displays a dendrogram plot with increased figure size, more y-axis ticks, a grid, \n        and a horizontal reference line at `y_line`.
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

def kmeans_clustering(data_scaled: pd.DataFrame, n_clusters: int, exclude_cols: list = None) -> pd.DataFrame:
    """
    Performs KMeans clustering on scaled data with optional exclusion of specified columns.

    Parameters
    ----------
    data_scaled : pandas.DataFrame
        The scaled dataset used for clustering.
    n_clusters : int
        The number of clusters to form.
    exclude_cols : list, optional
        List of column names to exclude from clustering (e.g., identifiers or non-feature columns). \n        Defaults to None.

    Returns
    -------
    result : pandas.DataFrame
        A copy of the input DataFrame with an added 'cluster' column containing cluster labels.
    centroids : pandas.DataFrame
        A DataFrame of cluster centroids for all columns, including excluded columns \n        (where centroid values are the mean of those columns within each cluster).
    """
    if exclude_cols is None:
        exclude_cols = []

    features = data_scaled.drop(columns=exclude_cols, errors='ignore') if exclude_cols else data_scaled

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)

    result = data_scaled.copy()
    result['cluster'] = cluster_labels

    # Prepare centroids DataFrame with all columns (excluded columns use their mean values)
    all_columns = list(data_scaled.columns)
    centroids = pd.DataFrame(index=range(n_clusters), columns=all_columns)

    # Set centroids for features used in clustering
    centroids.loc[:, features.columns] = kmeans.cluster_centers_

    # For excluded columns, use their mean values from the original data
    for col in exclude_cols:
        if col in data_scaled.columns:
            col_means = result.groupby('cluster')[col].mean().values
            centroids[col] = col_means

    return result, centroids

def plot_elbow(data: pd.DataFrame, max_n_cluster: int = 10, exclude_cols: list = None) -> None:
    """
    Plots the Elbow Method graph to help determine the optimal number of clusters for KMeans.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to use for clustering.
    max_n_cluster : int, optional
        The maximum number of clusters to evaluate (from 1 to `max_n_cluster`). Defaults to 10.
    exclude_cols : list, optional
        List of columns to exclude from clustering (e.g., non-feature or identifier columns). \n        Defaults to None.

    Returns
    -------
    None
        Displays an interactive Plotly line chart showing inertia values for each cluster count.
    """
    if exclude_cols is None:
        exclude_cols = []
    features = data.drop(columns=exclude_cols) if exclude_cols else data

    inertia = []
    for n_cluster in range(1, max_n_cluster + 1):
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        kmeans.fit(features)
        inertia.append(kmeans.inertia_)

    fig = go.Figure(
        data=go.Scatter(
            x=list(range(1, max_n_cluster + 1)),
            y=inertia,
            mode='lines+markers',
            marker=dict(size=10, color='royalblue'),
            line=dict(width=2),
            hovertemplate='<b>Number of Clusters = %{x}</b><br>Inertia = %{y:.2f}<extra></extra>'
        )
    )

    fig.update_layout(
        title="Elbow Method for Optimal K (Plotly)",
        xaxis_title="Number of Clusters",
        yaxis_title="Inertia (Within-Cluster Sum of Squares)",
        xaxis=dict(dtick=1),
        template="plotly_white",
        hovermode='x unified'
    )
    fig.show()

def avg_silhouette_score(data: pd.DataFrame, n_cluster: int, exclude_cols: list = None) -> float:
    """
    Calculates the average silhouette score for KMeans clustering on the dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to cluster.
    n_cluster : int
        The number of clusters to form.
    exclude_cols : list, optional
        List of columns to exclude from clustering (e.g., identifiers or non-feature columns). \n        Defaults to None.

    Returns
    -------
    silhouette_avg : float
        The average silhouette score, measuring how well samples are clustered \n        (higher values indicate better clustering).
    """
    if exclude_cols is None:
        exclude_cols = []

    features = data.drop(columns=exclude_cols) if exclude_cols else data

    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    silhouette_avg = silhouette_score(features, kmeans.fit_predict(features))

    return silhouette_avg

def plot_silhouette(data: pd.DataFrame, cluster_col: str, exclude_cols: list = None, title: str = None):
    """
    Creates a silhouette plot to visualize the quality of clustering.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame containing features and cluster assignments.
    cluster_col : str
        Name of the column with cluster labels.
    exclude_cols : list, optional
        List of columns to exclude from features before computing silhouette scores. Defaults to None.
    title : str, optional
        Title of the silhouette plot. Defaults to a generic title if not provided.

    Returns
    -------
    None
        Displays an interactive Plotly silhouette plot with silhouette scores for each cluster,\n        including an average silhouette score reference line.
    """
    if exclude_cols is None:
        exclude_cols = []
    features = data.drop(columns=[cluster_col] + exclude_cols, errors='ignore')
    cluster_labels = data[cluster_col].values
    n_clusters = len(np.unique(cluster_labels))

    if n_clusters < 2:
        print("Silhouette plot requires at least 2 clusters.")
        return

    silhouette_avg = silhouette_score(features, cluster_labels)
    sample_silhouette_values = silhouette_samples(features, cluster_labels)

    # Prepare data for Plotly
    y_lower = 10
    traces = []
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        y_vals = np.arange(y_lower, y_upper)
        color = f"hsl({int(360 * i / n_clusters)},70%,50%)"
        traces.append(
            go.Scatter(
                x=ith_cluster_silhouette_values,
                y=y_vals,
                mode='lines',
                fill='tozerox',
                line=dict(color=color),
                name=f'Cluster {i}',
                showlegend=True
            )
        )
        y_lower = y_upper + 10

    # Add average silhouette line
    traces.append(
        go.Scatter(
            x=[silhouette_avg, silhouette_avg],
            y=[0, y_lower],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Average Silhouette'
        )
    )

    layout = go.Layout(
        title=title or f"Silhouette Plot (n_clusters={n_clusters})",
        xaxis=dict(title="Silhouette Coefficient Values", range=[-0.1, 1.0], dtick=0.1),
        yaxis=dict(title="Sample Index", showticklabels=False),
        legend=dict(title="Cluster"),
        height=600,
        template="plotly_white"
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.show()

#######################################
######## Spectral Clustering ##########
#######################################

def spectral_clustering(
    data: pd.DataFrame,
    n_clusters: int,
    exclude_cols: list = None,
    assign_labels: str = "kmeans",
    affinity: str = "nearest_neighbors",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Performs Spectral Clustering on the dataset with optional exclusion of specified columns.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to cluster.
    n_clusters : int
        The number of clusters to form.
    exclude_cols : list, optional
        List of columns to exclude from clustering (e.g., identifiers or non-feature columns). Defaults to None.
    assign_labels : str, optional
        The strategy to assign labels after clustering. Options are 'kmeans' or 'discretize'. Defaults to 'kmeans'.
    affinity : str, optional
        How to construct the affinity matrix. Options include 'nearest_neighbors', 'rbf', etc. Defaults to 'nearest_neighbors'.
    random_state : int, optional
        Seed for random number generation for reproducibility. Defaults to 42.

    Returns
    -------
    result : pandas.DataFrame
        Copy of the input DataFrame with an added 'cluster' column containing cluster labels.
    """
    if exclude_cols is None:
        exclude_cols = []
    features = data.drop(columns=exclude_cols, errors='ignore') if exclude_cols else data

    model = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        assign_labels=assign_labels,
        random_state=random_state
    )
    cluster_labels = model.fit_predict(features)

    result = data.copy()
    result['cluster'] = cluster_labels

    return result