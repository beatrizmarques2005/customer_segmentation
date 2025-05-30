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
import plotly.express as px

#################################################################
######################### Profile Chart #########################
#################################################################

# Computing necessary parameters for profile chart

def compute_average_multi(df, cluster_column):
    """
    Compute average values for each cluster and the entire dataset.

    Parameters:
    - df: pandas DataFrame
    - cluster_column: name of the column that identifies clusters (e.g., 'cluster')

    Returns:
    - variables: list of variable names
    - cluster_averages: DataFrame where each row is a cluster and columns are mean values
    - database_avg: list of mean values for the entire dataset
    """

    # Exclude the cluster column to get only variable columns
    variable_cols = df.columns.difference([cluster_column, 'customer_id'])

    # Compute cluster averages
    cluster_averages = df.groupby(cluster_column)[variable_cols].mean()

    # Compute database average
    database_avg = df[variable_cols].mean().values.tolist()

    return variable_cols.tolist(), cluster_averages, database_avg

# Plotting the profile chart in plotly

def plot_all_clusters_profile_plotly(variables, cluster_averages, database_avg):
    """
    Interactive profile plot using Plotly with solid circular markers.
    Database average is shown as a slightly larger dot.

    Parameters:
    - variables: list of variable names
    - cluster_averages: DataFrame (rows = clusters, columns = variables)
    - database_avg: list of overall averages for each variable
    """
    # Create database average DataFrame
    df_database = pd.DataFrame({
        'Variable': variables,
        'Value': database_avg,
        'Cluster': 'Database Average',
        'Size': [12] * len(variables)  # Larger dots for DB average
    })

    # Melt cluster averages
    df_clusters = cluster_averages.reset_index().melt(
        id_vars=cluster_averages.index.name or 'cluster',
        var_name='Variable',
        value_name='Value'
    )
    df_clusters.rename(columns={cluster_averages.index.name or 'cluster': 'Cluster'}, inplace=True)
    df_clusters['Size'] = 8  # Smaller dots for clusters

    # Combine
    df_plot = pd.concat([df_clusters, df_database], ignore_index=True)

    # Create scatter plot with clean filled circle dots
    fig = px.scatter(
        df_plot,
        x='Value',
        y='Variable',
        color='Cluster',
        size='Size',
        size_max=12,
        hover_data={'Value': ':.2f', 'Variable': True, 'Cluster': True, 'Size': False},
        title='Cluster Profiles vs. Database Average',
        height=40 * len(variables) + 100,
    )

    fig.update_traces(marker=dict(symbol='circle'))

    fig.update_layout(
        yaxis=dict(autorange='reversed'),
        plot_bgcolor='#f9f9f9',
        legend_title='Cluster',
        xaxis_title='Mean / Normalized Value',
        margin=dict(l=60, r=40, t=60, b=40)
    )

    fig.show()

def map_visualization(customer_info):
    fig = px.scatter_mapbox(
    customer_info,
    lat='latitude',
    lon='longitude',
    zoom=3,
    height=500
    )

    # Use a free open-street basemap
    fig.update_layout(mapbox_style="open-street-map")

    # Remove extra margins
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    # Show in notebook
    fig.show()