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
import plotly.subplots as sp
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import plotly.express as px
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import ast
from pyECLAT import ECLAT

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


#################################################################
######################### Assosiation Rules #####################
#################################################################

def transform_dataset(data:pd.DataFrame, data_clusters:pd.DataFrame, num_cluster: int) -> pd.DataFrame:
    '''
    Transforms the dataset where each column is an item from the cutomer basket that will be used for the Apriori algorithm.
    Parameters: 
    - data: customer basket dataset
    - data_clusters: DataFrame with customer segmentation in clusters
    - num_cluster: number of the cluster to be transformed
    Returns:
     - df_items: DataFrame with true or false for each item in the transaction ready for the Apriori algorithm   
    '''

    data.set_index('customer_id', inplace=True)
    items = data.sort_values(by='customer_id')

    clusters = data_clusters[['customer_id', 'cluster']]
    items_clusters = data.merge(clusters, on='customer_id', how='inner')

    cluster = items_clusters[items_clusters['cluster'] == num_cluster]

    items_list = cluster.list_of_goods.to_list()

    items_list = [ast.literal_eval(item) for item in items_list]

    te = TransactionEncoder()
    te_ary = te.fit(items_list).transform(items_list)
    df_items = pd.DataFrame(te_ary, columns=te.columns_)

    return df_items, items_list

def apriori_algorithm(data: pd.DataFrame, min_support: float = 0.2, metric: str = 'confidence', confidence_threshold: float= 0.6) -> pd.DataFrame:
    """
    Apply the Apriori algorithm to find frequent itemsets and association rules.

    Parameters:
    - data: DataFrame where each column is an item and each row is a transaction with boolean values (True/False)
    - min_support: Minimum support for frequent itemsets
    - metric: Metric to use for association rules 'confidence' or 'lift'
    - confidence_threshold: Minimum confidence threshold for assosiation rules

    Returns:
    - DataFrame with association rules
    """
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=confidence_threshold)

    return frequent_itemsets, rules

def eclat_algorithm(lt: list, min_combination: int, max_combination: int, min_support: float = 0.2) -> pd.DataFrame:
    """
    Apply the Eclat algorithm to find support of items.

    Parameters:
    - lt: List of transactions, where each transaction is a list of items
    - min_combintion: Minimum number of items in a combination
    - max_combination: Maximum number of items in a combination
    - min_support: Minimum support for frequent itemsets

    Returns:
    - DataFrame with frequent itemsets nd their support
    """
    eclat = ECLAT(data=pd.DataFrame(lt))

    items_rules_indexes, items_rules_supports = eclat.fit(min_support=min_support,min_combination=min_combination, max_combination=max_combination)

    rules_eclat_groceries = pd.DataFrame(
    list(items_rules_supports.values()),
    index=list(items_rules_supports.keys()),
    columns=['support']
    )

    rules_eclat_groceries.sort_values(by='support', ascending=False, inplace=True)

    return rules_eclat_groceries

#################################################################
######################### Radar Chart ###########################
#################################################################

#This radar (spider) chart provides a visual summary of how different clusters compare across multiple numerical variables. Each axis represents a feature,
# and each cluster is plotted as a closed line connecting the average value of that cluster on each feature.

def radar_chart_by_cluster(df, cluster_col='cluster', title='Radar Chart by Cluster', exclude=None):
    """
    Plots an interactive radar chart using Plotly, showing mean values per cluster.

    Args:
        df (pd.DataFrame): Input DataFrame with numeric columns and a cluster column.
        cluster_col (str): Column indicating cluster membership.
        title (str): Title for the radar chart.
        exclude (list of str, optional): Columns to exclude from the plot.
    """
    if exclude is None:
        exclude = []

    # Step 1: Select numeric columns excluding the cluster column and any excluded columns
    numeric_cols = df.select_dtypes(include=np.number).columns
    numeric_cols = [col for col in numeric_cols if col != cluster_col and col not in exclude]

    if not numeric_cols:
        raise ValueError("No numeric columns remaining after exclusion.")

    # Step 2: Compute cluster means
    cluster_means = df.groupby(cluster_col)[numeric_cols].mean()

    # Step 3: Prepare plotly figure
    fig = go.Figure()

    for cluster in cluster_means.index:
        values = cluster_means.loc[cluster].tolist()
        values += values[:1]  # close the loop
        categories = numeric_cols + [numeric_cols[0]]  # close the loop

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            mode='lines+markers',
            name=f'Cluster {cluster}',
            hovertemplate='%{theta}: %{r:.2f}<extra>Cluster ' + str(cluster) + '</extra>'
        ))

    fig.update_layout(
        title=title,
        width=1200,
        height=800,
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        showlegend=True
    )

    fig.show()


def plot_cluster_boxplots(df, cluster_col, variables, ncols=2, title='Clustered Boxplots'):
    """
    Creates interactive boxplots of multiple variables by cluster using Plotly.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cluster_col (str): Column name indicating cluster/group.
        variables (list of str): List of numeric variables to plot.
        ncols (int): Number of columns in subplot grid.
        title (str): Main title of the plot.
    """
    n_vars = len(variables)
    nrows = -(-n_vars // ncols)  # ceiling division

    fig = sp.make_subplots(rows=nrows, cols=ncols, subplot_titles=variables)

    for i, var in enumerate(variables):
        row = i // ncols + 1
        col = i % ncols + 1
        for cluster in sorted(df[cluster_col].unique()):
            cluster_data = df[df[cluster_col] == cluster][var]
            fig.add_trace(
                go.Box(
                    y=cluster_data,
                    name=f'{cluster}',
                    boxpoints='outliers',
                    marker_color=None,
                    showlegend=(i == 0),
                    legendgroup=f'Cluster {cluster}',
                    hoverinfo='y+name',
                ),
                row=row, col=col
            )
        fig.update_xaxes(title_text='Cluster', row=row, col=col)
        fig.update_yaxes(title_text=var, row=row, col=col)

    fig.update_layout(
        height=300 * nrows,
        width=500 * ncols,
        title_text=title,
        boxmode='group',
        showlegend=True
    )

    fig.show()

