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

def compute_average_multi(df: pd.DataFrame, cluster_column: str) -> tuple[list[str], pd.DataFrame, list[float]]:
    """
    Computes the average values of numerical variables for each cluster 
    and for the entire dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset containing customer features and cluster assignments.
    cluster_column : str
        The name of the column in `df` that indicates cluster labels 
        (e.g., 'cluster').

    Returns
    -------
    variables : list of str
        List of feature/variable names used in the averaging (excluding 
        the cluster and customer ID columns).
    cluster_averages : pandas.DataFrame
        A DataFrame where each row corresponds to a cluster and each column 
        contains the mean value of a feature within that cluster.
    database_avg : list of float
        A list of mean values for each variable across the entire dataset 
        (not grouped by cluster).
    """

    # Exclude the cluster column to get only variable columns
    variable_cols = df.columns.difference([cluster_column, 'customer_id'])

    # Compute cluster averages
    cluster_averages = df.groupby(cluster_column)[variable_cols].mean()

    # Compute database average
    database_avg = df[variable_cols].mean().values.tolist()

    return variable_cols.tolist(), cluster_averages, database_avg

# Plotting the profile chart in plotly

def plot_all_clusters_profile(variables: list[str], cluster_averages: pd.DataFrame, database_avg: list[float]) -> None:
    """
    Generates an interactive profile plot using Plotly to visualize the average 
    values of key variables across clusters, compared against the overall 
    database average.

    Parameters
    ----------
    variables : list of str
        List of variable names corresponding to the features used in clustering.
    cluster_averages : pandas.DataFrame
        A DataFrame containing average values for each variable across clusters. 
        Each row represents a cluster, and each column corresponds to a variable.
    database_avg : list of float
        A list containing the overall average value for each variable across 
        the entire dataset.

    Returns
    -------
    None
        Displays an interactive Plotly scatter plot comparing cluster profiles 
        to the database average. No object is returned.
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
    """
    Transforms the customer basket data for a specific cluster into a format 
    suitable for the Apriori algorithm.

    Parameters
    ----------
    data : pandas.DataFrame
        The original customer basket dataset, which must include a 
        'customer_id' column and a 'list_of_goods' column containing items 
        purchased as stringified Python lists.
    data_clusters : pandas.DataFrame
        A DataFrame that maps 'customer_id' to assigned cluster labels, 
        typically from a segmentation model.
    num_cluster : int
        The cluster number to filter transactions by.

    Returns
    -------
    df_items : pandas.DataFrame
        A one-hot encoded DataFrame where each column represents an item and 
        each row a transaction (True/False), suitable for Apriori.
    items_list : list of list of str
        The original list of goods per customer in the specified cluster, 
        parsed into Python lists.
    """

    data = data.copy()

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

    return df_items

def apriori_algorithm(data: pd.DataFrame, min_support: float = 0.2, metric: str = 'confidence', met_threshold: float= 0.6) -> pd.DataFrame:
    """
    Applies the Apriori algorithm to identify frequent itemsets and generate 
    association rules from transaction data.

    Parameters
    ----------
    data : pandas.DataFrame
        A one-hot encoded DataFrame where each column represents an item and 
        each row is a transaction with boolean values (True/False).
    min_support : float, optional (default=0.2)
        The minimum support threshold to consider an itemset as frequent.
    metric : str, optional (default='confidence')
        The metric to evaluate the strength of the association rules. Common 
        choices are 'confidence' or 'lift'.
    confidence_threshold : float, optional (default=0.6)
        The minimum threshold for the selected metric when generating rules.

    Returns
    -------
    frequent_itemsets : pandas.DataFrame
        A DataFrame containing the frequent itemsets that meet the support threshold.
    rules : pandas.DataFrame
        A DataFrame of generated association rules with metrics such as support, 
        confidence, and lift.
    """
    frequent_itemsets = apriori(data, min_support=min_support, use_colnames=True)

    rules = association_rules(frequent_itemsets, metric=metric, min_threshold=met_threshold)

    rules = rules.sort_values('lift', ascending=False)

    return rules

def total_appriori_algorithm(data: pd.DataFrame, data_clusters:pd.DataFrame, num_clusters: int, min_support: float = 0.2, metric: str = 'confidence', met_threshold: float= 0.6) -> None:

    for i in range(num_clusters):
        transform_data = transform_dataset(data, data_clusters, i)
        rules = apriori_algorithm(transform_data, min_support=min_support, metric=metric, met_threshold=met_threshold)
        rules = rules.sort_values('lift', ascending=False)
        print(f"Cluster {i}")
        display(rules)






def eclat_algorithm(lt: list, min_combination: int, max_combination: int, min_support: float = 0.2) -> pd.DataFrame:
    """
    Applies the Eclat algorithm to identify frequent itemsets based on their support.

    Parameters
    ----------
    lt : list of list of str
        A list of transactions, where each transaction is a list of items.
    min_combination : int
        The minimum number of items in a frequent itemset.
    max_combination : int
        The maximum number of items in a frequent itemset.
    min_support : float, optional (default=0.2)
        The minimum support threshold required for an itemset to be considered frequent.

    Returns
    -------
    rules_eclat_groceries : pandas.DataFrame
        A DataFrame containing the frequent itemsets as index and their 
        corresponding support values, sorted in descending order.
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

def radar_chart_by_cluster(df: pd.DataFrame, cluster_col: str = 'cluster', title: str = 'Radar Chart by Cluster', exclude: list[str] = None) -> None:
    """
    Generates an interactive radar (spider) chart to compare the mean profiles of different clusters 
    across multiple numeric features.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame containing numeric variables and a column indicating cluster assignments.

    cluster_col : str, optional (default='cluster')
        Name of the column that defines the cluster or group label for each observation.

    title : str, optional (default='Radar Chart by Cluster')
        Title for the radar chart.

    exclude : list of str, optional (default=None)
        List of column names to exclude from the analysis. Commonly used to omit identifiers, 
        target variables, or any columns that should not be plotted.

    Returns:
    -------
    None
        Displays an interactive Plotly radar chart. Does not return a value.
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

#################################################################
######################### Multi Boxplot #########################
#################################################################

def plot_cluster_boxplots(df: pd.DataFrame, cluster_col: str, exclude: list[str] = None, title: str = 'Clustered Boxplots') -> None:
    """
    Generates interactive boxplots for numeric variables across clusters, allowing users 
    to switch between variables using a dropdown menu.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame containing the clustering information and the numeric variables to plot.

    cluster_col : str
        Name of the column representing the cluster or group label.

    exclude : list of str, optional (default=None)
        List of column names to exclude from plotting. Useful for ignoring identifier or 
        target columns. If None, only the cluster column is excluded.

    title : str, optional (default='Clustered Boxplots')
        Title for the overall plot. The selected variable name will be appended dynamically.

    Returns:
    -------
    None
        Displays an interactive Plotly boxplot chart. Does not return a value.
    """

    if exclude is None:
        exclude = []

    variables = df.select_dtypes(include=np.number).columns
    variables = [col for col in variables if col != cluster_col and col not in exclude]

    fig = go.Figure()
    cluster_labels = sorted(df[cluster_col].unique())

    # Create one trace per variable per cluster (but set only first variable visible)
    for var_index, var in enumerate(variables):
        for cluster in cluster_labels:
            cluster_data = df[df[cluster_col] == cluster]
            visible = (var_index == 0)

            fig.add_trace(
                go.Box(
                    y=cluster_data[var],
                    x=[str(cluster)] * len(cluster_data),
                    name=f'Cluster {cluster}',
                    boxpoints='outliers',
                    marker=dict(opacity=0.7),
                    hovertemplate=f'{var}<br>Cluster: {cluster}<br>Value: %{{y}}<extra></extra>',
                    legendgroup=f'Cluster {cluster}',
                    showlegend=True,
                    visible=visible
                )
            )

    # Create dropdown buttons
    buttons = []
    n_clusters = len(cluster_labels)

    for i, var in enumerate(variables):
        visibility = [False] * len(variables) * n_clusters
        for j in range(n_clusters):
            visibility[i * n_clusters + j] = True

        buttons.append(dict(
            label=var,
            method='update',
            args=[
                {'visible': visibility},
                {
                    'yaxis': {'title': var},
                    'title': f'{title} - {var}'
                }
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=buttons,
            direction='down',
            x=0.60,
            y=1.10,
            xanchor='left',
            yanchor='top',
            showactive=True
        )],
        title=f'{title} - {variables[0]}',
        yaxis_title=variables[0],
        boxmode='group',
        height=800,
        width=1200
    )

    fig.show()

#################################################################
############### Profile Comparison Line Version #################
#################################################################

def plot_line_comparing_profiles(df: pd.DataFrame, cluster_col: str, exclude: list[str] = None, title: str = 'Comparing Cluster Profiles (Line Plot)') -> None:
    """
    Generates an interactive line plot to compare the average profiles of different clusters 
    across a set of numeric variables.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing both the cluster assignments and the variables to compare.

    cluster_col : str
        The name of the column indicating the cluster label or group membership for each observation.

    exclude : list of str, optional (default=None)
        List of column names to exclude from the analysis (e.g., identifiers, target variables, etc.).
        If None, only the cluster column will be excluded.

    title : str, optional (default='Comparing Cluster Profiles (Line Plot)')
        Title for the resulting plot.

    Returns:
    -------
    None
        Displays an interactive Plotly line chart. Does not return a value.
    """
    if exclude is None:
        exclude = []

    variables = df.select_dtypes(include=np.number).columns
    variables = [col for col in variables if col != cluster_col and col not in exclude]

    # Compute mean profiles
    cluster_means = df.groupby(cluster_col)[variables].mean()

    fig = go.Figure()

    # Loop over clusters
    for cluster in cluster_means.index:
        values = cluster_means.loc[cluster]
        fig.add_trace(go.Scatter(
            x=variables,
            y=values,
            mode='lines+markers',
            name=f'Cluster {cluster}',
            hovertemplate='<b>%{x}</b><br>Value: %{y:.2f}<extra>Cluster ' + str(cluster) + '</extra>',
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Variables',
        yaxis_title='Mean Value',
        height=800,
        width=1200,
        hovermode='x unified',
        legend_title='Clusters'
    )

    fig.show()

#################################################################
######### Ratio of Total Individuals with Total Spent ###########
#################################################################

def plot_cluster_bars_percent(df: pd.DataFrame, cluster_col: str = 'cluster', spend_cols: list[str] = None) -> None:
    """
    Generates an interactive grouped bar chart comparing the percentage distribution 
    of individuals and total lifetime spend across clusters.

    Parameters:
    ----------
    df : pandas.DataFrame
        Input DataFrame containing individual-level data, including clustering labels 
        and spend-related columns.

    cluster_col : str, optional (default='cluster')
        Name of the column in `df` that identifies the cluster label for each individual.

    spend_cols : list of str, optional (default=None)
        List of column names representing spend metrics to be summed per individual. 
        If None, all numeric columns except the cluster column will be used.

    Returns:
    -------
    None
        Displays an interactive Plotly bar chart. No object is returned.
    """
    if spend_cols is None:
        spend_cols = [col for col in df.columns if col != cluster_col and df[col].dtype in ['float64', 'int64']]

    # Compute total spend per individual
    df = df.copy()
    df['total_spend'] = df[spend_cols].sum(axis=1)

    # Group by cluster
    grouped = df.groupby(cluster_col)['total_spend'].agg(['count', 'sum']).reset_index()
    grouped.columns = [cluster_col, 'individuals', 'total_spend']

    # Normalize to percent
    grouped['individuals_pct'] = grouped['individuals'] / grouped['individuals'].sum() * 100
    grouped['spend_pct'] = grouped['total_spend'] / grouped['total_spend'].sum() * 100
    grouped['ratio'] = grouped['spend_pct'] / grouped['individuals_pct']

    x_vals = grouped[cluster_col].astype(str)

    # Create bar traces
    trace_individuals = go.Bar(
        x=x_vals,
        y=grouped['individuals_pct'],
        name='% Individuals',
        marker_color='steelblue'
    )
    trace_spend = go.Bar(
        x=x_vals,
        y=grouped['spend_pct'],
        name='% Total Spend',
        marker_color='orange'
    )

    # Ratio annotations
    annotations = []
    for i, row in grouped.iterrows():
        y_pos = max(row['individuals_pct'], row['spend_pct']) + 2
        annotations.append(dict(
            x=str(row[cluster_col]),
            y=y_pos,
            text=f"Ratio: {row['ratio']:.2f}",
            showarrow=False,
            font=dict(size=11)
        ))

    layout = go.Layout(
        title='Cluster Comparison (% of Total Individuals and Spend)',
        xaxis=dict(title='Cluster'),
        yaxis=dict(title='Percentage'),
        barmode='group',
        annotations=annotations
    )

    fig = go.Figure(data=[trace_individuals, trace_spend], layout=layout)
    fig.show()
