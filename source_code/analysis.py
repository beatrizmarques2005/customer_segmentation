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

#################################################################
######################### Multi Boxplot #########################
#################################################################

#Function to create interactive boxplots of one variable at a time by cluster using Plotly dropdowns.

def plot_cluster_boxplots_dropdown(df, cluster_col, exclude=None, title='Clustered Boxplots'):
    """
    Creates interactive boxplots of one variable at a time by cluster using Plotly dropdowns.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cluster_col (str): Column name indicating cluster/group.
        variables (list of str): List of numeric variables to plot.
        title (str): Main title of the plot.
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

#Function that creates an interactive line plot comparing mean profiles of clusters across given variables.

def plot_line_comparing_profiles(df, cluster_col, exclude=None, title='Comparing Cluster Profiles (Line Plot)'):
    """
    Creates an interactive line plot comparing mean profiles of clusters across given variables.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        cluster_col (str): Column name indicating cluster/group.
        variables (list of str): List of numeric variables to compare.
        title (str): Title of the plot.
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

#This function creates an interactive grouped bar chart using Plotly that compares the percentage distribution of individuals and total lifetime spend across clusters, 
#and annotates each cluster with the ratio of spend percentage to individual percentage, allowing for easy visual comparison of relative value contribution per cluster.

def plot_cluster_bars_percent(df, cluster_col='cluster', spend_cols=None):
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
