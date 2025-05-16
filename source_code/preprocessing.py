import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import MeanShift, DBSCAN, KMeans
from minisom import MiniSom
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#######################################
######### GENERAL EXPLORATION #########
#######################################

def initial_exploration(data: pd.DataFrame) -> None:
    """
    Perform an initial exploration of the given dataset.
    This function provides a quick overview of the dataset by displaying:
    - The first few rows of the dataset.
    - The shape (number of rows and columns) of the dataset.
    - The data types of each column.
    - Basic statistical summary of the dataset (transposed for better readability).
    - The count of missing values in each column.
    Parameters:
    ----------
    data : pandas.DataFrame
        The dataset to be explored, provided as a pandas DataFrame.
    Returns:
    -------
    None
        This function does not return any value. It prints the exploration results to the console.
    """
    
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    display(data.head())
    
    # Display the shape of the dataset
    print("\nShape of the dataset:")
    print(data.shape)
    
    # Display the data types of each column
    print("\ndata types of each column:")
    print(data.dtypes)
    
    # Display basic statistics of the dataset
    print("\nBasic statistics of the dataset:")
    display(data.describe().T)
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(data.isnull().sum())


#######################################
######### GENERAL CORRECTIONS #########
#######################################

def general_customer_info_corrections(customer_info: pd.DataFrame, customer_basket: pd.DataFrame) -> pd.DataFrame:
    
    """
    Perform general corrections and transformations on customer information data.
    This function processes a DataFrame containing customer information by:
    1. Splitting the `customer_name` column into two parts: `education_level` and `customer_name`.
       - The `education_level` is extracted from the prefix of the `customer_name` (before the first period).
       - The `customer_name` is updated to exclude the prefix and is stripped of leading/trailing whitespace.
    2. Converting the `customer_birthdate` column to a datetime format and calculating the customer's age in years.
       - The `age` is computed based on the difference between a fixed reference date (2023-06-09) and the birthdate.
    3. Calculating the number of unique invoices and distinct products bought for each customer.
        - Customers without any invoices or products bought in the `customer_basket` DataFrame will be set to NaN, as customer_basket is just a sample dataset.
    4. Removing the `gender` column from the DataFrame.

    Parameters:
    -----------
    customer_info : pandas.DataFrame
        A DataFrame containing customer information with at least the following columns:
        - 'customer_name': str, the name of the customer (may include education level as a prefix).
        - 'customer_birthdate': str or datetime, the birthdate of the customer.
    customer_basket : pandas.DataFrame
        A DataFrame containing customer transaction data.

    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with the following changes:
        - A new column `education_level` containing the extracted education level (if available).
        - The `customer_name` column updated to exclude the education level prefix.
        - The `customer_birthdate` column converted to datetime format.
        - A new column `age` containing the customer's age in years (calculated as an integer).
        - The `gender` column removed.
    Notes:
    ------
    - If the `customer_name` does not contain a period, the entire name is retained, and `education_level` is set to NaN.
    - Invalid or missing birthdates are coerced to NaT, and the corresponding `age` values are set to NaN.
    """

    # 1
    split_names = customer_info['customer_name'].str.split('.', n=1, expand=True)

    customer_info['education_level'] = split_names[0].where(split_names[1].notna(), np.nan)
    customer_info['customer_name'] = split_names[1].fillna(split_names[0]).str.strip()

    # 2
    customer_info['customer_birthdate'] = pd.to_datetime(customer_info['customer_birthdate'], errors='coerce') # object --> datetime64[ns]
    customer_info['age'] = (pd.Timestamp('2023-06-09 23:59:59') - customer_info['customer_birthdate']).dt.days // 365.25
    customer_info['birth_month'] = customer_info['customer_birthdate'].dt.month
    customer_info['birth_day'] = customer_info['customer_birthdate'].dt.day
    customer_info['birth_year'] = customer_info['customer_birthdate'].dt.year

    # 3
    summary_df = customer_basket.groupby('customer_id').agg(
        distinct_invoice_sum=('invoice_id', 'nunique'),
        distinct_products_sum=('list_of_goods', lambda x: len(set(item for sublist in x for item in eval(sublist))))
    ).reset_index()
    
    summary_df.loc[summary_df['distinct_invoice_sum'] == 0, ['distinct_invoice_sum', 'distinct_products_sum']] = np.nan

    customer_info = customer_info.merge(
        summary_df[['customer_id', 'distinct_invoice_sum', 'distinct_products_sum']],
        on='customer_id',
        how='left'
    )

    # 4
    if 'customer_gender' in customer_info.columns:
        customer_info.drop(columns=['customer_gender'], inplace=True)

    return customer_info


def general_customer_basket_corrections(customer_basket: pd.DataFrame) -> tuple:
    """
    Processes and corrects a customer basket DataFrame by performing various transformations 
    and generating a summary of items.
    Args:
        customer_basket (pd.DataFrame): A DataFrame containing customer basket data. 
            Expected columns:
                - 'list_of_goods': A column containing string representations of lists of items.
                - 'invoice_id': A column containing invoice identifiers.
                - 'customer_id': A column containing customer identifiers.
    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The updated customer basket DataFrame with the following modifications:
                - A new column 'items_count' indicating the number of items in each basket.
                - The 'list_of_goods' column converted to actual lists if it was in string format.
            - pd.DataFrame: A summary DataFrame with the following columns:
                - 'list_of_goods': Unique items from the exploded 'list_of_goods' column.
                - 'invoice_count': The number of unique invoices each item appears in.
                - 'customer_count': The number of unique customers associated with each item.
    """

    customer_basket['items_count'] = customer_basket['list_of_goods'].apply(len)

    customer_basket['list_of_goods'] = customer_basket['list_of_goods'].apply(eval)

    customer_basket_exploded = customer_basket.explode('list_of_goods')

    items_summary = customer_basket_exploded.groupby('list_of_goods').agg(
        invoice_count=('invoice_id', 'nunique'),
        customer_count=('customer_id', 'nunique')
    ).reset_index()

    return customer_basket, items_summary


#######################################
############## DUPLICATES #############
#######################################

def check_duplicates(data: pd.DataFrame) -> str:
    """
    Checks for duplicate customer IDs in a given dataset.

    Args:
        data (pandas.DataFrame): The dataset to check for duplicate customer IDs.

    Returns:
        str: A message indicating the number of duplicate customer IDs in the dataset.
    """
    duplicate_count = data['customer_id'].duplicated().sum()
    return f"Number of duplicate customer IDs: {duplicate_count}"

def treat_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with duplicate customer IDs, keeping the first occurrence.

    Args:
        data (pandas.DataFrame): The dataset to process.

    Returns:
        pandas.DataFrame: The dataset with duplicate customer IDs removed.
    """

    if int(check_duplicates(data).split(' ')[-1]) == 0:
        print("No duplicates found.")

    else:
        data = data.drop_duplicates(subset='customer_id', keep='first')
        print("Duplicates have been removed.")

    return data


#######################################
############### OUTLIERS ##############
#######################################

## One Dimensional Outliers

def check_outliers_numerical_boxplot(customer_info: pd.DataFrame) -> None:
    """
    Generates interactive box plots for numerical columns in the customer DataFrame to visualize outliers.
    Certain columns are excluded as they exhibit a categorical behavior.
    Args:
        customer_info (pd.DataFrame): A pandas DataFrame containing customer information 
                                      with numerical columns to analyze.
    Excluded Columns:
        - 'customer_id'
        - 'kids_home'
        - 'teens_home'
        - 'number_complaints'
        - 'year_first_transaction'
        - 'distinct_stores_visited'
        - 'typical_hour'

    Returns:
        None: The function directly displays the Plotly figure in the output.

    """

    numerical_columns = list(customer_info.select_dtypes(include=['number']))
    columns_to_remove=['customer_id', 'kids_home', 'teens_home', 'number_complaints', 'year_first_transaction', 'distinct_stores_visited', 'typical_hour']
    
    numerical_columns = [col for col in numerical_columns if col not in columns_to_remove]
    
    fig = go.Figure()

    buttons = []

    for idx, column in enumerate(numerical_columns):
        fig.add_trace(go.Box(
            x=customer_info[column],
            name=f'<i>{column}</i>',
            marker_color='darkgreen',
            visible=(idx == 0)
        ))

        buttons.append(
            {'method': 'update',
             'label': column,
             'args': [{'visible': [i == idx for i in range(len(numerical_columns))]},
                      {'title': f'Box Plot for <i>{column}</i>', 'showlegend': True}]}
        )

    fig.update_layout(
        updatemenus=[{'type': 'dropdown', 'active': 0, 'buttons': buttons, 'x': 1, 'y': 1.15}],
        title=f'Box Plot for <i>{numerical_columns[0]}</i>',
        showlegend=True,
        yaxis=dict(showticklabels=False)
    )

    fig.show()

def check_outliers_numerical_histogram(customer_info: pd.DataFrame) -> None:
    """
    Generates interactive histograms for numerical columns in a DataFrame to check for outliers.
    Certain columns are excluded as they exhibit a categorical behavior.
    Args:
        customer_info (pd.DataFrame): A pandas DataFrame containing customer information, including numerical columns.
    Excluded Columns:
        - 'customer_id'
        - 'kids_home'
        - 'teens_home'
        - 'number_complaints'
        - 'year_first_transaction'
        - 'distinct_stores_visited'
        - 'typical_hour'
    Returns:
        None: The function displays the interactive Plotly figure and does not return any value.

    """
    numerical_columns = list(customer_info.select_dtypes(include=['number']))
    columns_to_remove = ['customer_id', 'kids_home', 'teens_home', 'number_complaints', 'year_first_transaction', 'distinct_stores_visited', 'typical_hour']
    
    numerical_columns = [col for col in numerical_columns if col not in columns_to_remove]
    
    fig = go.Figure()
    buttons = []
    
    for idx, column in enumerate(numerical_columns):
        fig.add_trace(go.Histogram(
            x=customer_info[column],
            name=f'<i>{column}</i>',
            marker_color='darkgreen',
            visible=(idx == 0),
            nbinsx=30
        ))
        
        buttons.append(
            {'method': 'update',
             'label': column,
             'args': [{'visible': [i == idx for i in range(len(numerical_columns))]},
                      {'title': f'Histogram for <i>{column}</i>', 'showlegend': True}]}
        )
    
    fig.update_layout(
        updatemenus=[{'type': 'dropdown', 'active': 0, 'buttons': buttons, 'x': 1, 'y': 1.15}],
        title=f'Histogram for <i>{numerical_columns[0]}</i>',
        showlegend=True,
        xaxis_title='Value',
        yaxis_title='Count'
    )
    
    fig.show()

def check_outliers_categorical(customer_info: pd.DataFrame) -> None:
    """
    Generates interactive bar plots for categorical columns in a DataFrame to help identify outliers.
    It includes both explicitly categorical columns and numerical columns that exhibit categorical behavior.
    Args:
        customer_info (pd.DataFrame): A pandas DataFrame containing customer information. The DataFrame
                                      should include the specified categorical columns and numerical
                                      columns with categorical behavior.
    Categorical Columns:
        - 'education_level'
    Numerical Columns with Categorical Behavior:
        - 'kids_home'
        - 'teens_home'
        - 'number_complaints'
        - 'year_first_transaction'
        - 'distinct_stores_visited'
        - 'typical_hour'
    Returns:
        None: The function displays the interactive Plotly figure and does not return any value.

    """
    categorical_columns = ['education_level']
    numerical_with_categorical_behaviour= ['kids_home', 'teens_home', 'number_complaints', 'year_first_transaction', 'distinct_stores_visited', 'typical_hour']
    categorical_columns.extend(numerical_with_categorical_behaviour)

    fig = go.Figure()

    buttons = []

    for idx, column in enumerate(categorical_columns):
        value_counts = customer_info[column].value_counts()

        fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            name=f'<i>{column}</i>',
            marker_color='green',
            visible=(idx == 0)
        ))

        buttons.append(
            {'method': 'update',
             'label': column,
             'args': [{'visible': [i == idx for i in range(len(categorical_columns))]},
                      {'title': f'Bar Plot for <i>{column}</i>', 'showlegend': True}]}
        )

    fig.update_layout(
        updatemenus=[{'type': 'dropdown', 'active': 0, 'buttons': buttons, 'x': 1, 'y': 1.15}],
        title=f'Bar Plot for <i>{categorical_columns[0]}</i>',
        showlegend=True
    )

    fig.show()

## Multi Dimensional Outliers --> DBSCAN

def check_multidimensional_outliers_dbscan(customer_info: pd.DataFrame, min_samples: int, eps: float) -> None:

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    customer_info['cluster_dbscan'] = dbscan.fit_predict(customer_info)

    customer_info['is_outlier_dbscan'] = customer_info['cluster_dbscan'] == -1

    return customer_info

def defining_params_dbscan_outliers(customer_info: pd.DataFrame, min_samples: int, eps_range: tuple) -> None:

    for eps in np.arange(*eps_range):
        
        print('=====================================')
        ci_dbscan = check_multidimensional_outliers_dbscan(customer_info, min_samples, eps)
        cluster_comparison = ci_dbscan.groupby(['cluster_dbscan', 'is_outlier_dbscan']).size().reset_index(name='number_of_customers')

        num_clusters = cluster_comparison[cluster_comparison['is_outlier_dbscan'] == 0]['cluster_dbscan'].nunique()
        total_outliers = cluster_comparison[cluster_comparison["is_outlier_dbscan"] == 1]["number_of_customers"].sum()

        print(f"With eps = {eps}\n\tNumber of clusters (not outliers): {num_clusters}\n\tTotal number of outliers: {total_outliers}")

def treat_multidimensional_outliers_dbscan(customer_info: pd.DataFrame, min_samples: int, eps: float) -> pd.DataFrame:

    customer_info = check_multidimensional_outliers_dbscan(customer_info, min_samples, eps)
    customer_info = customer_info[customer_info['is_outlier_dbscan'] == False]
    customer_info.drop(columns=['cluster_dbscan', 'is_outlier_dbscan'], inplace=True)

    return customer_info

# SOM
def check_multidimensional_outliers_som(data_np: np.ndarray, 
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
    #som.random_weights_init(data)
    #som.train_random(data, num_iteration=number_of_iterations)
    som.train_batch(data_np, number_of_iterations)
    return som

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

    plt.figure(figsize=(15, 15))
    for i, f in enumerate(feature_names):
        plt.subplot(5, 5, i + 1)
        plt.title(f)
        plt.pcolor(W[:, :, i].T, cmap='coolwarm')
        plt.xticks(np.arange(10 + 1))
        plt.yticks(np.arange(10 + 1))
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


#######################################
############ MISSING VALUES ###########
#######################################

def impute_loyalty_card(customer_info:pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the 'loyalty_card_number' column of the customer_info DataFrame with 0.

    Parameters:
    customer_info (pd.DataFrame): A pandas DataFrame containing customer information, 
                                  including a 'loyalty_card_number' column.

    Returns:
    pd.DataFrame: The updated DataFrame with missing values in 'loyalty_card_number' replaced by 0.
    """
    customer_info['loyalty_card_number'].fillna(0, inplace=True)

    return customer_info

def impute_kids_teens_home(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    This function imputes missing values in the 'kids_home' and 'teens_home' columns based on the following logic:
    1. If 'kids_home' is missing (NaN) and 'teens_home' is not missing and greater than 0, then 'kids_home' is set to 0.
    2. If 'teens_home' is missing (NaN) and 'kids_home' is not missing and greater than 0, then 'teens_home' is set to 0.
    3. If both 'kids_home' and 'teens_home' are missing (NaN), both are set to 0.

    Args:
        customer_info (pd.DataFrame): A pandas DataFrame containing customer information,
                                      including the 'kids_home' and 'teens_home' columns.

    Returns:
        pd.DataFrame: The updated DataFrame with imputed values in the 'kids_home' and 'teens_home' columns.
    """
    # 1
    customer_info['kids_home'] = np.where(
        (customer_info['kids_home'].isna()) & 
        ((~customer_info['teens_home'].isna()) & (customer_info['teens_home'] > 0)), 
        0, 
        customer_info['kids_home']
    )
    
    # 2
    customer_info['teens_home'] = np.where(
        (customer_info['teens_home'].isna()) & 
        ((~customer_info['kids_home'].isna()) & (customer_info['kids_home'] > 0)), 
        0, 
        customer_info['teens_home']
    )
    
    # 3
    mask = customer_info['kids_home'].isna() & customer_info['teens_home'].isna()
    customer_info['kids_home'] = np.where(mask, 0, customer_info['kids_home'])
    customer_info['teens_home'] = np.where(mask, 0, customer_info['teens_home'])
    
    return customer_info

def impute_lifetime_spend_alcohol_drinks(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the 'lifetime_spend_alcohol_drinks' column of the customer_info DataFrame.

    This function replaces all NaN (missing) values in the 'lifetime_spend_alcohol_drinks' column with 0.

    Parameters:
    customer_info (pd.DataFrame): A pandas DataFrame containing customer information, including the 
                                  'lifetime_spend_alcohol_drinks' column.

    Returns:
    None: The function modifies the input DataFrame in place.
    """
    customer_info['lifetime_spend_alcohol_drinks'].replace(np.nan, 0, inplace = True)

    return customer_info

def impute_education_level(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Impute the 'education_level' column based on the 'age' column.
    This function assigns an education level to individuals based on their age,
    following the education system described at:
    https://edu.azores.gov.pt/seccoes/matriculas-escolaridade-obrigatoria/
    The mapping is as follows:
    - Age >= 59: '4th'
    - 44 <= Age <= 58: '6th'
    - 30 <= Age <= 43: '9th'
    - Age <= 29: 'Hs' (High School)
    Parameters:
    -----------
    customer_info : pandas.DataFrame
        A DataFrame containing at least the columns 'age' and 'education_level'.
    Returns:
    --------
    pandas.DataFrame
        The input DataFrame with the 'education_level' column updated based on the age criteria.
    """
    customer_info['education_level'] = np.where((customer_info['age']>= 59) & (~customer_info['age'].isna()), '4th', customer_info['education_level'])
    customer_info['education_level'] = np.where((customer_info['age']>= 44)  & (customer_info['age'] <=58), '6th', customer_info['education_level'])
    customer_info['education_level'] = np.where((customer_info['age']>= 30)  & (customer_info['age'] <=43), '9th', customer_info['education_level'])
    customer_info['education_level'] = np.where(customer_info['age']<= 29, 'Hs', customer_info['education_level'])

    return customer_info

def impute_missing_values(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in the dataset by applying a series of imputation functions.

    Args:
        customer_info (pd.DataFrame): The input DataFrame with potential missing values.

    Returns:
        pd.DataFrame: The DataFrame with missing values imputed.
    """
    customer_info = impute_loyalty_card(customer_info)
    customer_info = impute_kids_teens_home(customer_info)
    customer_info = impute_lifetime_spend_alcohol_drinks(customer_info)
    customer_info = impute_education_level(customer_info)

    return customer_info

def knn_imputing(customer_info: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    This function applies KNN imputation to all columns except 'customer_id', ensuring that
    the 'customer_id' column remains unchanged. The imputed values are based on the 
    similarity of rows in the dataset.

    Parameters:
    -----------
    customer_info : pd.DataFrame
        The input DataFrame containing the data to be imputed. It must include a 'customer_id' column.
    n_neighbors : int, optional (default=5)
        The number of nearest neighbors to use for imputing missing values.

    Returns:
    --------
    pd.DataFrame
        A new DataFrame with missing values imputed. The 'customer_id' column is preserved as is.

    Notes:
    ------
    - This function assumes that the input DataFrame contains numeric data for imputation.
    - The KNN imputation is performed using sklearn's KNNImputer.
    - The 'customer_id' column is excluded from the imputation process and added back after imputation.
    """

    features = customer_info.drop(columns=['customer_id'])

    imputer = KNNImputer(n_neighbors=n_neighbors)

    imputed_features = imputer.fit_transform(features)

    imputed_data = pd.DataFrame(imputed_features, columns=features.columns)

    imputed_data['customer_id'] = customer_info['customer_id'].values

    return imputed_data

# Note: we can not impute either in loyalty card number or customer birthdate.

#######################################
############## ENCODING ###############
#######################################

def customer_info_encoding(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes and preprocesses customer information by performing the following transformations:
    1. Drops the 'customer_name' column as it is not required for analysis.
    2. Encodes the 'education_level' column into numerical values representing years of education.
       - Mapping: {'4th': 4, '6th': 6, '9th': 9, 'Hs': 12, 'Bsc': 16, 'Msc': 18, 'Phd': 21}
    3. Encodes the 'customer_gender' column into binary values:
       - Mapping: {'male': 0, 'female': 1}
    4. Drops the original 'customer_gender' and 'education_level' columns after encoding.
    Args:
        customer_info (pd.DataFrame): A pandas DataFrame containing customer information with 
                                      columns 'customer_name', 'education_level', and 'customer_gender'.
    Returns:
        pd.DataFrame: A pandas DataFrame with the processed customer information, including:
                      - 'education_years': Numerical representation of education level.
                      - 'gender': Binary representation of gender.
                      - All other columns from the original DataFrame except the dropped ones.
    """
    # 1
    customer_info.drop(['customer_name', 'customer_birthdate'], axis = 1, inplace = True)

    # 2
    education_mapping = {'4th': 4, '6th': 6, '9th': 9, 'Hs': 12, 'Bsc': 15, 'Msc': 17, 'Phd': 20}
    customer_info['education_years'] = customer_info['education_level'].map(education_mapping)

    # 4
    customer_info.drop(['education_level'], axis = 1, inplace = True)

    return customer_info

#######################################
########### INCONSISTENCIES ###########
#######################################


# age and education
# - 9 <= Age : '4th'            
# - 11 <= Age: '6th'       
# - 14 <= Age: '9th'
# - 17 <= Age: 'Hs' (High School)   
# - 21 <= Age: 'Bsc'    
# - 23 <= Age: 'Msc'
# - 26 <= Age: 'Phd'

# people that live more than 1km from the coast (Latitude should be between -90 and +90 and Longitude should be between -180 and +180)
# "Latitude out of range": (~customer_info['latitude'].between(-90, 90, inclusive='both')).sum(),
# "Longitude out of range": (~customer_info['longitude'].between(-180, 180, inclusive='both')).sum(),

def check_inconsistencies(customer_info: pd.DataFrame) -> (pd.Series, pd.DataFrame):
    """
    Counts the number of occurrences for each inconsistency in customer_info,
    and returns the inconsistent rows in a new DataFrame with a column indicating the inconsistency.

    Args:
        customer_info (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.Series, pd.DataFrame]: 
            - A series where the index is the inconsistency description and the value is the count.
            - A DataFrame containing the rows with inconsistencies and a column 'inconsistency' describing the issue.
    """
    inconsistencies = {}
    inconsistent_rows = pd.DataFrame()

    # Helper to collect inconsistent rows
    def collect(mask, label):
        nonlocal inconsistent_rows
        inconsistencies[label] = mask.sum()
        temp = customer_info[mask].copy()
        temp['inconsistency'] = label
        inconsistent_rows = pd.concat([inconsistent_rows, temp])

    # 1. Negative kids_home
    mask = (customer_info['kids_home'] < 0).fillna(False)
    collect(mask, "Negative kids_home")

    # 2. Negative teens_home
    mask = (customer_info['teens_home'] < 0).fillna(False)
    collect(mask, "Negative teens_home")

    # 3. Negative number_complaints
    mask = (customer_info['number_complaints'] < 0).fillna(False)
    collect(mask, "Negative number_complaints")

    # 4. Negative distinct_stores_visited
    mask = (customer_info['distinct_stores_visited'] < 0).fillna(False)
    collect(mask, "Negative distinct_stores_visited")

    # 5. lifetime_total_distinct_products issues
    mask = (
        (customer_info['lifetime_total_distinct_products'] <= 0).fillna(False) |
        (customer_info['distinct_products_sum'] > customer_info['lifetime_total_distinct_products']).fillna(False)
    )
    collect(mask, "Problem with lifetime_total_distinct_products")

    # 6. Year of first transaction > 2025
    mask = (customer_info['year_first_transaction'] > 2025).fillna(False)
    collect(mask, "Year of first transaction > 2025")

    # 7. Percentage of products bought promotion < 0 or > 1
    mask = (
        (customer_info['percentage_of_products_bought_promotion'] < 0) |
        (customer_info['percentage_of_products_bought_promotion'] > 1)
    ).fillna(False)
    collect(mask, "Percentage of products bought promotion < 0 or > 1")

    # 8. Negative lifetime spend values
    spend_cols = [col for col in customer_info.columns if col.startswith('lifetime_spend_')]
    for col in spend_cols:
        col_mask = (customer_info[col] < 0).fillna(False)
        collect(col_mask, f"Negative value in {col}")

    display(inconsistent_rows)

def correcting_inconsistencies(customer_info: pd.DataFrame) -> pd.DataFrame:
    for index, row in customer_info.iterrows():

        if row['kids_home'] < 0:
            customer_info.loc[index, 'kids_home'] *= -1
        if row['teens_home'] < 0:
            customer_info.loc[index, 'teens_home'] *= -1

        if row['number_complaints'] < 0:
            customer_info.loc[index, 'number_complaints'] = 0

        if row['distinct_stores_visited'] < 0:
            customer_info.loc[index, 'distinct_stores_visited'] *= -1

        if row['year_first_transaction'] > 2025:
            customer_info.loc[index, 'year_first_transaction'] = 2025

        if row['lifetime_total_distinct_products'] < 0:
            customer_info.loc[index, 'lifetime_total_distinct_products'] *= -1
        elif row['lifetime_total_distinct_products'] == 0:
            customer_info.loc[index, 'lifetime_total_distinct_products'] = 1

        if row['distinct_products_sum'] > row['lifetime_total_distinct_products']:
            customer_info.loc[index, 'lifetime_total_distinct_products'] = row['distinct_products_sum']

        if row['percentage_of_products_bought_promotion'] < 0:
            customer_info.loc[index, 'percentage_of_products_bought_promotion'] = 0
        elif row['percentage_of_products_bought_promotion'] > 1:
            customer_info.loc[index, 'percentage_of_products_bought_promotion'] = 1

        for col in customer_info.columns:
            if col.startswith('lifetime_spend_'):
                if row[col] < 0:
                    customer_info.loc[index, col] *= -1

    return customer_info


#######################################
############### SCALING ###############
#######################################

def scaling(data: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the input DataFrame using Min-Max scaling.

    Parameters:
    -----------
    data : pd.DataFrame
        The input DataFrame containing the data to be scaled.
    Returns:
    --------
    pd.DataFrame
        A new DataFrame with the scaled data, where all feature values are
        normalized to the range [0, 1].
    Notes:
    ------
    - This function assumes that the input DataFrame contains only numeric data.
    - The scaling is performed column-wise.
    """
    
    scaler = MinMaxScaler()

    scaled_data = scaler.fit_transform(data)

    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)
 
    return scaled_data


#######################################
############# REDUNDANCY ##############
#######################################

def correlation_matrix(customer_info: pd.DataFrame) -> list:

    """
    Generates an interactive correlation matrix heatmap using Plotly and identifies strongly correlated pairs.

    Args:
        customer_info (pd.DataFrame): The input DataFrame containing numerical data.
        threshold (float): The correlation threshold to highlight strong correlations.

    Returns:
        list: A list of tuples containing pairs of columns with correlation above the threshold.
    """
    corr = customer_info.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.columns,
        colorscale='rdylbu',
        colorbar=dict(title="Correlation"),
        zmin=-1, zmax=1
    ))

    fig.update_layout(
        title="Correlation Matrix",
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
        autosize=True,
        width=1000,
        height=1000
    )

    fig.show()

#######################################
########## FEATURE SELECTION ##########
#######################################


def feature_selection_with_clustering(data: pd.DataFrame, n_clusters: int = 7) -> pd.DataFrame:
    feature_importance = pd.DataFrame(index=data.columns)

    data_np = data.values  # Conversão para numpy array

    # SOM
    som = MiniSom(x=1, y=n_clusters, input_len=data.shape[1], sigma=0.5, learning_rate=0.5)
    som.random_weights_init(data_np)
    som.train_random(data_np, 100)
    som_weights = som.get_weights()[0]
    som_importance = np.std(som_weights, axis=0) > np.mean(np.std(som_weights, axis=0))
    feature_importance['SOM'] = som_importance

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(data.T)
    kmeans_importance = pd.Series(kmeans.labels_).value_counts(normalize=True) < 0.5
    feature_importance['K-Means'] = [kmeans_importance[label] for label in kmeans.labels_]

    # Hierarchical Clustering
    linkage_matrix = linkage(data.T, method='ward')
    hierarchical_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
    hierarchical_importance = pd.Series(hierarchical_labels).value_counts(normalize=True) < 0.5
    feature_importance['Hierarchical'] = [hierarchical_importance[label] for label in hierarchical_labels]

    return feature_importance

def feature_selection(data: pd.DataFrame, n_clusters: int = 7) -> pd.DataFrame:

    feature_importance = feature_selection_with_clustering(data, n_clusters).T
    columns_to_delete = feature_importance.columns[feature_importance.sum(axis=0) == 0]
    feature_importance.drop(columns=columns_to_delete, inplace=True)
    data = data.loc[:, feature_importance.columns]

    return data

def pca(data: pd.DataFrame, num_components: int = None) -> PCA:
    pca = PCA(num_components=25)
    pca.fit(data)
    return pca
