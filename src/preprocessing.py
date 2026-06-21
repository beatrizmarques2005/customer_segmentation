import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN

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
    Cleans and enriches customer information data by performing several preprocessing steps.
    This function applies the following transformations:
    1. Splits the 'customer_name' column to extract the education level (if present) and cleans the name.
    2. Converts the 'customer_birthdate' column to datetime, calculates age as of June 9, 2023, and extracts birth month, day, and year.
    3. Aggregates customer basket data to compute the number of distinct invoices and distinct products per customer, merging these features into the customer info.
    4. Creates a binary 'has_loyalty_card' feature indicating the presence of a loyalty card.
    5. Calculates 'customer_lifetime' as the difference between 2025 and the year of the first transaction, only for customers with a loyalty card.
    6. Drops columns that are no longer needed after processing.
    Parameters:
        customer_info (pd.DataFrame): DataFrame containing customer information, including columns such as 'customer_name', 'customer_birthdate', 'loyalty_card_number', and 'year_first_transaction'.
        customer_basket (pd.DataFrame): DataFrame containing basket information, including 'customer_id', 'invoice_id', and 'list_of_goods'.
    Returns:
        pd.DataFrame: The processed customer information DataFrame with new features and cleaned data.
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
    customer_info['has_loyalty_card'] = customer_info['loyalty_card_number'].notna().astype(int)

    # 5
    customer_info['customer_lifetime'] = (2025 - customer_info['year_first_transaction']).where(customer_info['has_loyalty_card'] == 1,np.nan)

    # 6
    customer_info.drop(['customer_name', 'customer_birthdate', 'loyalty_card_number', 'year_first_transaction'], axis=1, inplace=True)

    return customer_info


def general_customer_basket_corrections(customer_basket: pd.DataFrame) -> tuple:
    """
    Performs general corrections and feature engineering on a customer basket DataFrame.
    This function adds two new columns to the input DataFrame:
        - 'items_count': The total number of items in each customer's basket.
        - 'distinct_items_count': The number of unique items in each customer's basket.
    It also explodes the 'list_of_goods' column to create a summary DataFrame with the count of unique invoices and customers per item.
    Parameters:
        customer_basket (pd.DataFrame): \n            A DataFrame containing at least the columns 'list_of_goods', 'invoice_id', and 'customer_id'.
            The 'list_of_goods' column should contain string representations of lists of items.
    Returns:
        tuple:
            - pd.DataFrame: The original DataFrame with added 'items_count' and 'distinct_items_count' columns.
            - pd.DataFrame: A summary DataFrame with columns:
                - 'list_of_goods': The item name.
                - 'invoice_count': Number of unique invoices containing the item.
                - 'customer_count': Number of unique customers who purchased the item.
    """


    customer_basket['items_count'] = customer_basket['list_of_goods'].apply(len)
    customer_basket['distinct_items_count'] = customer_basket['list_of_goods'].apply(
        lambda x: sum(x.count(item) == 1 for item in x)
    )

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
########### INCONSISTENCIES #..........
#######################################

def check_inconsistencies(customer_info: pd.DataFrame) -> (pd.Series, pd.DataFrame):
    """
    Checks for various data inconsistencies in a customer information DataFrame.
    This function examines the provided DataFrame for a set of predefined data quality issues, such as negative values in certain columns, logical inconsistencies, and out-of-range values. It collects and summarizes the number of inconsistencies found for each type and returns both a summary and the rows where inconsistencies were detected.
    Parameters
    ----------
    customer_info : pd.DataFrame
        DataFrame containing customer information with expected columns such as 'kids_home', 'teens_home', 'number_complaints', \n        'distinct_stores_visited', 'lifetime_total_distinct_products', 'distinct_products_sum', 'customer_lifetime', \n        'percentage_of_products_bought_promotion', 'age', and columns starting with 'lifetime_spend_'.
    Returns
    -------
    inconsistencies : pd.Series
        A summary of the number of inconsistencies found for each type, indexed by inconsistency label.
    inconsistent_rows : pd.DataFrame
        DataFrame containing all rows from the input that were found to be inconsistent, with an additional column 'inconsistency'\n        indicating the type of inconsistency detected.
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

    # 6. customer_lifetime < 0 years
    mask = (customer_info['customer_lifetime'] < 0).fillna(False)
    collect(mask, "customer_lifetime < 0")

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

    # 9. Age VS year of first transaction (age at first transaction should be >= 0)
    mask = (
        (customer_info['age'].notna()) &
        (customer_info['customer_lifetime'].notna()) &
        ((customer_info['age'] - customer_info['customer_lifetime']) < 18)
    )
    collect(mask, "Age at first transaction < 18")

    display(inconsistent_rows)

def correcting_inconsistencies(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects data inconsistencies in the customer_info DataFrame.
    This function iterates through each row of the input DataFrame and applies the following corrections:
    - Converts negative values in 'kids_home', 'teens_home', 'distinct_stores_visited', and 'lifetime_total_distinct_products' to their absolute values.
    - Sets negative values in 'number_complaints' and 'customer_lifetime' to 0.
    - Ensures 'lifetime_total_distinct_products' is at least 1 if it is 0.
    - If 'distinct_products_sum' exceeds 'lifetime_total_distinct_products', updates 'lifetime_total_distinct_products' to match 'distinct_products_sum'.
    - Clamps 'percentage_of_products_bought_promotion' to the range [0, 1].
    - Converts negative values in columns starting with 'lifetime_spend_' to their absolute values.
    - If the difference between 'age' and 'customer_lifetime' is less than 18, sets 'customer_lifetime' to 0.
    Args:
        customer_info (pd.DataFrame): DataFrame containing customer information with potential inconsistencies.
    Returns:
        pd.DataFrame: The corrected DataFrame with inconsistencies resolved.
    """
    for index, row in customer_info.iterrows():

        if row['kids_home'] < 0:
            customer_info.loc[index, 'kids_home'] *= -1
        if row['teens_home'] < 0:
            customer_info.loc[index, 'teens_home'] *= -1

        if row['number_complaints'] < 0:
            customer_info.loc[index, 'number_complaints'] = 0

        if row['distinct_stores_visited'] < 0:
            customer_info.loc[index, 'distinct_stores_visited'] *= -1

        if row['customer_lifetime'] < 0:
            customer_info.loc[index, 'customer_lifetime'] = 0

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

        if (
            pd.notna(row['age']) and
            pd.notna(row['customer_lifetime']) and
            (row['age'] - row['customer_lifetime']) < 18
        ):
            customer_info.loc[index, 'customer_lifetime'] = 0

    return customer_info

#######################################
############### OUTLIERS ##############
#######################################

## One Dimensional Outliers

def check_outliers_numerical_boxplot(customer_info: pd.DataFrame) -> None:
    """
    Display interactive box plots for numerical columns in the customer DataFrame to help visualize potential outliers.

    This function generates a dropdown-enabled Plotly figure, allowing users to select and inspect box plots for each
    numerical feature (excluding those with categorical behavior or identifiers). This aids in identifying outliers and
    understanding the distribution of each variable.

    Parameters
    ----------
    customer_info : pd.DataFrame
        DataFrame containing customer information with numerical columns to analyze.

    Excludes Columns
    ----------------
    - 'customer_id'
    - 'kids_home'
    - 'teens_home'
    - 'number_complaints'
    - 'customer_lifetime'
    - 'distinct_stores_visited'
    - 'typical_hour'
    - 'has_loyalty_card'

    Returns
    -------
    None
        Displays the interactive Plotly box plot figure.
    """

    numerical_columns = list(customer_info.select_dtypes(include=['number']))
    columns_to_remove = [
        'customer_id', 'kids_home', 'teens_home', 'number_complaints',
        'customer_lifetime', 'distinct_stores_visited', 'typical_hour', 'has_loyalty_card'
    ]
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
    Display interactive histograms for numerical columns in the customer DataFrame to help visualize potential outliers.

    This function creates an interactive Plotly figure with a dropdown menu, allowing users to select and view histograms
    for each numerical feature (excluding columns that are identifiers or exhibit categorical behavior). This visualization
    helps in identifying outliers and understanding the distribution of each numerical variable.

    Parameters
    ----------
    customer_info : pd.DataFrame
        DataFrame containing customer information with numerical columns to analyze.

    Notes
    -----
    The following columns are excluded from the analysis as they are either identifiers or considered categorical:
        - 'customer_id'
        - 'kids_home'
        - 'teens_home'
        - 'number_complaints'
        - 'customer_lifetime'
        - 'distinct_stores_visited'
        - 'typical_hour'
        - 'has_loyalty_card'

    Returns
    -------
    None
        Displays the interactive Plotly histogram figure in the output cell.
    """
    numerical_columns = list(customer_info.select_dtypes(include=['number']))
    columns_to_remove = [
        'customer_id', 'kids_home', 'teens_home', 'number_complaints',
        'customer_lifetime', 'distinct_stores_visited', 'typical_hour', 'has_loyalty_card'
    ]
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
    Visualizes the distribution of categorical and categorical-like numerical columns in a DataFrame using interactive Plotly bar plots.
    This function helps identify potential outliers or anomalies in categorical data by generating an interactive bar chart for each specified column. Users can switch between columns using a dropdown menu. Both explicitly categorical columns and numerical columns that represent discrete categories are included.
        customer_info (pd.DataFrame): \n            DataFrame containing customer information. Must include the following columns:
                - Categorical columns: \n                    * 'education_level'
                - Numerical columns with categorical behavior:
                    * 'kids_home'
                    * 'teens_home'
                    * 'number_complaints'
                    * 'customer_lifetime'
                    * 'distinct_stores_visited'
                    * 'typical_hour'
                    * 'has_loyalty_card'
        None: \n            Displays an interactive Plotly figure in the output cell or browser. Does not return any value.
    """
    categorical_columns = ['education_level']
    numerical_with_categorical_behaviour= ['kids_home', 'teens_home', 'number_complaints', 'customer_lifetime', 'distinct_stores_visited', 'typical_hour', 'has_loyalty_card']
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

def treat_outliers(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):

    """
    Identifies and removes outlier rows from the input DataFrame based on predefined upper thresholds for specific columns.
    For each column listed in the internal `thresholds` dictionary, any row with a value exceeding the corresponding threshold is considered an outlier and removed. Missing values (NaN) are not treated as outliers.
    
    Parameters:
        data (pd.DataFrame): The input DataFrame containing the data to be filtered for outliers.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - The first DataFrame contains the rows with outliers removed.
            - The second DataFrame contains only the rows identified as outliers (i.e., those that were removed).
    """
    
    thresholds = {
        'kids_home': 8,
        'teens_home': 3,
        'number_complaints': 3,
        'distinct_stores_visited': 8,
        'lifetime_spend_groceries': 100000,
        'lifetime_spend_electronics': 17000, # ?24000
        'lifetime_spend_vegetables': 2500, # 3000
        'lifetime_spend_nonalcohol_drinks': 1500,
        'lifetime_spend_alcohol_drinks': 2500, # 3000
        'lifetime_spend_meat': 2600,
        'lifetime_spend_fish': 3200,
        'lifetime_spend_hygiene': 2800,
        'lifetime_spend_videogames': 1600, # 1900
        'lifetime_spend_petfood': 900,
        'lifetime_total_distinct_products': 600,
        'customer_lifetime': 25
    }

    mask = pd.Series(True, index=data.index)
    for col, max_val in thresholds.items():
        if col in data.columns:
            mask &= (data[col].isna() | (data[col] <= max_val))
    kept = data[mask].reset_index(drop=True)
    removed = data[~mask].reset_index(drop=True)
    return kept, removed

## Multi Dimensional Outliers --> DBSCAN

def check_multidimensional_outliers_dbscan(customer_info: pd.DataFrame, min_samples: int, eps: float) -> pd.DataFrame:
    """
    Identifies multidimensional outliers in a customer dataset using the DBSCAN clustering algorithm.

    Parameters:
    ----------
    customer_info : pd.DataFrame
        A DataFrame containing customer data. If present, the 'customer_id' column will be excluded \n        from clustering input as it's typically a non-numeric identifier.
    min_samples : int
        The minimum number of samples in a neighborhood for a point to be considered a core point \n        in the DBSCAN algorithm.
    eps : float
        The maximum distance between two samples for them to be considered as in the same neighborhood.

    Returns:
    -------
    pd.DataFrame
        A DataFrame identical to the input with two additional columns:
        - 'cluster_dbscan': the DBSCAN-assigned cluster label for each observation.
                            A value of -1 indicates an outlier.
        - 'is_outlier_dbscan': a boolean column indicating whether each observation is considered an outlier.
    """
    # Exclude 'customer_id' from DBSCAN input
    features = customer_info.drop(columns=['customer_id']) if 'customer_id' in customer_info.columns else customer_info.copy()
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(features)
    customer_info = customer_info.copy()
    customer_info['cluster_dbscan'] = clusters
    customer_info['is_outlier_dbscan'] = customer_info['cluster_dbscan'] == -1
    return customer_info

def defining_params_dbscan_outliers(customer_info: pd.DataFrame, min_samples: int, eps_range: tuple) -> None:

    for eps in np.arange(*eps_range):

        print('=====================================')

        ci_dbscan = check_multidimensional_outliers_dbscan(customer_info, min_samples, eps)
        cluster_comparison = ci_dbscan.groupby(['cluster_dbscan', 'is_outlier_dbscan']).size().reset_index(name='number_of_customers')

        num_clusters = cluster_comparison[cluster_comparison['is_outlier_dbscan'] == False]['cluster_dbscan'].nunique()
        total_outliers = cluster_comparison[cluster_comparison["is_outlier_dbscan"] == True]["number_of_customers"].sum()

        print(f"With eps = {eps}\n\tNumber of clusters (not outliers): {num_clusters}\n\tTotal number of outliers: {total_outliers}")

def treat_multidimensional_outliers_dbscan(customer_info: pd.DataFrame, min_samples: int, eps: float) -> tuple:
    """
    Detects and removes multidimensional outliers from a DataFrame using the DBSCAN clustering algorithm.
    This function applies DBSCAN to the input DataFrame to identify outliers based on density. Rows classified as outliers are separated from the main dataset. The function returns two DataFrames: one with outliers removed, and another containing only the excluded outlier rows.

    Args:
        customer_info (pd.DataFrame): The input DataFrame containing customer information and features for outlier detection.
        min_samples (int): The minimum number of samples required in a neighborhood for a point to be considered a core point in DBSCAN.
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood (DBSCAN epsilon parameter).

    Returns:
        tuple:
            - pd.DataFrame: DataFrame with outliers removed (cleaned data).
            - pd.DataFrame: DataFrame containing only the excluded outlier rows.
    """
    customer_info = check_multidimensional_outliers_dbscan(customer_info, min_samples, eps)
    kept = customer_info[customer_info['is_outlier_dbscan'] == False].drop(columns=['cluster_dbscan', 'is_outlier_dbscan'])
    excluded = customer_info[customer_info['is_outlier_dbscan'] == True].drop(columns=['cluster_dbscan', 'is_outlier_dbscan'])
    return kept, excluded

#######################################
############ MISSING VALUES ###########
#######################################

def impute_kids_teens_home(customer_info: pd.DataFrame) -> pd.DataFrame:

    """
    Impute missing values in the 'kids_home' and 'teens_home' columns of a customer DataFrame using logical rules.
    This function fills missing values (NaN) in the 'kids_home' and 'teens_home' columns based on the following logic:
        1. If 'kids_home' is missing and 'teens_home' is present and greater than 0, set 'kids_home' to 0.
        2. If 'teens_home' is missing and 'kids_home' is present and greater than 0, set 'teens_home' to 0.
        3. If both 'kids_home' and 'teens_home' are missing, set both to 0.
    Args:
        customer_info (pd.DataFrame): DataFrame containing customer information, including 'kids_home' and 'teens_home' columns.
    Returns:
        pd.DataFrame: DataFrame with imputed values in 'kids_home' and 'teens_home'.
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
    Imputes missing values in the customer_info DataFrame using specific imputation functions.
    This function applies a series of imputation steps to handle missing values in the input DataFrame:
    1. Imputes missing values related to the number of kids and teens at home.
    2. Imputes missing values for education level.
    Args:
        customer_info (pd.DataFrame): DataFrame containing customer information with possible missing values.
    Returns:
        pd.DataFrame: DataFrame with missing values imputed for specified columns.
    """

    customer_info = impute_kids_teens_home(customer_info)
    customer_info = impute_education_level(customer_info)

    return customer_info

def knn_imputing(customer_info: pd.DataFrame, n_neighbors: int = 5) -> pd.DataFrame:
    """
    Imputes missing values in a customer information DataFrame using K-Nearest Neighbors (KNN) imputation.
    Parameters
    ----------
    customer_info : pd.DataFrame
        DataFrame containing customer information, including a 'customer_id' column and features with possible missing values.
    n_neighbors : int, optional (default=5)
        Number of neighboring samples to use for imputation.
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values imputed, preserving the original 'customer_id' column.
    """

    id_col = customer_info['customer_id']
    features = customer_info.drop(columns=['customer_id'])

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_features = imputer.fit_transform(features)
    imputed_data = pd.DataFrame(imputed_features, columns=features.columns, index=customer_info.index)

    imputed_data.insert(0, 'customer_id', id_col)
    return imputed_data


#######################################
############## ENCODING ###############
#######################################

def customer_info_encoding(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Encodes categorical features in the customer_info DataFrame.
    This function maps the 'education_level' column to the corresponding number of years of education
    and the 'customer_gender' column to a binary gender representation. The original 'education_level'
    and 'customer_gender' columns are dropped from the DataFrame.
    Parameters:
        customer_info (pd.DataFrame): DataFrame containing customer information with at least
            'education_level' and 'customer_gender' columns.
    Returns:
        pd.DataFrame: The input DataFrame with encoded 'education_years' and 'gender' columns,
        and without the original 'education_level' and 'customer_gender' columns.
    """

    education_mapping = {'4th': 4, '6th': 6, '9th': 9, 'Hs': 12, 'Bsc': 15, 'Msc': 17, 'Phd': 20}
    customer_info['education_years'] = customer_info['education_level'].map(education_mapping)

    gender_mapping = {'male': 0, 'female': 1}
    customer_info['gender'] = customer_info['customer_gender'].map(gender_mapping)

    customer_info.drop(['education_level', 'customer_gender'], axis = 1, inplace = True)

    return customer_info

#######################################
############### SCALING ###############
#######################################

def scaling(data: pd.DataFrame) -> (pd.DataFrame, MinMaxScaler):
    """
    Scales the features of a DataFrame using MinMaxScaler, optionally preserving a 'customer_id' column.
    Parameters:
        data (pd.DataFrame): Input DataFrame containing features to be scaled. If a 'customer_id' column is present, it will be excluded from scaling and preserved in the output.
    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]: 
            - A DataFrame with scaled features (and 'customer_id' column if present).
            - The fitted MinMaxScaler instance used for scaling.
    """

    if 'customer_id' in data.columns:
        id_col = data['customer_id']
        features = data.drop(columns=['customer_id'])
    else:
        id_col = None
        features = data

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_df = pd.DataFrame(scaled_features, columns=features.columns, index=data.index)

    if id_col is not None:
        scaled_df.insert(0, 'customer_id', id_col)

    scaled_df.attrs['scaler'] = scaler

    return scaled_df, scaler

def unscale_dataframe(scaled_df, scaler, columns=None):
    """
    Reverts the scaling transformation applied to a DataFrame using the provided scaler.
    Parameters:
        scaled_df (pd.DataFrame): The DataFrame containing scaled features. May include a 'customer_id' column.
        scaler (sklearn.base.TransformerMixin): The scaler object used to scale the data (e.g., StandardScaler, MinMaxScaler).
        columns (list, optional): List of column names to unscale. If None, all columns in scaled_df are considered except 'customer_id'.
    Returns:
        pd.DataFrame: A DataFrame with the specified columns unscaled. If 'customer_id' was present in the input, it is included as the first column.
    """

    cols = [col for col in (columns if columns is not None else scaled_df.columns) if col != 'customer_id']
    unscaled_array = scaler.inverse_transform(scaled_df[cols])
    unscaled_df = pd.DataFrame(unscaled_array, columns=cols, index=scaled_df.index)

    if 'customer_id' in scaled_df.columns:
        unscaled_df.insert(0, 'customer_id', scaled_df['customer_id'])
    return unscaled_df


#######################################
############# REDUNDANCY ##############
#######################################

def correlation_matrix(customer_info: pd.DataFrame) -> list:
    """
    Generates and displays a masked correlation matrix heatmap for the given customer information DataFrame.
    This function computes the correlation matrix of the input DataFrame, masks the upper triangle for better readability,
    and visualizes the result as an interactive heatmap using Plotly.
    Parameters:
        customer_info (pd.DataFrame): A DataFrame containing customer information with numerical features.
    Returns:
        list: This function does not return any value explicitly, but displays an interactive correlation matrix heatmap.
    """

    corr = customer_info.corr()

    # Mask upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.mask(mask)

    fig = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
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

def treat_redundancy(data: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Removes specified redundant columns from a pandas DataFrame.
    Args:
        data (pd.DataFrame): The input DataFrame from which columns will be removed.
        cols (list): List of column names to be dropped from the DataFrame.
    Returns:
        pd.DataFrame: A new DataFrame with the specified columns removed.
    """

    return data.drop(cols, axis = 1)