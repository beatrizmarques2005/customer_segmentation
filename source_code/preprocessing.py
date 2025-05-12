import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
from shapely.geometry import Point
from sklearn.cluster import MeanShift, DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from minisom import MiniSom
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
import matplotlib.pyplot as plt

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


######### OUTLIERS #########

#why

######### MISSING VALUES #########

def impute_loyalty_card(data):
    data['loyalty_card_number'].replace(np.nan, 0, inplace = True)
    return data

def impute_kids_home(data):
    data['kids_home'] = np.where(((data['kids_home'].isna()) & ((~data['teens_home'].isna()) & (data['teens_home'] > 0))), 0, data['kids_home'])
    return data

def impute_teens_home(data):
    data['teens_home'] = np.where(((data['teens_home'].isna()) & ((~data['kids_home'].isna()) & (data['kids_home'] > 0))), 0, data['teens_home'])
    return data

def impute_teens_kids_home(data):
    mask = data['kids_home'].isna() & data['teens_home'].isna()
    data['kids_home'] = np.where(mask, 0, data['kids_home'])
    data['teens_home'] = np.where(mask, 0, data['teens_home'])
    return data

def impute_lifetime_spend_alcohol_drinks(data):
    data['lifetime_spend_alcohol_drinks'].replace(np.nan, 0, inplace = True)


######### ENCODING #########


######### INCONSISTENCIES #########


######### SCALING #########
