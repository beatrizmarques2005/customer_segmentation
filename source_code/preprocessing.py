import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
from shapely.geometry import Point

######### GENERAL EXPLORATION #########

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


######### GENERAL CORRECTIONS #########

def general_customer_info_corrections(customer_info: pd.DataFrame) -> pd.DataFrame:
    """
    Perform general corrections and transformations on customer information data.
    This function processes a DataFrame containing customer information by:
    1. Splitting the `customer_name` column into two parts: `education_level` and `customer_name`.
       - The `education_level` is extracted from the prefix of the `customer_name` (before the first period).
       - The `customer_name` is updated to exclude the prefix and is stripped of leading/trailing whitespace.
    2. Converting the `customer_birthdate` column to a datetime format and calculating the customer's age in years.
       - The `age` is computed based on the difference between a fixed reference date (2023-06-09) and the birthdate.
    Parameters:
    -----------
    customer_info : pandas.DataFrame
        A DataFrame containing customer information with at least the following columns:
        - 'customer_name': str, the name of the customer (may include education level as a prefix).
        - 'customer_birthdate': str or datetime, the birthdate of the customer.
    Returns:
    --------
    pandas.DataFrame
        The modified DataFrame with the following changes:
        - A new column `education_level` containing the extracted education level (if available).
        - The `customer_name` column updated to exclude the education level prefix.
        - The `customer_birthdate` column converted to datetime format.
        - A new column `age` containing the customer's age in years (calculated as an integer).
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


######### DUPLICATES #########

def check_duplicates(data: pd.DataFrame) -> str:
    """
    Checks for duplicate rows in a given dataset.

    Args:
        data (pandas.DataFrame): The dataset to check for duplicates.

    Returns:
        str: A message indicating the number of duplicate rows in the dataset.
    """
    duplicate_count = data.duplicated().sum()
    return f"Number of duplicate rows: {duplicate_count}"

def treat_duplicates(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes duplicate rows from a pandas DataFrame if any are found.
    Args:
        data (pd.DataFrame): The input DataFrame to check and treat for duplicates.
    Returns:
        pd.DataFrame: The DataFrame with duplicates removed, if any were present.
    """

    if int(check_duplicates(data).split(' ')[-1]) == 0:
        print("No duplicates found.")

    else:
        data = data.drop_duplicates()
        print("Duplicates have been removed.")

    return data



######### OUTLIERS #########

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
        - 'customer_gender'
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
    categorical_columns = ['customer_gender', 'education_level']
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

def remove_outliers(data):
    outlier_conditions = (
        (data['kids_home'] <= 8) &
        (data['teens_home'] <= 4) &
        (data['number_complaints'] <= 4) &
        (data['distinct_stores_visited'] <= 8) &
        (data['lifetime_spend_groceries'] <= 100000) &
        (data['lifetime_spend_electronics'] <= 20000) &
        (data['lifetime_spend_vegetables'] <= 2600) &
        (data['lifetime_spend_nonalcohol_drinks'] <= 1400) &
        (data['lifetime_spend_alcohol_drinks'] <= 2800) &
        (data['lifetime_spend_meat'] <= 2600) &
        (data['lifetime_spend_fish'] <= 2800) &
        (data['lifetime_spend_hygiene'] <= 2400) &
        (data['lifetime_spend_videogames'] <= 1700) &
        (data['lifetime_spend_petfood'] <= 850) &
        (data['lifetime_total_distinct_products'] <= 600) &
        (data['percentage_of_products_bought_promotion'] >= -0.5) &
        (data['percentage_of_products_bought_promotion'] <= 1.5)
    )
    return data[outlier_conditions]

def amount_deleted_rows(original_df, final_df):
    shape_difference = (final_df.shape[0] - original_df.shape[0], final_df.shape[1] - original_df.shape[1])
    print(f'It was deleted: {round((original_df.shape[0] - final_df.shape[0]) / original_df.shape[0] * 100, 2)}% of the original train dataset.')

def remove_inconsistencies(data):
    data = data[data['percentage_of_products_bought_promotion'] >= 0]
    return data

def plot_geolocation_interactive(data, latitude_col='latitude', longitude_col='longitude'):
    # Ensure latitude and longitude columns exist
    if latitude_col not in data.columns or longitude_col not in data.columns:
        raise ValueError(f"Columns '{latitude_col}' and '{longitude_col}' must exist in the dataframe.")
    
    # Create a scatter mapbox plot using Plotly
    fig = go.Figure(go.Scattermapbox(
        lat=data[latitude_col],
        lon=data[longitude_col],
        mode='markers',
        marker=go.scattermapbox.Marker(size=9, color='red'),
        text=data.index,  # Display index or other relevant info on hover
        hoverinfo='text'
    ))
    
    # Set the layout for the map
    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            zoom=3,  # Adjust zoom level as needed
            center=dict(lat=data[latitude_col].mean(), lon=data[longitude_col].mean())
        ),
        margin={"r":0,"t":0,"l":0,"b":0},  # Remove margins for a cleaner look
        title="Interactive Geolocation Map"
    )
    
    fig.show()

def remove_far_from_coast(data, latitude_col='latitude', longitude_col='longitude'):
    # Load a world map shapefile using GeoPandas
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world[world['geometry'].type == 'Polygon']  # Keep only polygons (land masses)
    # Convert the DataFrame to a GeoDataFrame
    geometry = [Point(xy) for xy in zip(data[longitude_col], data[latitude_col])]
    geo_data = gpd.GeoDataFrame(data, geometry=geometry)
    # Re-project to a projected CRS for accurate buffering
    world = world.to_crs(epsg=3395)  # Use a projected CRS like EPSG:3395 (World Mercator)
    # Buffer the land polygons by 1 km
    buffered_land = world.buffer(1000)  # Buffer by 1000 meters (1 km)
    # Re-project back to the original geographic CRS
    buffered_land = buffered_land.to_crs(epsg=4326)  # EPSG:4326 is WGS84 (geographic CRS)
    # Check if points are within the buffered land polygons
    geo_data['near_coast'] = geo_data['geometry'].apply(
        lambda point: any(buffered_land.contains(point))
    )
    # Filter out points that are not near the coast
    filtered_data = geo_data[geo_data['near_coast']].drop(columns=['geometry', 'near_coast'])
    return filtered_data



######### MISSING VALUES #########

def impute_loyalty_card(customer_info:pd.DataFrame) -> pd.DataFrame:
    """
    Imputes missing values in the 'loyalty_card_number' column of the customer_info DataFrame with 0.

    Parameters:
    customer_info (pd.DataFrame): A pandas DataFrame containing customer information, 
                                  including a 'loyalty_card_number' column.

    Returns:
    pd.DataFrame: The updated DataFrame with missing values in 'loyalty_card_number' replaced by 0.
    """
    customer_info['loyalty_card_number'].replace(np.nan, 0, inplace = True)
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

def impute_lifetime_spend_alcohol_drinks(customer_info):
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


def impute_education_level(customer_info):
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


######### ENCODING #########

def customer_info_encoding(customer_info):
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
    customer_info.drop('customer_name', axis = 1, inplace = True)

    # 2
    education_mapping = {'4th': 4, '6th': 6, '9th': 9, 'Hs': 12, 'Bsc': 16, 'Msc': 18, 'Phd': 21}
    customer_info['education_years'] = customer_info['education_level'].map(education_mapping)

    # 3
    gender_map = {'male': 0, 'female': 1}
    customer_info['gender'] = customer_info['customer_gender'].map(gender_map)

    # 4
    customer_info.drop(['customer_gender', 'education_level'], axis = 1, inplace = True)

    return customer_info

######### INCONSISTENCIES #########


######### SCALING #########

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

    data = scaler.fit_transform(data)

    data = pd.DataFrame(data, columns=data.columns)
 
    return data






