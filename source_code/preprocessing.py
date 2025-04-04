import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

######### GENERAL EXPLORATION #########

def initial_exploration(data):
    
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

######### DATA TYPES #########

def customer_info_data_types(customer_info):

    # customer_id                                  int64
    # customer_name                               object
    # customer_gender                             object
    # customer_birthdate                          object --> datetime64[ns]
    # customer_info['birth_date'] = pd.to_datetime(customer_info['birth_date'], errors='coerce')
    # kids_home                                  float64
    # teens_home                                 float64
    # number_complaints                          float64
    # distinct_stores_visited                    float64
    # lifetime_spend_groceries                   float64
    # lifetime_spend_electronics                 float64
    # typical_hour                               float64
    # lifetime_spend_vegetables                  float64
    # lifetime_spend_nonalcohol_drinks           float64
    # lifetime_spend_alcohol_drinks              float64
    # lifetime_spend_meat                        float64
    # lifetime_spend_fish                        float64
    # lifetime_spend_hygiene                     float64
    # lifetime_spend_videogames                  float64
    # lifetime_spend_petfood                     float64
    # lifetime_total_distinct_products           float64
    # percentage_of_products_bought_promotion    float64
    # year_first_transaction                     float64
    # loyalty_card_number                        float64
    # latitude                                   float64
    # longitude                                  float64

    pass

######### GENERAL CORRECTIONS #########

def general_customer_info_corrections(customer_info):

    # customer_name --> customer_name + education_level
    split_names = customer_info['customer_name'].str.split('.', n=1, expand=True)

    customer_info['education_level'] = split_names[0].where(split_names[1].notna(), np.nan)
    customer_info['customer_name'] = split_names[1].fillna(split_names[0]).str.strip()

    # birth_date --> age (years)
    customer_info['customer_birthdate'] = pd.to_datetime(customer_info['customer_birthdate'], errors='coerce') # object --> datetime64[ns]
    customer_info['age'] = (pd.to_datetime('today') - customer_info['customer_birthdate']).dt.days // 365.25
    # ?????????? customer_info['birthdate_month'] = customer_info['customer_birthdate'].dt.month

    return customer_info


def general_customer_basket_corrections(customer_basket):

    customer_basket['items_count'] = customer_basket['list_of_goods'].apply(len)

    # Ensure the list_of_goods column is converted from string representation to actual lists if needed
    customer_basket['list_of_goods'] = customer_basket['list_of_goods'].apply(eval)

    # Explode the list_of_goods column to create one row per item
    customer_basket_exploded = customer_basket.explode('list_of_goods')

    # Group by the items and calculate the count of invoices and customers
    items_summary = customer_basket_exploded.groupby('list_of_goods').agg(
        invoice_count=('invoice_id', 'nunique'),
        customer_count=('customer_id', 'nunique')
    ).reset_index()

    return customer_basket, items_summary


######### DUPLICATES #########

def check_duplicates(data):
    duplicate_count = data.duplicated().sum()
    return f"Number of duplicate rows: {duplicate_count}"

def treat_duplicates(data):

    if int(check_duplicates(data).split(' ')[-1]) == 0:
        print("No duplicates found.")

    else:
        data = data.drop_duplicates()
        print("Duplicates have been removed.")

    return data


######### OUTLIERS #########

def check_outliers_numerical_boxplot(df):

    numerical_columns = list(df.select_dtypes(include=['number']))
    columns_to_remove=['customer_id', 'kids_home', 'teens_home', 'number_complaints', 'year_first_transaction', 'distinct_stores_visited', 'typical_hour']
    for value in columns_to_remove:
        while value in numerical_columns:  # Ensures all occurrences are removed
            numerical_columns.remove(value)
    # Exclude categorical columns that are numerical but represent categories
    fig = go.Figure()

    buttons = []

    for idx, column in enumerate(numerical_columns):
        fig.add_trace(go.Box(
            x=df[column],
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

def check_outliers_numerical_histogram(df):
    numerical_columns = list(df.select_dtypes(include=['number']))
    columns_to_remove = ['customer_id', 'kids_home', 'teens_home', 'number_complaints', 'year_first_transaction', 'distinct_stores_visited', 'typical_hour']
    
    numerical_columns = [col for col in numerical_columns if col not in columns_to_remove]
    
    fig = go.Figure()
    buttons = []
    
    for idx, column in enumerate(numerical_columns):
        fig.add_trace(go.Histogram(
            x=df[column],
            name=f'<i>{column}</i>',
            marker_color='darkgreen',
            visible=(idx == 0),
            nbinsx=30  # Adjust bin size as needed
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

def check_outliers_categorical(df):
    categorical_columns = ['customer_gender', 'education_level']
    numerical_with_categorical_behaviour= ['kids_home', 'teens_home', 'number_complaints', 'year_first_transaction', 'distinct_stores_visited', 'typical_hour']
    #the others were continuous, that's why they were not included 
    categorical_columns.extend(numerical_with_categorical_behaviour)
    fig = go.Figure()

    buttons = []

    for idx, column in enumerate(categorical_columns):
        value_counts = df[column].value_counts()

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

# impute education_level --> https://edu.azores.gov.pt/seccoes/matriculas-escolaridade-obrigatoria/
# '4th', '6th', '9th', 'HighSchool', 'Bsc', 'Msc', 'Phd'


def imput_educ(data):
    data['education_level'] = np.where(data['age']>= 59, '4th', data['education_level'])
    data['education_level'] = np.where((data['age']>= 44)  & (data['age'] <=58), '6th', data['education_level'])
    data['education_level'] = np.where((data['age']>= 30)  & (data['age'] <=43), '9th', data['education_level'])
    data['education_level'] = np.where(data['age']<= 29, 'Hs', data['education_level'])

    return data

def combine_impute(data):
    data1 = impute_loyalty_card(data)
    data2 = impute_kids_home(data1)
    data3 = impute_teens_home(data2)
    data4 = impute_teens_kids_home(data3)
    data5 = impute_lifetime_spend_alcohol_drinks(data4)
    data6 = imput_educ(data5)
    return data6


######### ENCODING #########

def customer_info_encoding(customer_info):

    customer_info.drop('customer_name', axis = 1, inplace = True)

    # Encode education_level into years of education
    education_mapping = {'4th': 4, '6th': 6, '9th': 9, 'Hs': 12, 'Bsc': 16, 'Msc': 18, 'Phd': 21}
    customer_info['education_years'] = customer_info['education_level'].map(education_mapping)

    # Encode customer_gender into binary values (0 for Male, 1 for Female)
    customer_info['customer_gender'] = customer_info['customer_gender'].map({'Male': 0, 'Female': 1})

    return customer_info

######### INCONSISTENCIES #########


######### SCALING #########

def customer_info_scaling(data):
    
    scaler = MinMaxScaler()
    # ???????? which scaler
    df_imputed = scaler.fit_transform(data)

    df_imputed = pd.DataFrame(df_imputed, columns=data.columns)
 
    return df_imputed

def KNN_imputing(data, n_neighbors, weights):

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)

    df_imputed = imputer.fit_transform(data)

    df_imputed = pd.DataFrame(df_imputed, columns=data.columns)

    return df_imputed




