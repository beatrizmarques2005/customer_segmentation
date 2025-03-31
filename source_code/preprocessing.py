import pandas as pd
import numpy as np

import pandas as pd

######### GENERAL EXPLORATION #########

def initial_exploration(data):
    
    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(data.head())
    
    # Display the shape of the dataset
    print("\nShape of the dataset:")
    print(data.shape)
    
    # Display the data types of each column
    print("\ndata types of each column:")
    print(data.dtypes)
    
    # Display basic statistics of the dataset
    print("\nBasic statistics of the dataset:")
    print(data.describe().T)
    
    # Check for missing values
    print("\nMissing values in each column:")
    print(data.isnull().sum())

######### CUSTOMER_BASKET #########

def correct_customer_backet_format(df):
    pass


######### DUPLICATES #########

def check_duplicates(data):
    duplicate_count = data.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    return duplicate_count

def treat_duplicates(data):
    data = data.drop_duplicates()
    print("Duplicates have been removed.")
    return data


######### OUTLIERS #########



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
