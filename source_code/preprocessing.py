import pandas as pd
import numpy as np

#MISSING VALUES

#function for manually imputing for loyalty card
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