
import pandas as pd


def load_dataset(dataset_path):

    if dataset_path.endswith('.csv'):
        data = pd.read_csv(dataset_path)
    
    elif dataset_path.endswith('.xlsx'):
        data = pd.read_excel(dataset_path)
    
    else:
        raise ValueError("The dataset must be in CSV or EXCEL format.")
    
    if data.columns[0] == 'Unnamed: 0':
        data = data.drop(columns=data.columns[0])

    return data


def save_dataset(data, format, path):

    if format == 'csv':
        data.to_csv(path, index = False)

    elif format == 'xlsx':
        data.to_excel(path, index = False)

    else:
        raise ValueError("The format must be in CSV or EXCEL.")