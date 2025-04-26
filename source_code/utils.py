
import pandas as pd


def load_dataset(dataset_path: str) -> pd.DataFrame:
    """
    Loads a dataset from a specified file path. Supports CSV and Excel file formats.
    Args:
        dataset_path (str): The file path to the dataset. Must be a .csv or .xlsx file.
    Returns:
        pandas.DataFrame: The loaded dataset as a pandas DataFrame.
    Raises:
        ValueError: If the file is not in CSV or Excel format.
    Notes:
        - If the first column of the dataset is named 'Unnamed: 0', it will be automatically dropped.
    """

    if dataset_path.endswith('.csv'):
        data = pd.read_csv(dataset_path)
    
    elif dataset_path.endswith('.xlsx'):
        data = pd.read_excel(dataset_path)
    
    else:
        raise ValueError("The dataset must be in CSV or EXCEL format.")
    
    if data.columns[0] == 'Unnamed: 0':
        data = data.drop(columns=data.columns[0])

    return data


def save_dataset(data: pd.DataFrame, format: str, path: str) -> None:
    """
    Saves a pandas DataFrame to a file in the specified format.

    Parameters:
    -----------
    data : pd.DataFrame
        The DataFrame to be saved.
    format : str
        The format in which to save the file. 
    path : str
        The file path where the dataset will be saved.
    Raises:
    -------
    ValueError
        If the specified format is not 'csv' or 'xlsx'.
    Returns:
    --------
    None
    """

    if format == 'csv':
        data.to_csv(path, index = False)

    elif format == 'xlsx':
        data.to_excel(path, index = False)

    else:
        raise ValueError("The format must be in CSV or EXCEL.")
    

def amount_deleted_rows(original_df: pd.DataFrame, final_df: pd.DataFrame) -> None:
    """
    Calculates and prints the percentage of rows deleted from the original DataFrame 
    to the final DataFrame.
    Args:
        original_df (pd.DataFrame): The original DataFrame before any rows were removed.
        final_df (pd.DataFrame): The final DataFrame after rows were removed.
    Returns:
        None: This function only prints the percentage of rows deleted.
    """

    print(f'It was deleted: {round((original_df.shape[0] - final_df.shape[0]) / original_df.shape[0] * 100, 2)}% of the original train dataset.')