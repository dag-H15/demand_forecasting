"""
Data Preprocessing Module

This module handles loading and cleaning the demand forecasting dataset.
It performs data transformation and categorical encoding.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean(path):
    """
    Load and preprocess the demand forecasting dataset.

    Args:
        path (str): Path to the CSV file

    Returns:
        tuple: (processed_dataframe, encoders_dict)
    """
    # Load the dataset
    df = pd.read_csv(path)

    # Handle missing values by removing rows with NaN
    df = df.dropna()

    # Convert date column and extract temporal features
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day

    # Remove the original date column as we now have separate features
    df = df.drop(columns=['Date'])

    # Define categorical columns that need encoding
    cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']

    # Dictionary to store label encoders for later use in predictions
    encoders = {}

    # Encode each categorical column to numerical values
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df, encoders