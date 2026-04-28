"""
Prediction Module

This module handles demand prediction using the trained Ridge Regression model.
It loads the model and encoders, processes input data, and returns predictions.
"""

import joblib
import pandas as pd
import os

def predict(input_data):
    """
    Predict demand based on input features.

    Args:
        input_data (dict): Dictionary containing all feature values

    Returns:
        float: Predicted demand value
    """
    # Define paths to model files
    model_dir = os.path.join(os.path.dirname(__file__), "../models")
    model_path = os.path.join(model_dir, "ridge_regression.pkl")
    encoders_path = os.path.join(model_dir, "encoders.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    # Load trained model, label encoders, and scaler
    model = joblib.load(model_path)
    encoders = joblib.load(encoders_path)
    scaler = joblib.load(scaler_path)

    # Define categorical columns that need encoding
    cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']

    # Encode categorical inputs using the same encoders from training
    for col in cat_cols:
        if col in input_data:
            input_data[col] = encoders[col].transform([str(input_data[col])])[0]

    # Add default date-derived features (these are derived from Date in training)
    # Using reasonable defaults for prediction
    input_data['Year'] = 2022  # Default year
    input_data['Month'] = 6    # Default month (June)
    input_data['Day'] = 15     # Default day

    # Convert input dictionary to DataFrame for prediction
    input_df = pd.DataFrame([input_data])

    # Scale the input features using the same scaler from training
    input_scaled = scaler.transform(input_df)

    # Make prediction using the trained model
    prediction = model.predict(input_scaled)

    # Return the predicted demand value
    return prediction[0]