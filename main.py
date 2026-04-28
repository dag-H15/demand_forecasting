"""
Main Script for Demand Forecasting

This script demonstrates how to use the trained model to make predictions
using sample input data.
"""

from src.predict import predict

# Sample input data for testing the prediction
# This represents a typical product scenario with all required features
sample_input = {
    "Store ID": "S001",           # Store identifier
    "Product ID": "P0001",        # Product identifier
    "Category": "Electronics",    # Product category
    "Region": "North",            # Geographic region
    "Inventory Level": 195,       # Current stock level
    "Units Sold": 102,            # Historical sales
    "Units Ordered": 252,         # Recent orders
    "Price": 72.72,               # Product price
    "Discount": 5,                # Applied discount
    "Weather Condition": "Snowy", # Weather impact
    "Promotion": 0,               # Promotion status (0=No, 1=Yes)
    "Competitor Pricing": 85.73,  # Competitor's price
    "Seasonality": "Winter",      # Seasonal factor
    "Epidemic": 0,                # Epidemic impact (0=No, 1=Yes)
    "Year": 2022,                 # Year
    "Month": 1,                   # Month
    "Day": 1                      # Day
}

# Make prediction and display result
predicted_demand = predict(sample_input)
print(f"Predicted Demand: {predicted_demand:.2f} units")