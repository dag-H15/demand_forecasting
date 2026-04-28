"""
Model Training Module

This module trains and evaluates two regression models:
1. Linear Regression (baseline)
2. Ridge Regression (regularized version)

It also saves evaluation data for visualization purposes.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from preprocess import load_and_clean
import os
import numpy as np

# Load and preprocess the data
data_path = os.path.join(os.path.dirname(__file__), "../data/demand_forecasting.csv")
df, encoders = load_and_clean(data_path)

# Separate features (X) and target variable (y)
X = df.drop('Demand', axis=1)  # All columns except Demand
y = df['Demand']  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features for proper coefficient interpretation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize and train Linear Regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

# Initialize and train Ridge Regression model with cross-validated alpha
ridge_alphas = np.logspace(-4, 3, 20)
ridge_cv = RidgeCV(alphas=ridge_alphas, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train_scaled, y_train)

ridge = ridge_cv
best_alpha = ridge_cv.alpha_
print('Best Ridge alpha:', best_alpha)

# Make predictions on test set
lr_pred = lr.predict(X_test_scaled)
ridge_pred = ridge.predict(X_test_scaled)

# Calculate evaluation metrics
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

ridge_mae = mean_absolute_error(y_test, ridge_pred)
ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
ridge_r2 = r2_score(y_test, ridge_pred)

# Print results
print("Linear Regression - MAE:", lr_mae, "RMSE:", lr_rmse, "R²:", lr_r2)
print("Ridge Regression - MAE:", ridge_mae, "RMSE:", ridge_rmse, "R²:", ridge_r2)

# Determine which model performs better
if ridge_mae < lr_mae:
    print("Ridge Regression performs better.")
else:
    print("Linear Regression performs better.")

# Define model directory path
model_dir = os.path.join(os.path.dirname(__file__), "../models")

# Save evaluation data for visualization in the app
eval_data = pd.DataFrame({
    'Actual': y_test,
    'Linear_Regression_Pred': lr_pred,
    'Ridge_Regression_Pred': ridge_pred
})
eval_data.to_csv(os.path.join(model_dir, 'evaluation_data.csv'), index=False)

# Save feature importance using Ridge coefficients (absolute values for ranking)
feature_names = X.columns
ridge_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': np.abs(ridge.coef_)
}).sort_values('importance', ascending=False)
ridge_importance.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)

# Save performance metrics
metrics = {
    'Linear_Regression': {'MAE': lr_mae, 'RMSE': lr_rmse, 'R2': lr_r2},
    'Ridge_Regression': {'MAE': ridge_mae, 'RMSE': ridge_rmse, 'R2': ridge_r2}
}
joblib.dump(metrics, os.path.join(model_dir, 'metrics.pkl'))

# Save trained models, scaler, and encoders
joblib.dump(lr, os.path.join(model_dir, "linear_regression.pkl"))
joblib.dump(ridge, os.path.join(model_dir, "ridge_regression.pkl"))
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
joblib.dump(encoders, os.path.join(model_dir, "encoders.pkl"))

print("Models trained and saved successfully!")