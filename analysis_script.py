import pandas as pd
import numpy as np
import joblib
import os

# Load dataset
df = pd.read_csv('data/demand_forecasting.csv')

print("="*80)
print("1. DATASET OVERVIEW")
print("="*80)
print(f"Total Rows: {len(df)}")
print(f"Total Columns: {len(df.columns)}")
print(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\nColumn Names and Types:")
for col in df.columns:
    print(f"  {col}: {df[col].dtype}")

print("\n" + "="*80)
print("2. TARGET VARIABLE ANALYSIS (DEMAND)")
print("="*80)
print(f"Min: {df['Demand'].min()}")
print(f"Max: {df['Demand'].max()}")
print(f"Mean: {df['Demand'].mean():.2f}")
print(f"Median: {df['Demand'].median():.2f}")
print(f"Std Dev: {df['Demand'].std():.2f}")
print(f"Q1 (25%): {df['Demand'].quantile(0.25):.2f}")
print(f"Q3 (75%): {df['Demand'].quantile(0.75):.2f}")
print(f"IQR: {df['Demand'].quantile(0.75) - df['Demand'].quantile(0.25):.2f}")
print(f"Skewness: {df['Demand'].skew():.4f}")
print(f"Kurtosis: {df['Demand'].kurtosis():.4f}")

print("\n" + "="*80)
print("3. CATEGORICAL FEATURES ANALYSIS")
print("="*80)
categorical_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
for col in categorical_cols:
    unique_vals = df[col].nunique()
    print(f"\n{col}:")
    print(f"  Unique Values: {unique_vals}")
    print(f"  Values: {sorted(df[col].unique())}")

print("\n" + "="*80)
print("4. NUMERICAL FEATURES ANALYSIS")
print("="*80)
numerical_cols = ['Inventory Level', 'Units Sold', 'Units Ordered', 'Price', 'Discount', 'Competitor Pricing', 'Promotion', 'Epidemic']
for col in numerical_cols:
    print(f"\n{col}:")
    print(f"  Min: {df[col].min()}")
    print(f"  Max: {df[col].max()}")
    print(f"  Mean: {df[col].mean():.2f}")
    print(f"  Std: {df[col].std():.2f}")

print("\n" + "="*80)
print("5. MISSING VALUES AND DUPLICATES")
print("="*80)
print("Missing Values:")
print(df.isnull().sum())
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

print("\n" + "="*80)
print("6. CORRELATION ANALYSIS WITH TARGET")
print("="*80)
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['Demand'].sort_values(ascending=False)
print("Correlation with Demand:")
for feat, corr in correlations.items():
    print(f"  {feat}: {corr:.4f}")

print("\n" + "="*80)
print("7. STORE-WISE ANALYSIS")
print("="*80)
store_analysis = df.groupby('Store ID').agg({
    'Demand': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
print(store_analysis)

print("\n" + "="*80)
print("8. CATEGORY-WISE ANALYSIS")
print("="*80)
category_analysis = df.groupby('Category').agg({
    'Demand': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
print(category_analysis)

print("\n" + "="*80)
print("9. REGION-WISE ANALYSIS")
print("="*80)
region_analysis = df.groupby('Region').agg({
    'Demand': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
print(region_analysis)

print("\n" + "="*80)
print("10. SEASONALITY ANALYSIS")
print("="*80)
season_analysis = df.groupby('Seasonality').agg({
    'Demand': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
print(season_analysis)

print("\n" + "="*80)
print("11. WEATHER IMPACT ANALYSIS")
print("="*80)
weather_analysis = df.groupby('Weather Condition').agg({
    'Demand': ['mean', 'std', 'min', 'max', 'count']
}).round(2)
print(weather_analysis)

print("\n" + "="*80)
print("12. BINARY FEATURES ANALYSIS")
print("="*80)
print("\nPromotion Impact:")
promo_analysis = df.groupby('Promotion').agg({
    'Demand': ['mean', 'count']
}).round(2)
print(promo_analysis)

print("\nEpidemic Impact:")
epidemic_analysis = df.groupby('Epidemic').agg({
    'Demand': ['mean', 'count']
}).round(2)
print(epidemic_analysis)

print("\n" + "="*80)
print("13. LOADING MODEL EVALUATION METRICS")
print("="*80)
try:
    metrics = joblib.load('models/metrics.pkl')
    print("\nModel Performance Metrics:")
    for model_name, scores in metrics.items():
        print(f"\n{model_name}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.4f}")
except:
    print("Metrics file not found")

print("\n" + "="*80)
print("14. FEATURE IMPORTANCE ANALYSIS")
print("="*80)
try:
    feature_importance = pd.read_csv('models/feature_importance.csv')
    print("\nTop 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    cumsum = feature_importance['importance'].cumsum()
    cumsum_norm = cumsum / cumsum.iloc[-1]
    threshold_idx = (cumsum_norm <= 0.8).sum()
    print(f"\nFeatures explaining 80% of importance: {threshold_idx}")
except:
    print("Feature importance file not found")

print("\n" + "="*80)
print("15. EVALUATION DATA ANALYSIS")
print("="*80)
try:
    eval_data = pd.read_csv('models/evaluation_data.csv')
    print(f"Evaluation samples: {len(eval_data)}")
    print("\nLinear Regression Performance:")
    lr_mae = np.mean(np.abs(eval_data['Actual'] - eval_data['Linear_Regression_Pred']))
    lr_rmse = np.sqrt(np.mean((eval_data['Actual'] - eval_data['Linear_Regression_Pred'])**2))
    lr_r2 = 1 - (np.sum((eval_data['Actual'] - eval_data['Linear_Regression_Pred'])**2) / np.sum((eval_data['Actual'] - eval_data['Actual'].mean())**2))
    print(f"  MAE: {lr_mae:.4f}")
    print(f"  RMSE: {lr_rmse:.4f}")
    print(f"  R²: {lr_r2:.4f}")
    
    print("\nRidge Regression Performance:")
    ridge_mae = np.mean(np.abs(eval_data['Actual'] - eval_data['Ridge_Regression_Pred']))
    ridge_rmse = np.sqrt(np.mean((eval_data['Actual'] - eval_data['Ridge_Regression_Pred'])**2))
    ridge_r2 = 1 - (np.sum((eval_data['Actual'] - eval_data['Ridge_Regression_Pred'])**2) / np.sum((eval_data['Actual'] - eval_data['Actual'].mean())**2))
    print(f"  MAE: {ridge_mae:.4f}")
    print(f"  RMSE: {ridge_rmse:.4f}")
    print(f"  R²: {ridge_r2:.4f}")
    
    print("\nActual Demand Distribution in Test Set:")
    print(f"  Mean: {eval_data['Actual'].mean():.2f}")
    print(f"  Std: {eval_data['Actual'].std():.2f}")
    print(f"  Min: {eval_data['Actual'].min():.2f}")
    print(f"  Max: {eval_data['Actual'].max():.2f}")
except:
    print("Evaluation data file not found")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
