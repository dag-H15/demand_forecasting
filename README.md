# Demand Forecasting System

A comprehensive machine learning project for predicting product demand using regression algorithms.

## 🎯 Project Overview

This system implements a complete machine learning workflow for demand forecasting:

- **Data Preprocessing**: Cleaning, feature engineering, categorical encoding, and feature scaling
- **Model Training**: Linear Regression and Ridge Regression algorithms
- **Model Evaluation**: Performance comparison using MAE, RMSE, and R² metrics
- **Interactive Web App**: Streamlit application with multiple analysis dashboards
- **Visualization**: Comprehensive charts and graphs for model interpretation

## 📁 Project Structure

```
demand_forecasting/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── main.py                   # Command-line prediction demo
├── data/
│   └── demand_forecasting.csv  # Training dataset
├── models/                   # Saved models and evaluation data
├── src/                      # Source code modules
│   ├── preprocess.py         # Data preprocessing functions
│   ├── train.py             # Model training and evaluation
│   └── predict.py           # Prediction functions
└── app/                      # Web application
|    └── app.py               # Streamlit dashboard
└──analysis_script.py
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Models
```bash
python src/train.py
```

### 3. Run Sample Prediction
```bash
python main.py
```

### 4. Launch Web Application
```bash
streamlit run app/app.py
```

## 📊 Application Features

### 🔮 Prediction Tab
- Interactive form with dropdown menus for categorical features
- Real-time demand prediction
- Input validation and summary display

### ⚖️ Model Comparison Tab
- Performance metrics comparison (MAE, RMSE)
- Interactive bar charts and scatter plots
- Residual analysis visualizations

### 📊 Data Analysis Tab
- Dataset overview and statistics
- Correlation heatmap
- Distribution plots and time series analysis
- Categorical feature exploration

### 🎯 Feature Importance Tab
- Random Forest feature importance ranking
- Cumulative importance analysis
- Detailed importance metrics

## 🧠 Machine Learning Details

### Algorithms Used
- **Linear Regression**: Baseline model for comparison
- **Random Forest**: Ensemble method for improved accuracy

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Square root of average squared errors

### Data Preprocessing
- Date feature extraction (Year, Month, Day)
- Categorical encoding using Label Encoding
- Missing value handling
- Feature scaling (handled by tree-based models)

## 📈 Results

Based on the evaluation:
- Random Forest typically outperforms Linear Regression
- Key factors: Inventory Level, Units Sold, Product ID, Store ID
- Model achieves reasonable prediction accuracy for demand forecasting

## 👥 Group Project Notes

This codebase is well-documented and modular for easy understanding:

- **Clear separation of concerns**: Preprocessing, training, prediction
- **Comprehensive comments**: Each function and section explained
- **Modular design**: Easy to modify or extend individual components
- **Professional structure**: Follows ML project best practices

## 🔧 Technical Requirements

- Python 3.8+
- Libraries: pandas, scikit-learn, streamlit, plotly, matplotlib, seaborn
- Dataset: demand_forecasting.csv (included)

## 📝 Usage Examples

### Command Line Prediction
```python
from src.predict import predict

input_data = {
    "Store ID": "S001",
    "Product ID": "P0001",
    "Category": "Electronics",
    # ... other features
}

prediction = predict(input_data)
print(f"Predicted demand: {prediction}")
```

### Web App Usage
1. Select feature values using dropdown menus
2. Click "Predict Demand" for instant results
3. Explore other tabs for detailed analysis

## 🎓 Educational Value

This project demonstrates:
- End-to-end machine learning workflow
- Model evaluation and comparison techniques
- Interactive data visualization
- Web application development with Streamlit
- Best practices for ML project organization

## Project Structure

- `data/`: Contains the dataset
- `models/`: Saved trained models and encoders
- `src/`: Source code for preprocessing, training, and prediction
- `app/`: Streamlit web application
- `main.py`: Script for sample prediction
- `requirements.txt`: Python dependencies

## Models Used

- Linear Regression (baseline)
- Random Forest Regressor

Evaluation metrics: MAE and RMSE