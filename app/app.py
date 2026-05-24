"""
Demand Forecasting Web Application

A comprehensive Streamlit application for demand forecasting with:
- Interactive prediction interface
- Model comparison visualizations
- Data analysis dashboard
- Feature importance analysis

This app demonstrates a complete machine learning workflow.
"""

import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import joblib

# Add source directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from predict import predict

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Store Retail Product Demand Prediction System",
    page_icon="📊",
    layout="wide"
)

# Cache data loading functions for better performance
@st.cache_data
def load_visualization_data():
    """Load evaluation data, feature importance, and metrics for visualizations."""
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    eval_data = pd.read_csv(os.path.join(model_dir, 'evaluation_data.csv'))
    feature_importance = pd.read_csv(os.path.join(model_dir, 'feature_importance.csv'))
    metrics = joblib.load(os.path.join(model_dir, 'metrics.pkl'))
    return eval_data, feature_importance, metrics

@st.cache_data
def load_original_data():
    """Load the original dataset for analysis."""
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'demand_forecasting.csv')
    df = pd.read_csv(data_path)
    return df

st.title("📊 Multi-Store Retail Product Demand Prediction System")

# Sidebar navigation for different app sections
page = st.sidebar.selectbox("Choose a page", ["Prediction", "Model Comparison", "Data Analysis", "Feature Importance"])

if page == "Prediction":
    st.header("🔮 Make a Prediction")

    # Create two columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Categorical Features")
        # Dropdown menus ensure only valid inputs are selected
        store_id = st.selectbox("Store ID", ['S001', 'S002', 'S003', 'S004', 'S005'], index=0)
        product_id = st.selectbox("Product ID", ['P0001', 'P0002', 'P0003', 'P0004', 'P0005', 'P0006', 'P0007', 'P0008', 'P0009', 'P0010', 'P0011', 'P0012', 'P0013', 'P0014', 'P0015', 'P0016', 'P0017', 'P0018', 'P0019', 'P0020'], index=0)
        category = st.selectbox("Category", ['Clothing', 'Electronics', 'Furniture', 'Groceries', 'Toys'], index=1)
        region = st.selectbox("Region", ['East', 'North', 'South', 'West'], index=1)
        weather_condition = st.selectbox("Weather Condition", ['Cloudy', 'Rainy', 'Snowy', 'Sunny'], index=2)
        seasonality = st.selectbox("Seasonality", ['Autumn', 'Spring', 'Summer', 'Winter'], index=3)

    with col2:
        st.subheader("Numerical Features")
        inventory_level = st.number_input("Inventory Level", min_value=0, value=195)
        units_sold = st.number_input("Units Sold", min_value=0, value=102)
        units_ordered = st.number_input("Units Ordered", min_value=0, value=252)
        price = st.number_input("Price", min_value=0.0, value=72.72)
        discount = st.number_input("Discount", min_value=0.0, value=5.0)
        promotion = st.selectbox("Promotion", [0, 1], index=0)
        competitor_pricing = st.number_input("Competitor Pricing", min_value=0.0, value=85.73)
        epidemic = st.selectbox("Epidemic", [0, 1], index=0)
        year = st.number_input("Year", min_value=2000, max_value=2030, value=2022)
        month = st.number_input("Month", min_value=1, max_value=12, value=1)
        day = st.number_input("Day", min_value=1, max_value=31, value=1)

    if st.button("Predict Demand", type="primary"):
        input_data = {
            "Store ID": store_id,
            "Product ID": product_id,
            "Category": category,
            "Region": region,
            "Inventory Level": inventory_level,
            "Units Sold": units_sold,
            "Units Ordered": units_ordered,
            "Price": price,
            "Discount": discount,
            "Weather Condition": weather_condition,
            "Promotion": promotion,
            "Competitor Pricing": competitor_pricing,
            "Seasonality": seasonality,
            "Epidemic": epidemic,
            "Year": year,
            "Month": month,
            "Day": day
        }

        try:
            prediction = predict(input_data)
            st.success(f"🎯 Predicted Demand: **{prediction:.2f} units**")

            # Show input summary
            st.subheader("Input Summary")
            summary_df = pd.DataFrame(list(input_data.items()), columns=['Feature', 'Value'])
            st.dataframe(summary_df)

        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")

elif page == "Model Comparison":
    st.header("⚖️ Model Comparison")

    eval_data, _, metrics = load_visualization_data()

    # Metrics comparison
    st.subheader("📈 Performance Metrics")

    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df.style.highlight_min(axis=0))

    # Bar chart for metrics
    fig_metrics = go.Figure()
    fig_metrics.add_trace(go.Bar(name='MAE', x=list(metrics.keys()), y=[m['MAE'] for m in metrics.values()]))
    fig_metrics.add_trace(go.Bar(name='RMSE', x=list(metrics.keys()), y=[m['RMSE'] for m in metrics.values()]))
    fig_metrics.update_layout(title="Model Performance Comparison", barmode='group')
    st.plotly_chart(fig_metrics)

    # Actual vs Predicted scatter plots
    st.subheader("🎯 Actual vs Predicted Values")

    col1, col2 = st.columns(2)

    with col1:
        fig_lr = px.scatter(eval_data, x='Actual', y='Linear_Regression_Pred',
                           title="Linear Regression: Actual vs Predicted")
        fig_lr.add_trace(go.Scatter(x=[eval_data['Actual'].min(), eval_data['Actual'].max()],
                                   y=[eval_data['Actual'].min(), eval_data['Actual'].max()],
                                   mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
        st.plotly_chart(fig_lr)

    with col2:
        fig_ridge = px.scatter(eval_data, x='Actual', y='Ridge_Regression_Pred',
                           title="Ridge Regression: Actual vs Predicted")
        fig_ridge.add_trace(go.Scatter(x=[eval_data['Actual'].min(), eval_data['Actual'].max()],
                                   y=[eval_data['Actual'].min(), eval_data['Actual'].max()],
                                   mode='lines', name='Perfect Prediction', line=dict(dash='dash')))
        st.plotly_chart(fig_ridge)

    # Residual plots
    st.subheader("📊 Residual Analysis")

    eval_data['LR_Residuals'] = eval_data['Actual'] - eval_data['Linear_Regression_Pred']
    eval_data['Ridge_Residuals'] = eval_data['Actual'] - eval_data['Ridge_Regression_Pred']

    fig_resid = go.Figure()
    fig_resid.add_trace(go.Scatter(x=eval_data['Actual'], y=eval_data['LR_Residuals'],
                                  mode='markers', name='Linear Regression Residuals'))
    fig_resid.add_trace(go.Scatter(x=eval_data['Actual'], y=eval_data['Ridge_Residuals'],
                                  mode='markers', name='Ridge Regression Residuals'))
    fig_resid.add_hline(y=0, line_dash="dash", line_color="red")
    fig_resid.update_layout(title="Residual Plot", xaxis_title="Actual Values", yaxis_title="Residuals")
    st.plotly_chart(fig_resid)

elif page == "Data Analysis":
    st.header("📊 Data Analysis")

    df = load_original_data()

    # Dataset overview
    st.subheader("📋 Dataset Overview")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())

    # Basic statistics
    st.subheader("📈 Basic Statistics")
    st.dataframe(df.describe())

    # Correlation heatmap
    st.subheader("🔗 Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                        title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr)

    # Distribution of target variable
    st.subheader("🎯 Demand Distribution")
    fig_demand = px.histogram(df, x='Demand', nbins=50, title="Distribution of Demand")
    st.plotly_chart(fig_demand)

    # Categorical feature distributions
    st.subheader("📊 Categorical Features Distribution")

    cat_cols = ['Store ID', 'Product ID', 'Category', 'Region', 'Weather Condition', 'Seasonality']
    selected_cat = st.selectbox("Select categorical feature:", cat_cols)

    fig_cat = px.bar(df[selected_cat].value_counts().head(20),
                    title=f"Top 20 {selected_cat} Distribution")
    st.plotly_chart(fig_cat)

    # Time series analysis
    st.subheader("📅 Time Series Analysis")
    df['Date'] = pd.to_datetime(df['Date'])
    daily_demand = df.groupby('Date')['Demand'].mean().reset_index()

    fig_ts = px.line(daily_demand, x='Date', y='Demand',
                    title="Average Daily Demand Over Time")
    st.plotly_chart(fig_ts)

elif page == "Feature Importance":
    st.header("🎯 Feature Importance Analysis")

    _, feature_importance, _ = load_visualization_data()

    # Feature importance bar chart
    st.subheader("🌟 Ridge Regression Feature Importance (Scaled Features)")

    fig_imp = px.bar(feature_importance.head(15), x='importance', y='feature',
                    orientation='h', title="Top 15 Most Important Features")
    fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_imp)

    # Feature importance table
    st.subheader("📋 Feature Importance Details")
    st.dataframe(feature_importance.style.background_gradient(cmap='viridis'))

    # Cumulative importance
    feature_importance['cumulative'] = feature_importance['importance'].cumsum()
    fig_cum = px.line(feature_importance, x=range(1, len(feature_importance)+1),
                     y='cumulative', title="Cumulative Feature Importance")
    fig_cum.add_hline(y=0.95, line_dash="dash", line_color="red",
                     annotation_text="95% of total importance")
    st.plotly_chart(fig_cum)

# Footer
st.markdown("---")
st.markdown("**Demand Forecasting System** - Machine Learning Assignment")
st.markdown("Built with Streamlit, scikit-learn, and Plotly")