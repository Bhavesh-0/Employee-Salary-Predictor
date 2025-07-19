import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt # Although not directly used for plotting, often useful for charts
import time # Although not directly used for dynamic timing, often useful

# Sidebar navigation
st.sidebar.title("🔀 Navigation")
page = st.sidebar.radio("Go to", ["🏠 Predict Salary", "📊 Model Performance", "ℹ️ About"])

# Load model files
model_options = {
    "Linear Regression": "linearmodel.pkl",
    "Random Forest": "randomforest_model.pkl",
    "Gradient Boosting": "gradientboosting_model.pkl"
}

# Common: Load and preprocess the data (needed for main prediction page)
# Note: For the 'Model Performance' page, we are now using static pre-calculated values,
# so the train-test split and dynamic evaluation are no longer performed here.
try:
    data = pd.read_excel("Employees.xlsx")
    X = data[["Years", "Job Rate"]]
    Y = data["Annual Salary"]
except FileNotFoundError:
    st.error("❌ Employees.xlsx not found. Please ensure it's in the same directory as app.py.")
    st.stop()


# 🏠 Main Page: Salary Prediction
if page == "🏠 Predict Salary":
    st.title("💼 Salary Prediction App")
    st.markdown("---")
    st.write("🔍 Select a model, enter employee details, and get the salary prediction.")

    selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
    model_path = model_options[selected_model_name]

    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        st.error(f"❌ Model file not found: {model_path}. Please ensure all .pkl model files are in the same directory.")
        st.stop()

    # Input fields
    years = st.number_input("Enter Years at Company", min_value=0, value=3, step=1)
    job_rate = st.number_input("Enter Job Performance Rating (0 to 5)", min_value=0.0, max_value=5.0, value=3.5, step=0.5)

    if st.button("Predict Salary"):
        st.balloons()
        # Fix for UserWarning: X does not have valid feature names
        # Create a DataFrame with the correct feature names that the model was fitted with
        features_df = pd.DataFrame([[years, job_rate]], columns=["Years", "Job Rate"])
        prediction = model.predict(features_df)[0]
        st.success(f"💰 Predicted Annual Salary: **₹ {prediction:,.2f}**")

# 📊 Model Performance Page (Modified to show static pre-calculated values with better UI)
elif page == "📊 Model Performance":
    st.title("📈 Model Performance Comparison (Pre-calculated)")
    st.write("Below are the performance metrics calculated during model development on the test set. Lower MAE and RMSE are better, while higher R² Score is better.")

    # Create a DataFrame with the provided pre-calculated values
    # These values were provided by you based on an earlier run.
    results = {
        "Model": ["Linear Regression", "Gradient Boosting", "Random Forest"],
        "MAE": [8172.23, 8257.70, 8290.66],
        "RMSE": [9330.78, 9479.25, 9542.41],
        "R² Score": [-0.002, -0.034, -0.048]
    }
    results_df = pd.DataFrame(results).set_index("Model")

    # Sort the DataFrame for better visualization:
    # For MAE and RMSE, lower is better, so sort ascending
    # For R² Score, higher is better, so sort descending
    mae_sorted_df = results_df.sort_values(by="MAE", ascending=True)
    rmse_sorted_df = results_df.sort_values(by="RMSE", ascending=True)
    r2_sorted_df = results_df.sort_values(by="R² Score", ascending=False)

    st.subheader("Performance Metrics Table")
    st.dataframe(results_df.style.format("{:.2f}"))

    st.subheader("📊 Mean Absolute Error (MAE) Comparison")
    st.bar_chart(mae_sorted_df["MAE"], use_container_width=True)

    st.subheader("📊 Root Mean Squared Error (RMSE) Comparison")
    st.bar_chart(rmse_sorted_df["RMSE"], use_container_width=True)

    st.subheader("📊 R² Score Comparison")
    st.bar_chart(r2_sorted_df["R² Score"], use_container_width=True)

# ℹ️ About Page
elif page == "ℹ️ About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This salary prediction app uses multiple machine learning models to predict an employee's annual salary
    based on:
    - **Years at the company**
    - **Performance rating**

    Trained models included:
    - Linear Regression
    - Random Forest
    - Gradient Boosting

    Built using **Streamlit**, **scikit-learn**, and **joblib**.
    """)
    st.markdown("---") # Add a separator
    st.write("Built by **Bhavesh Bhati**") # Added credit