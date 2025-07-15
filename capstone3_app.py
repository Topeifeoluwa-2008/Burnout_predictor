# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Set up Streamlit page
st.set_page_config(page_title="Burnout Predictor", layout="wide")
st.title("🧠 NeuroWell Burnout Predictor")
st.write("Helping companies take proactive steps in employee well-being.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

Neuro_df = pd.read_csv('train.csv')

# Sidebar Navigation
section = st.sidebar.radio("Choose a section:", ["Overview", "EDA", "Predict Burnout"])

# --- Overview ---
if section == "Overview":
    st.markdown("""
    ### 💡 Overview
    This app explores employee burnout using machine learning models trained on workplace data.
    Use the EDA tab for insights and the Predict tab for burnout predictions based on employee profiles.
    """)

# --- Exploratory Data Analysis ---
elif section == "EDA":
    st.header("📊 Exploratory Data Analysis")
    
    st.subheader("Data Sample")
    st.dataframe(Neuro_df.head())

    st.subheader("Missing Values")
    st.write(Neuro_df.isnull().sum())

    st.subheader("Mental Fatigue Score Distribution")
    fig1 = plt.figure()
    sns.boxplot(x=Neuro_df["Mental Fatigue Score"])
    st.pyplot(fig1)

    st.subheader("Resource Allocation vs Burn Rate")
    fig2 = plt.figure()
    sns.scatterplot(x=Neuro_df["Resource Allocation"], y=Neuro_df["Burn Rate"])
    st.pyplot(fig2)

# --- Prediction Tool ---
elif section == "Predict Burnout":
    st.header("🔮 Burnout Prediction Tool")

    # Load model
    model = joblib.load("best_gradient_boosting_model.pkl")

    # Input form
    with st.form("predict_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        company = st.selectbox("Company Type", ["Product", "Service"])
        wfh = st.selectbox("WFH Setup Available", ["Yes", "No"])
        designation = st.slider("Designation Level", 0.0, 5.0, 2.0)
        allocation = st.slider("Resource Allocation", 0.0, 10.0, 5.0)
        fatigue = st.slider("Mental Fatigue Score", 0.0, 10.0, 5.0)
        tenure = st.slider("Tenure (Years)", 1, 30, 35)
        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame({
                "Gender": [1 if gender == "Female" else 0],
                "Company Type": [1 if company == "Service" else 0],
                "Designation": [designation],
                "Resource Allocation": [allocation],
                "Mental Fatigue Score": [fatigue],
                "Tenure (Years)": [tenure],
                "WFH Setup Available_No": [1 if wfh == "No" else 0],
                "WFH Setup Available_Yes": [1 if wfh == "Yes" else 0]
            })

            # Ensure column alignment
            for col in model.feature_names_in_:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[model.feature_names_in_]

            prediction = model.predict(input_data)[0]
            st.success(f"Predicted Burn Rate: {prediction:.2f}")

# Footer
st.markdown("---")
st.caption("Made with ❤️ by Temitope at NeuroWell Analytics")