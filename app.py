import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('rf_model.pkl')

# App Title
st.title("Global Food Wastage Prediction")
st.write("Predict Economic Loss based on food wastage factors")

# Input Features
total_waste = st.number_input("Enter Total Waste (Tons)", min_value=500.0, max_value=50000.0, step=500.0)
avg_waste_per_capita = st.number_input("Enter Avg Waste per Capita (Kg)", min_value=20.0, max_value=200.0, step=5.0)
population = st.number_input("Enter Population (Million)", min_value=10.0, max_value=1500.0, step=10.0)
household_waste = st.number_input("Enter Household Waste (%)", min_value=30.0, max_value=70.0, step=1.0)

country = st.selectbox("Select Country", [
    "Australia", "Brazil", "Canada", "China", "France", "Germany", "India",
    "Indonesia", "Italy", "Japan", "Mexico", "Russia", "Saudi Arabia",
    "South Africa", "South Korea", "Spain", "Turkey", "UK", "USA"
])

food_category = st.selectbox("Select Food Category", [
    "Beverages", "Dairy Products", "Frozen Food", "Fruits & Vegetables",
    "Grains & Cereals", "Meat & Seafood", "Prepared Food"
])

year = st.selectbox("Select Year", [2019, 2020, 2021, 2022, 2023, 2024])

# Prepare input data
data = {
    "Total Waste (Tons)": [total_waste],
    "Avg Waste per Capita (Kg)": [avg_waste_per_capita],
    "Population (Million)": [population],
    "Household Waste (%)": [household_waste],
    "Country": [country],
    "Food Category": [food_category],
    "Year": [year]
}

input_df = pd.DataFrame(data)

# One-hot encoding for categorical columns to match trained model features
input_df = pd.get_dummies(input_df, columns=["Country", "Food Category", "Year"], drop_first=False)

# Aligning with trained model features (filling missing columns with 0)
required_features = model.feature_names_in_
for col in required_features:
    if col not in input_df.columns:
        input_df[col] = 0  # Add missing columns as zeros

input_df = input_df[required_features]

# st.write("### Input Data for Prediction:")
# st.dataframe(input_df)

# Prediction
if st.button("Predict Economic Loss"):
    try:
        prediction = model.predict(input_df)
        st.write(f"### Predicted Economic Loss: ${prediction[0]:,.2f} Million")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
