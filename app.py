import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


# Streamlit app setup
st.set_page_config(
    page_title="🏠 House Price Prediction",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load preprocessed model and scaler
@st.cache_data
def load_model():
    with open('model/model.pkl', 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

model, scaler = load_model()



st.title("🏠 House Price Prediction")
st.write("This interactive application predicts house prices based on user inputs. Explore the dataset and see the model's performance.")

# Sidebar for user input
st.sidebar.header("Enter House Features")

area = st.sidebar.number_input("Area (sq. ft):", min_value=500, max_value=10000, value=1500)
bedrooms = st.sidebar.slider("Number of Bedrooms:", min_value=1, max_value=10, value=3)
bathrooms = st.sidebar.slider("Number of Bathrooms:", min_value=1, max_value=5, value=2)
stories = st.sidebar.slider("Number of Stories:", min_value=1, max_value=4, value=2)
parking = st.sidebar.slider("Parking Spaces:", min_value=0, max_value=5, value=1)
mainroad = st.sidebar.selectbox("Is it on a Main Road?", ("Yes", "No"))
guestroom = st.sidebar.selectbox("Does it have a Guest Room?", ("Yes", "No"))
basement = st.sidebar.selectbox("Does it have a Basement?", ("Yes", "No"))
hotwaterheating = st.sidebar.selectbox("Hot Water Heating Available?", ("Yes", "No"))
airconditioning = st.sidebar.selectbox("Air Conditioning Available?", ("Yes", "No"))
prefarea = st.sidebar.selectbox("Preferred Area?", ("Yes", "No"))
furnishingstatus = st.sidebar.selectbox("Furnishing Status:", ("Furnished", "Semi-Furnished", "Unfurnished"))

# Validate inputs
if area < 500 or area > 10000:
    st.sidebar.error("Area must be between 500 and 10,000 sq. ft.")

# Prepare input data
input_data = {
    'area': area,
    'bedrooms': bedrooms,
    'bathrooms': bathrooms,
    'stories': stories,
    'parking': parking,
    'mainroad_yes': 1 if mainroad == "Yes" else 0,
    'guestroom_yes': 1 if guestroom == "Yes" else 0,
    'basement_yes': 1 if basement == "Yes" else 0,
    'hotwaterheating_yes': 1 if hotwaterheating == "Yes" else 0,
    'airconditioning_yes': 1 if airconditioning == "Yes" else 0,
    'prefarea_yes': 1 if prefarea == "Yes" else 0,
    'furnishingstatus_semi-furnished': 1 if furnishingstatus == "Semi-Furnished" else 0,
    'furnishingstatus_unfurnished': 1 if furnishingstatus == "Unfurnished" else 0
}

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Normalize numerical features
numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])

# Make predictions
if st.button("Predict Price"):
    with st.spinner("Calculating..."):
        prediction = model.predict(input_df)[0]
        st.success(f"The predicted house price is: ₹{prediction * 1e7:.2f}")

        # Feature importance
        st.subheader("Feature Importance")
        feature_importance = model.feature_importances_
        feature_names = input_df.columns

        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(feature_names, feature_importance, color='skyblue')
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance")
        st.pyplot(plt)

# Dataset Summary
st.sidebar.subheader("Explore Dataset")
if st.sidebar.button("Show Dataset Summary"):
    with st.spinner("Loading dataset..."):
        data_path = 'data/Housing.csv'
        dataset = pd.read_csv(data_path)
        st.write("### Dataset Preview")
        st.dataframe(dataset.head())

        st.write("### Summary Statistics")
        st.write(dataset.describe())

# Footer
st.markdown("---")
st.markdown("Developed by **Vidish Bijalwan**. Data Source: **https://www.kaggle.com/datasets/yasserh/housing-prices-dataset**")
