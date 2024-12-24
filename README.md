# 🏠 House Price Prediction 

This is an interactive web application that predicts house prices based on user inputs like area, number of bedrooms, bathrooms, and other house features. It uses a **Random Forest Regressor** model trained on a housing dataset and is deployed using **Streamlit** for an interactive user interface. The app also provides a feature importance visualization to help users understand which features affect the price prediction.



## Table of Contents:
1. [Features](#features)
2. [Requirements](#requirements)
3. [Setup Instructions](#setup-instructions)
4. [Folder Structure](#folder-structure)
5. [Model Training](#model-training)


## Features:
- **Price Prediction**: Users can input various house features, such as area, number of bedrooms, and bathrooms, and the app will predict the house price.
- **Feature Importance**: The model shows a bar chart indicating the importance of each feature in predicting the price.
- **Dataset Exploration**: Users can explore the dataset used for training, including a preview and summary statistics.
- **Interactive Interface**: Easy-to-use sidebar for entering house features and receiving predictions.

## Requirements:
Make sure you have **Python 3.7** or higher installed, along with the following libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`
- `matplotlib`
- `pickle`

To install the required dependencies, use the `requirements.txt` file provided in the project.

## Setup Instructions:

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

### 2. Create a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'

### 3. Install the required dependencies:

pip install -r requirements.txt

### 4.  Run the Streamlit app:
streamlit run app.py


##Folder Structure:
The project structure is as follows:

house-price-prediction/
│
├── data/
│   └── Housing.csv         # Housing dataset for training the model
├── model/
│   └── model.pkl           # Trained model and scaler
├── app.py                  # Streamlit app to make predictions
├── requirements.txt        # List of dependencies
└── README.md               # This file


## Model Training:

The model is trained using Random Forest Regressor. Here’s an overview of the steps followed to train the model:

Load and Preprocess Data: The dataset (data/Housing.csv) is preprocessed by encoding categorical features and scaling numerical features.
Train Model: A Random Forest Regressor is trained on the preprocessed data.
Save Model: The trained model and the scaler are saved as model/model.pkl, which is then loaded in the Streamlit app for predictions.

# Model training code example
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import pickle

# Load dataset
housing_data = pd.read_csv('data/Housing.csv')

# Preprocess dataset
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea', 'furnishingstatus']
housing_data_encoded = pd.get_dummies(housing_data, columns=categorical_columns, drop_first=True)

scaler = MinMaxScaler()
numerical_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'parking']
housing_data_encoded[numerical_columns] = scaler.fit_transform(housing_data_encoded[numerical_columns])

# Split data
X = housing_data_encoded.drop('price', axis=1)
y = housing_data_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save model
with open('model/model.pkl', 'wb') as file:
    pickle.dump((model, scaler), file)
