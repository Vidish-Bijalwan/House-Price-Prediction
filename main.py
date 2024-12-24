import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load dataset
file_path = 'data/Housing.csv'
housing_data = pd.read_csv(file_path)

# Preprocess dataset
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                       'airconditioning', 'prefarea', 'furnishingstatus']
housing_data_encoded = pd.get_dummies(housing_data, columns=categorical_columns, drop_first=True)

scaler = MinMaxScaler()
numerical_columns = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']  # Exclude 'price' from scaling
housing_data_encoded[numerical_columns] = scaler.fit_transform(housing_data_encoded[numerical_columns])

# Split data
X = housing_data_encoded.drop('price', axis=1)
y = housing_data_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save model and scaler
with open('model/model.pkl', 'wb') as file:
    pickle.dump((model, scaler), file)

print("Model training and saving completed.")
