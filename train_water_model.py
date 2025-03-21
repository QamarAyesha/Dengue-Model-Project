# train_water_model.py
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Load dataset
data = pd.read_csv("data/water_data.csv")

# Features and target variable
X = data[["Stagnant_Water_Coverage"]]
y = data["Water_Risk_Score"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Save the model
joblib.dump(model, "models/water_model.pkl")
print("Stagnant water model saved to 'models/water_model.pkl'")
