# predict_risk.py
import joblib
import pandas as pd

# Load the submodels
weather_model = joblib.load("models/weather_model.pkl")
water_model = joblib.load("models/water_model.pkl")
cases_model = joblib.load("models/cases_model.pkl")

# Example input data
input_data = {
    "Rainfall": 120.5,  # Weather data
    "Temperature": 28.3,
    "Humidity": 85.0,
    "Vegetation_Index": 0.65,
    "Stagnant_Water_Coverage": 0.3,  # Stagnant water data
    "Historical_Cases": 50  # Reported cases data
}

# Predict risk scores using submodels
weather_risk = weather_model.predict(pd.DataFrame([[
    input_data["Rainfall"],
    input_data["Temperature"],
    input_data["Humidity"],
    input_data["Vegetation_Index"]
]]).T)[0]

water_risk = water_model.predict(pd.DataFrame([[
    input_data["Stagnant_Water_Coverage"]
]]).T)[0]

cases_risk = cases_model.predict(pd.DataFrame([[
    input_data["Historical_Cases"]
]]).T)[0]

# Combine risk scores (weighted average)
weights = {
    "weather": 0.4,
    "water": 0.3,
    "cases": 0.3
}

final_risk_score = (
    weights["weather"] * weather_risk +
    weights["water"] * water_risk +
    weights["cases"] * cases_risk
)

# Display results
print("Weather Risk Score:", weather_risk)
print("Stagnant Water Risk Score:", water_risk)
print("Reported Cases Risk Score:", cases_risk)
print("Final Risk Score:", final_risk_score)
