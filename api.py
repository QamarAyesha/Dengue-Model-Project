# api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/dengue_risk_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input data from the request
    data = request.json
    input_data = pd.DataFrame([data])
    
    # Make predictions
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    # Return the result
    return jsonify({
        "prediction": int(prediction[0]),
        "probability": float(prediction_proba[0][1])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
