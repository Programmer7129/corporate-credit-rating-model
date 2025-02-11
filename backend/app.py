from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "best_rf_model.pkl")
model = joblib.load(model_path)

# Load the scaler (ensure it was saved with the model)
scaler_path = os.path.join(os.path.dirname(__file__), "scaler.pkl")
scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else StandardScaler()

# Define expected feature names based on the training data
expected_features = [
    "cik", "binary_rating", "sic_code", "current_ratio",
    "long_term_debt_capital", "debt_equity_ratio", "gross_margin",
    "operating_margin", "ebit_margin", "ebitda_margin",
    "pre_tax_profit_margin", "net_profit_margin", "asset_turnover",
    "roe", "return_on_tangible_equity",
    "roa", "roi", "operating_cash_flow_per_share",
    "free_cash_flow_per_share"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json  # Get user input as JSON
        if not data:
            return jsonify({"error": "Invalid input: No JSON received"}), 400

        input_features = pd.DataFrame([data])  # Convert to DataFrame

        # Validate input features
        missing_features = [feature for feature in expected_features if feature not in input_features.columns]
        if missing_features:
            return jsonify({"error": f"Missing required fields: {missing_features}"}), 400

        # Ensure the feature order matches training
        input_features = input_features[expected_features]

        # Scale input features
        input_features_scaled = scaler.transform(input_features)
        print("Python Scaled Input Features:", input_features_scaled.tolist())

        # Make prediction
        prediction = model.predict(input_features_scaled)

        # Convert numeric prediction to credit rating label
        rating_map = {0: "AAA/AA", 1: "A", 2: "BBB", 3: "BB", 4: "B or below"}
        predicted_rating = rating_map.get(prediction[0], "Unknown")

        response = {"prediction": predicted_rating}
        print("API Response:", response)  # Debugging print statement

        return jsonify(response)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
