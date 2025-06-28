from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load trained model
model = joblib.load("house_price_model.pkl")

# Define the expected feature columns (order matters)
expected_columns = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms',
    'total_bedrooms', 'population', 'households', 'median_income',
    '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Convert input into a DataFrame
        df_input = pd.DataFrame([data])
        
        # Ensure all expected dummy columns are present
        for col in expected_columns:
            if col not in df_input:
                df_input[col] = 0  # Fill missing dummy categories with 0

        # Reorder columns to match training set
        df_input = df_input[expected_columns]

        # Make prediction
        prediction = model.predict(df_input)[0]
        return jsonify({"predicted_median_house_value": round(prediction, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    port=int(os.environ.get("PORT",5000))
    app.run(debug=False, host="0.0.0.0",port=port)
