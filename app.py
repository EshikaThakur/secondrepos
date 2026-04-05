from flask import Flask, request, jsonify
import numpy as np

from fraud_detection import predict_transaction

app = Flask(__name__)

@app.route("/")
def home():
    return "Fraud Detection API is running 🚀"
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    amount = float(data['amount'])
    location_risk = float(data['location_risk'])
    txn_frequency = float(data['txn_frequency'])
    hour = float(data['hour'])

    result = predict_transaction(amount, location_risk, txn_frequency, hour)

    return jsonify({
        "label": result["label"],
        "fraud_probability": result["fraud_probability"]
    })

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
    