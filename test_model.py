import joblib

# Load the trained model from the saved file
loaded_model = joblib.load("linear_regression_model.joblib")

# Custom input values for prediction
custom_values = [[1024, 11651]]  # Replace with your custom input values

# Initialize the Min-Max scaler and scale the custom input values
scaler = joblib.load("scaler.joblib")  # Load the scaler used during training
custom_values_scaled = scaler.transform(custom_values)

# Use the loaded model to make predictions on the scaled custom input values
prediction = loaded_model.predict(custom_values_scaled)

print(f"Prediction for custom values: {prediction}")
