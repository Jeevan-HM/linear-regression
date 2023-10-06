import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib  # Import joblib for model serialization

# Load the dataset from the CSV file
data = pd.read_csv("dataset.csv")

# Assuming your CSV file has columns named 'House Area', 'Pixel Count', and 'Division Result'
X = data[["House Area", "Pixel Count"]]
y = data["Division Result"]

# Split the data into a training set and a test set (e.g., 80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize the Min-Max scaler
scaler = MinMaxScaler()

# Fit the scaler to your training data and transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the scaled training data
model.fit(X_train_scaled, y_train)

# Save the trained model to a file using joblib
joblib.dump(model, "linear_regression_model.joblib")
joblib.dump(scaler, "scaler.joblib")

# Later, you can load the model from the file
loaded_model = joblib.load("linear_regression_model.joblib")

# Now you can use the loaded model to make predictions on custom values
custom_values = [[2470, 12000]]  # Replace with your custom input values
custom_values_scaled = scaler.transform(custom_values)  # Scale the custom input values
prediction = loaded_model.predict(custom_values_scaled)
print(f"Prediction for custom values: {prediction}")
