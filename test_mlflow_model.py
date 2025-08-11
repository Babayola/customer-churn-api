import mlflow
import pandas as pd

# Set MLflow tracking URI to your local registry
mlflow.set_tracking_uri("file:///C:/Users/User/mlruns")
model_name = "customer_churn_model"

# Load model from MLflow Registry (Production stage)
model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

# Create sample input (now includes customerID to match training columns)
sample_input = pd.DataFrame([{
    "customerID": "0001-ABCD",  # dummy ID
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.50
}])

# Make prediction
prediction = model.predict(sample_input)
print("Prediction:", prediction[0])
