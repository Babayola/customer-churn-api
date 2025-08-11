import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
from mlflow.tracking import MlflowClient

# MLflow tracking setup
mlflow.set_tracking_uri("file:///C:/Users/User/mlruns")
model_name = "customer_churn_model"

# Path to your saved pipeline
model_file = r"models/full_pipeline.joblib"
model = joblib.load(model_file)

# Create an example input DataFrame with customerID included
input_example = pd.DataFrame([{
    "customerID": "DUMMY-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
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
    "TotalCharges": 351.75
}])

# Log model with example
with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name=model_name
    )

# Promote latest version to Production
client = MlflowClient()
latest_version_info = client.get_latest_versions(model_name, stages=["None"])[0]
client.transition_model_version_stage(
    name=model_name,
    version=latest_version_info.version,
    stage="Production"
)

print(f"Model '{model_name}' version {latest_version_info.version} is now in PRODUCTION.")
