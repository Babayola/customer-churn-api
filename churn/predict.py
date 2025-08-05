import pandas as pd
import joblib
import os

# Point to the correct model version
MODEL_VERSION = "v1"
MODEL_PATH = os.path.join('models', MODEL_VERSION, 'full_pipeline.joblib')

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

# Define expected features
FEATURES = [
    'Contract', 'InternetService', 'OnlineSecurity', 'PhoneService',
    'TechSupport', 'PaperlessBilling', 'PaymentMethod',
    'MultipleLines', 'StreamingMovies'
]

# Prediction function
def predict_from_input_dict(input_data: dict):
    input_df = pd.DataFrame([input_data])
    prediction = pipeline.predict(input_df)[0]
    return prediction
