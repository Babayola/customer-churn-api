from flask import Flask, request, render_template
import mlflow
import mlflow.sklearn
import pandas as pd

app = Flask(__name__)

# MLflow tracking URI (local registry)
mlflow.set_tracking_uri("file:///C:/Users/User/mlruns")
model_name = "customer_churn_model"

# Load latest Production model from MLflow Registry
def load_latest_model():
    latest = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    return latest

model = load_latest_model()

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    if request.method == "POST":
        # Collect form inputs
        input_data = {
            "gender": request.form["gender"],
            "SeniorCitizen": int(request.form["SeniorCitizen"]),
            "Partner": request.form["Partner"],
            "Dependents": request.form["Dependents"],
            "tenure": int(request.form["tenure"]),
            "PhoneService": request.form["PhoneService"],
            "MultipleLines": request.form["MultipleLines"],
            "InternetService": request.form["InternetService"],
            "OnlineSecurity": request.form["OnlineSecurity"],
            "OnlineBackup": request.form["OnlineBackup"],
            "DeviceProtection": request.form["DeviceProtection"],
            "TechSupport": request.form["TechSupport"],
            "StreamingTV": request.form["StreamingTV"],
            "StreamingMovies": request.form["StreamingMovies"],
            "Contract": request.form["Contract"],
            "PaperlessBilling": request.form["PaperlessBilling"],
            "PaymentMethod": request.form["PaymentMethod"],
            "MonthlyCharges": float(request.form["MonthlyCharges"]),
            "TotalCharges": float(request.form["TotalCharges"])
        }

        # Add dummy customerID to match model's expected columns
        input_data["customerID"] = "DUMMY-0001"

        df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(df)[0]

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_params(input_data)
            mlflow.log_metric("prediction", int(prediction == "Yes"))

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
