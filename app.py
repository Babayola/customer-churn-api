from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model pipeline
model_path = os.path.join('models', 'full_pipeline.joblib')
pipeline = joblib.load(model_path)

# Web UI route: show form
@app.route('/')
def index():
    return render_template('index.html')

# Web UI route: handle form submit

@app.route('/predict', methods=['POST'])
def predict_form():
    try:
        input_data = {
            'gender': request.form['gender'],
            'SeniorCitizen': int(request.form['SeniorCitizen']),
            'Partner': request.form['Partner'],
            'Dependents': request.form['Dependents'],
            'tenure': int(request.form['tenure']),
            'PhoneService': request.form['PhoneService'],
            'MultipleLines': request.form['MultipleLines'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'OnlineBackup': request.form['OnlineBackup'],
            'DeviceProtection': request.form['DeviceProtection'],
            'TechSupport': request.form['TechSupport'],
            'StreamingTV': request.form['StreamingTV'],
            'StreamingMovies': request.form['StreamingMovies'],
            'Contract': request.form['Contract'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MonthlyCharges': float(request.form['MonthlyCharges']),
            'TotalCharges': float(request.form['TotalCharges']),
            'customerID': request.form['customerID']
        }

        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}")

# API route: JSON request
@app.route('/predict-json', methods=['POST'])
def predict_json():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        prediction = pipeline.predict(input_df)
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
