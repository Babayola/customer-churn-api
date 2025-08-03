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
            'Contract': request.form['Contract'],
            'InternetService': request.form['InternetService'],
            'OnlineSecurity': request.form['OnlineSecurity'],
            'PhoneService': request.form['PhoneService'],
            'TechSupport': request.form['TechSupport'],
            'PaperlessBilling': request.form['PaperlessBilling'],
            'PaymentMethod': request.form['PaymentMethod'],
            'MultipleLines': request.form['MultipleLines'],
            'StreamingMovies': request.form['StreamingMovies']
        }

        input_df = pd.DataFrame([input_data])
        prediction = pipeline.predict(input_df)[0]

        return render_template('index.html', prediction_text=f'Churn Prediction: {prediction}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

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
