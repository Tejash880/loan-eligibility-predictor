from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), 'frontend')
MODEL_PATH = os.path.join(BASE_DIR, 'loan_model.pkl')

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path='')
CORS(app)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health():
    model_loaded = os.path.exists(MODEL_PATH)
    return jsonify({
        "status": "healthy",
        "model_loaded": model_loaded
    })

@app.route('/predict', methods=['POST'])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet"}), 500
        
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
            
        data = request.json
        
        # Prepare data for prediction
        input_data = pd.DataFrame([{
            'Gender': 1 if data.get('Gender') == 'Male' else 0,
            'Married': 1 if data.get('Married') == 'Yes' else 0,
            'Dependents': int(data.get('Dependents', 0) if data.get('Dependents') != '3+' else 3),
            'Education': 1 if data.get('Education') == 'Graduate' else 0,
            'Self_Employed': 1 if data.get('Self_Employed') == 'Yes' else 0,
            'ApplicantIncome': float(data.get('ApplicantIncome', 0)),
            'CoapplicantIncome': float(data.get('CoapplicantIncome', 0)),
            'LoanAmount': float(data.get('LoanAmount', 0)),
            'Loan_Amount_Term': float(data.get('Loan_Amount_Term', 360)),
            'Credit_History': float(data.get('Credit_History', 1)),
            'Property_Area': 2 if data.get('Property_Area') == 'Urban' else (1 if data.get('Property_Area') == 'Semiurban' else 0)
        }])
        
        # Add engineered features
        input_data['Total_Income'] = input_data['ApplicantIncome'] + input_data['CoapplicantIncome']
        input_data['Income_Loan_Ratio'] = input_data['Total_Income'] / (input_data['LoanAmount'] if input_data['LoanAmount'][0] > 0 else 1)
        input_data['Loan_Amount_Term_Ratio'] = input_data['LoanAmount'] / (input_data['Loan_Amount_Term'] if input_data['Loan_Amount_Term'][0] > 0 else 1)
        
        features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                    'Credit_History', 'Property_Area', 'Total_Income', 'Income_Loan_Ratio', 
                    'Loan_Amount_Term_Ratio']
        
        X = input_data[features]
        
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        
        approved_prob = probabilities[1] * 100
        rejected_prob = probabilities[0] * 100
        
        return jsonify({
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "confidence": round(max(approved_prob, rejected_prob), 1),
            "probability": {
                "approved": round(approved_prob, 1),
                "rejected": round(rejected_prob, 1)
            }
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Default to 5000, but allow environment variable PORT for cloud deployments
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
