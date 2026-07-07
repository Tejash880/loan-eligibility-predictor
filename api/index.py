from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model once at cold start
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'backend', 'loan_model.pkl')
model = None

def get_model():
    global model
    if model is None:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    return model

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": os.path.exists(MODEL_PATH)
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        clf = get_model()
        data = request.json

        # Parse inputs
        gender = 1 if data.get('Gender') == 'Male' else 0
        married = 1 if data.get('Married') == 'Yes' else 0
        dep = data.get('Dependents', '0')
        dependents = 3 if dep == '3+' else int(dep)
        education = 1 if data.get('Education') == 'Graduate' else 0
        self_employed = 1 if data.get('Self_Employed') == 'Yes' else 0
        applicant_income = float(data.get('ApplicantIncome', 0))
        coapplicant_income = float(data.get('CoapplicantIncome', 0))
        loan_amount = float(data.get('LoanAmount', 1))
        loan_term = float(data.get('Loan_Amount_Term', 360))
        credit_history = float(data.get('Credit_History', 1))
        prop = data.get('Property_Area', 'Urban')
        property_area = 2 if prop == 'Urban' else (1 if prop == 'Semiurban' else 0)

        # Engineered features
        total_income = applicant_income + coapplicant_income
        income_loan_ratio = total_income / max(loan_amount, 1)
        loan_term_ratio = loan_amount / max(loan_term, 1)

        features = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_history, property_area, total_income, income_loan_ratio,
            loan_term_ratio
        ]])

        prediction = clf.predict(features)[0]
        probabilities = clf.predict_proba(features)[0]

        approved_prob = round(probabilities[1] * 100, 1)
        rejected_prob = round(probabilities[0] * 100, 1)

        return jsonify({
            "prediction": "Approved" if prediction == 1 else "Rejected",
            "confidence": max(approved_prob, rejected_prob),
            "probability": {
                "approved": approved_prob,
                "rejected": rejected_prob
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
