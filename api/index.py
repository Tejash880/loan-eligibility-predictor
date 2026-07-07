import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)
CORS(app)

# ── Global model cache (trained once per cold start) ────────────────────────
_model = None

def _get_model():
    global _model
    if _model is not None:
        return _model

    # Build the path to the dataset relative to this file
    BASE  = os.path.dirname(os.path.abspath(__file__))
    ROOT  = os.path.dirname(BASE)
    CSV   = os.path.join(ROOT, "dataset", "loan_data.csv")

    if not os.path.exists(CSV):
        raise FileNotFoundError(f"Dataset not found at: {CSV}")

    df = pd.read_csv(CSV)

    # ── Fill missing values ─────────────────────────────────────────────────
    df["Gender"]           = df["Gender"].fillna(df["Gender"].mode()[0])
    df["Married"]          = df["Married"].fillna(df["Married"].mode()[0])
    df["Dependents"]       = (df["Dependents"]
                               .replace("3+", "3")
                               .fillna("0")
                               .astype(int))
    df["Self_Employed"]    = df["Self_Employed"].fillna(df["Self_Employed"].mode()[0])
    df["LoanAmount"]       = df["LoanAmount"].fillna(df["LoanAmount"].median())
    df["Loan_Amount_Term"] = df["Loan_Amount_Term"].fillna(df["Loan_Amount_Term"].mode()[0])
    df["Credit_History"]   = df["Credit_History"].fillna(df["Credit_History"].mode()[0])

    # ── Encode categoricals ─────────────────────────────────────────────────
    df["Gender"]        = df["Gender"].map({"Male": 1, "Female": 0})
    df["Married"]       = df["Married"].map({"Yes": 1, "No": 0})
    df["Education"]     = df["Education"].map({"Graduate": 1, "Not Graduate": 0})
    df["Self_Employed"] = df["Self_Employed"].map({"Yes": 1, "No": 0})
    df["Property_Area"] = df["Property_Area"].map({"Urban": 2, "Semiurban": 1, "Rural": 0})
    df["Loan_Status"]   = df["Loan_Status"].map({"Y": 1, "N": 0})

    # ── Feature engineering ────────────────────────────────────────────────
    df["Total_Income"]          = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_Loan_Ratio"]     = df["Total_Income"] / df["LoanAmount"].replace(0, 1)
    df["Loan_Amount_Term_Ratio"]= df["LoanAmount"]   / df["Loan_Amount_Term"].replace(0, 1)

    FEATURES = [
        "Gender", "Married", "Dependents", "Education", "Self_Employed",
        "ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
        "Credit_History", "Property_Area",
        "Total_Income", "Income_Loan_Ratio", "Loan_Amount_Term_Ratio",
    ]

    X = df[FEATURES].fillna(0).astype(float)
    y = df["Loan_Status"].fillna(0).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    acc = clf.score(X_test, y_test)
    print(f"[LoanIQ] Model trained — test accuracy: {acc*100:.2f}%")

    _model = clf
    return _model


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        clf  = _get_model()
        data = request.get_json(force=True)

        # ── Parse & encode inputs ──────────────────────────────────────────
        gender          = 1 if data.get("Gender")       == "Male"      else 0
        married         = 1 if data.get("Married")      == "Yes"       else 0
        dep_raw         = str(data.get("Dependents", "0"))
        dependents      = 3 if dep_raw == "3+" else int(dep_raw)
        education       = 1 if data.get("Education")    == "Graduate"  else 0
        self_employed   = 1 if data.get("Self_Employed")== "Yes"       else 0
        prop            = data.get("Property_Area", "Urban")
        property_area   = 2 if prop == "Urban" else (1 if prop == "Semiurban" else 0)

        applicant_income    = float(data.get("ApplicantIncome",    0))
        coapplicant_income  = float(data.get("CoapplicantIncome",  0))
        loan_amount         = float(data.get("LoanAmount",         1))
        loan_term           = float(data.get("Loan_Amount_Term", 360))
        credit_history      = float(data.get("Credit_History",     1))

        # ── Engineered features ────────────────────────────────────────────
        total_income      = applicant_income + coapplicant_income
        income_loan_ratio = total_income     / max(loan_amount, 1)
        loan_term_ratio   = loan_amount      / max(loan_term,   1)

        X = np.array([[
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_history, property_area,
            total_income, income_loan_ratio, loan_term_ratio,
        ]])

        pred  = clf.predict(X)[0]
        proba = clf.predict_proba(X)[0]

        approved_pct = round(float(proba[1]) * 100, 1)
        rejected_pct = round(float(proba[0]) * 100, 1)

        return jsonify({
            "prediction": "Approved" if pred == 1 else "Rejected",
            "confidence": max(approved_pct, rejected_pct),
            "probability": {
                "approved": approved_pct,
                "rejected": rejected_pct,
            },
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── Local dev entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
