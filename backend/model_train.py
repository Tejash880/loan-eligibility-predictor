import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import os

def train_model():
    print("Loading dataset...")
    # Load dataset
    data_path = os.path.join("..", "dataset", "loan_data.csv")
    if not os.path.exists(data_path):
        print(f"Error: Dataset not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Fill missing values
    df['Gender'] = df['Gender'].fillna(df['Gender'].mode()[0])
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].replace('3+', '3').fillna(df['Dependents'].mode()[0]).astype(int)
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    
    # Feature Engineering based on README
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['Income_Loan_Ratio'] = df['Total_Income'] / df['LoanAmount']
    df['Loan_Amount_Term_Ratio'] = df['LoanAmount'] / df['Loan_Amount_Term']
    
    # Encode categorical variables
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    df['Self_Employed'] = df['Self_Employed'].map({'Yes': 1, 'No': 0})
    df['Property_Area'] = df['Property_Area'].map({'Urban': 2, 'Semiurban': 1, 'Rural': 0})
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    
    # Select features
    features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                'Credit_History', 'Property_Area', 'Total_Income', 'Income_Loan_Ratio', 
                'Loan_Amount_Term_Ratio']
    
    # Ensure all columns are numeric, filling any remaining NaNs
    X = df[features].fillna(0)
    y = df['Loan_Status'].fillna(0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model trained successfully! Test Accuracy: {accuracy*100:.2f}%")
    
    print("Saving model to loan_model.pkl...")
    with open('loan_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("Done!")

if __name__ == "__main__":
    train_model()
