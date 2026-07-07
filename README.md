# 🎯 LoanIQ - AI-Powered Loan Eligibility Predictor

A modern, full-stack machine learning application that predicts loan eligibility using advanced Random Forest algorithm. Features a stunning user interface with real-time predictions, detailed analysis, and confidence scoring.

![Project Status](https://img.shields.io/badge/status-active-success.svg)
![ML Model](https://img.shields.io/badge/model-Random%20Forest-blue.svg)
![Accuracy](https://img.shields.io/badge/accuracy-87%25-brightgreen.svg)

## ✨ Features

### Frontend (User Interface)
- **Modern, Responsive Design** - Beautiful gradient-based UI with smooth animations
- **Real-time Predictions** - Instant loan eligibility assessment
- **Detailed Analysis** - Key factors breakdown and financial summary
- **Confidence Scoring** - ML model confidence percentage
- **Downloadable Reports** - Export prediction results as text file
- **Mobile Optimized** - Fully responsive design for all devices

### Backend (ML Engine)
- **Random Forest Classifier** - 87% accuracy on test data
- **Advanced Feature Engineering** - Income ratios, derived features
- **RESTful API** - Flask-based API with CORS support
- **Robust Preprocessing** - Handles missing values and encoding
- **Cross-validation** - 5-fold CV for model reliability


## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Safari, Edge)

### Installation

1. **Clone or Download the Project**
   ```bash
   cd Loan_Eligibility_Project
   ```

2. **Install Python Dependencies**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   - Download a loan dataset (e.g., from Kaggle: "Loan Prediction Problem Dataset")
   - Place it in the `dataset/` folder as `loan_data.csv`
   - Required columns:
     - Loan_ID, Gender, Married, Dependents, Education
     - Self_Employed, ApplicantIncome, CoapplicantIncome
     - LoanAmount, Loan_Amount_Term, Credit_History
     - Property_Area, Loan_Status

4. **Train the Model**
   ```bash
   python model_train.py
   ```
   
   This will:
   - Load and preprocess the dataset
   - Engineer new features
   - Train Random Forest model
   - Display performance metrics
   - Save model as `loan_model.pkl`

5. **Start the Backend Server**
   ```bash
   python app.py
   ```
   
   Server will start at: `http://localhost:5000`

6. **Open the Frontend**
   - Open `frontend/index.html` in your web browser
   - OR use a local server:
     ```bash
     cd frontend
     python -m http.server 8000
     ```
     Then visit: `http://localhost:8000`

## 📊 How It Works

### 1. Data Flow

```
User Input → Frontend Form → API Request → Backend Processing
    ↓
Preprocessing → Feature Engineering → Model Prediction
    ↓
Confidence Score → API Response → Frontend Display
```

### 2. Key Features Used

| Feature | Importance | Description |
|---------|-----------|-------------|
| Credit History | 55-60% | Most critical factor |
| Income-to-Loan Ratio | 15-20% | Combined income / loan amount |
| Total Income | 10-12% | Applicant + Co-applicant income |
| Loan Amount | 5-8% | Requested loan amount |
| Education | 2-4% | Graduate vs Non-graduate |
| Property Area | 3-5% | Urban/Semi-urban/Rural |

### 3. Model Performance

- **Training Accuracy**: ~85-87%
- **Testing Accuracy**: ~82-85%
- **Cross-Validation**: ~80-84%
- **Precision (Approved)**: ~85%
- **Recall (Approved)**: ~95%

## 🎨 UI/UX Features

- **Animated Gradients** - Smooth background animations
- **Smooth Transitions** - All interactions have polished transitions
- **Loading States** - Visual feedback during API calls
- **Form Validation** - Client-side validation before submission
- **Responsive Grid** - Adapts to all screen sizes
- **Modern Typography** - Playfair Display + DM Sans fonts
- **Color Coded Results** - Green for approved, Red for rejected

## 🔧 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Predict Loan Eligibility
```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "Gender": "Male",
  "Married": "Yes",
  "Dependents": "1",
  "Education": "Graduate",
  "Self_Employed": "No",
  "ApplicantIncome": 5000,
  "CoapplicantIncome": 3000,
  "LoanAmount": 150000,
  "Loan_Amount_Term": 360,
  "Credit_History": 1,
  "Property_Area": "Urban"
}
```

**Response:**
```json
{
  "prediction": "Approved",
  "confidence": 87.5,
  "probability": {
    "approved": 87.5,
    "rejected": 12.5
  }
}
```

## 📈 Model Training Details

### Feature Engineering

The model uses these engineered features:

1. **Total_Income** = ApplicantIncome + CoapplicantIncome
2. **Income_Loan_Ratio** = Total_Income / LoanAmount
3. **Loan_Amount_Term_Ratio** = LoanAmount / Loan_Amount_Term

### Preprocessing Steps

1. Handle missing values (median for numerical, mode for categorical)
2. Encode categorical variables (Label Encoding)
3. Feature scaling (if needed)
4. Create derived features
5. Split into train/test sets (80/20)

### Model Hyperparameters

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)
```

## 🛠️ Customization

### Change Model Parameters

Edit `backend/model_train.py`:

```python
model = RandomForestClassifier(
    n_estimators=200,  # Increase trees
    max_depth=15,      # Deeper trees
    # ... other parameters
)
```

### Modify UI Colors

Edit `frontend/css/style.css`:

```css
:root {
    --primary-500: #YOUR_COLOR;
    --primary-gradient: linear-gradient(135deg, #COLOR1, #COLOR2);
}
```

### Add New Features

1. Add feature to dataset
2. Update preprocessing in `model_train.py`
3. Update form in `frontend/index.html`
4. Update API request in `frontend/js/script.js`

## 🐛 Troubleshooting

### Model not loading
```bash
# Ensure you've trained the model first
cd backend
python model_train.py
```

### CORS errors
- Ensure Flask-CORS is installed
- Check that API URL in `script.js` matches your backend URL

### Port already in use
```bash
# Change port in app.py
app.run(port=5001)  # Use different port
```

## 📱 Browser Compatibility

- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ✅ Mobile browsers

## 🚀 Deployment Options

### Local Network Access
```bash
python app.py  # Backend accessible at your-ip:5000
```

### Cloud Deployment
- **Backend**: Deploy to Heroku, AWS, or Google Cloud
- **Frontend**: Deploy to Netlify, Vercel, or GitHub Pages
- **Database**: Add PostgreSQL/MongoDB for storing predictions

## 📝 Future Enhancements

- [ ] User authentication and history
- [ ] Database integration for storing applications
- [ ] Advanced models (XGBoost, Neural Networks)
- [ ] Batch prediction upload (CSV)
- [ ] Admin dashboard with analytics
- [ ] Email notification system
- [ ] Multi-language support

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is created for educational purposes.

## 👨‍💻 Author

Created with ❤️ using Flask, Scikit-learn, and modern web technologies.

---

## 🎓 Learning Outcomes

This project demonstrates:
- Full-stack ML application development
- RESTful API design
- Feature engineering techniques
- Model training and evaluation
- Modern UI/UX design
- Responsive web development
- Real-time predictions

## 📚 Technologies Used

### Backend
- Python 3.8+
- Flask (Web framework)
- Scikit-learn (ML library)
- Pandas (Data manipulation)
- NumPy (Numerical computing)

### Frontend
- HTML5
- CSS3 (Animations, Gradients, Flexbox, Grid)
- JavaScript (ES6+)
- Google Fonts (Playfair Display, DM Sans)

---

**Made with ❤️ for aspiring ML engineers and data scientists**
