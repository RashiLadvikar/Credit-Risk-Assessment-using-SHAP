# Credit-Risk-Assessment-using-SHAP


An end-to-end Machine Learning application for assessing the credit default risk of loan applicants. This project combines data preprocessing, class-imbalance handling, XGBoost classification, hyperparameter tuning, probability calibration, SHAP explainability, and deployment through FastAPI and Streamlit.

The application accepts applicant and loan-related information, estimates the probability of loan default, and provides a High Risk / Low Risk classification.

---

## 📌 Project Overview

Credit risk assessment is an important problem in financial services, where the objective is to identify applicants who may have a higher probability of defaulting on a loan.

This project demonstrates a complete machine learning workflow:

**Data → EDA & Validation → Preprocessing → Model Training → Hyperparameter Tuning → Probability Calibration → SHAP Explainability → Model Saving → API → Streamlit Application**

The project focuses on building not only a predictive model, but also an interpretable and deployable ML solution.

---

## 🚀 Key Features

- Exploratory Data Analysis (EDA)
- Data validation and cleaning
- Missing-value handling
- Duplicate-record removal
- Numerical and categorical feature preprocessing
- Class-imbalance handling using XGBoost
- Logistic Regression baseline
- XGBoost classification
- Hyperparameter tuning using RandomizedSearchCV
- Stratified cross-validation
- Model evaluation using Precision, Recall and F1-score
- Probability calibration using CalibratedClassifierCV
- SHAP-based model explainability
- Probability-based risk classification
- Model serialization using Joblib
- FastAPI REST API
- Interactive Streamlit web application

---

## 📊 Dataset

The project uses a credit risk dataset containing **32,581 loan application records**.

### Features

| Feature | Description |
|---|---|
| `person_age` | Applicant's age |
| `person_income` | Applicant's annual income |
| `person_home_ownership` | Home ownership status |
| `person_emp_length` | Employment length |
| `loan_intent` | Purpose of the loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Loan interest rate |
| `loan_percent_income` | Loan amount as a percentage of income |
| `cb_person_default_on_file` | Previous default indicator |
| `cb_person_cred_hist_length` | Length of credit history |
| `loan_status` | Target variable indicating loan default |

The target variable is imbalanced, so class imbalance was considered during model development.

---

## 🔄 Machine Learning Workflow

### 1. Data Validation & Cleaning

The dataset was checked for:

- Missing values
- Duplicate records
- Invalid age values
- Unrealistic employment lengths
- Invalid loan amounts
- Data consistency

Duplicate records were removed before model development.

### 2. Train-Test Split

The dataset was divided into training and testing sets using a stratified split to maintain the target-class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
3. Data Preprocessing
Numerical Features

Missing numerical values were handled using median imputation.

Categorical Features

Categorical features were:

Imputed
One-hot encoded using OneHotEncoder

The preprocessing steps were integrated into the ML pipeline to ensure consistent transformations during training and prediction.

🤖 Model Development
Logistic Regression

Logistic Regression was used as a baseline classification model.

It provides a simple benchmark for comparing the performance of the more advanced XGBoost model.

XGBoost

XGBoost was selected as the primary model because of its strong performance on structured/tabular data and its ability to capture nonlinear relationships.

Class imbalance was handled using the scale_pos_weight parameter.

⚙️ Hyperparameter Tuning

XGBoost hyperparameters were optimized using RandomizedSearchCV.

The search included parameters such as:

n_estimators
max_depth
learning_rate
subsample
colsample_bytree
min_child_weight
gamma

The tuning process used 5-fold cross-validation and optimized Average Precision, which is particularly useful for imbalanced classification problems.

📈 Model Performance

The final XGBoost evaluation on the test set achieved:

Metric	Score
Accuracy	92%
Precision	0.81
Recall	0.81
F1-Score	0.81

The evaluation considers Precision, Recall and F1-score in addition to accuracy because correctly identifying default-risk applicants is important in credit-risk applications.

🎯 Probability Calibration

For credit-risk applications, the predicted probability can be more useful than a simple binary prediction.

The trained XGBoost model was therefore calibrated using:

CalibratedClassifierCV(
    best_model,
    method="sigmoid",
    cv=5
)

The application uses:

predict_proba()

to obtain the estimated probability of default before applying the selected decision threshold.

🔍 Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) was used to improve the interpretability of the XGBoost model.

Global Explainability

SHAP summary plots were used to understand which features have the greatest influence on predictions across the dataset.

Local Explainability

SHAP waterfall plots were used to understand why a particular applicant received a specific prediction.

This makes the system more interpretable than a simple black-box classification model.

🏗️ System Architecture
                    ┌──────────────────────┐
                    │    Loan Applicant    │
                    │        Input         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Streamlit UI      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   ML Preprocessing   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Calibrated XGBoost   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Default Probability  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Risk Threshold     │
                    └──────────┬───────────┘
                               │
                    ┌──────────┴───────────┐
                    ▼                      ▼
               ┌─────────┐            ┌─────────┐
               │Low Risk │            │High Risk│
               └─────────┘            └─────────┘
🌐 FastAPI

The project includes a FastAPI backend that exposes a /predict endpoint.

The API accepts applicant and loan information and returns:

Default probability
Default prediction
Decision threshold
Risk classification

Example response:

{
    "default_probability": 0.18,
    "default_prediction": 0,
    "threshold": 0.50,
    "Result": "Low Risk"
}
🖥️ Streamlit Application

A Streamlit interface provides an interactive way to test the trained model.

Users can enter:

Applicant information
Employment details
Loan information
Credit history information

The application then displays:

Estimated default probability
Risk classification
Model decision
🛠️ Tech Stack
Programming
Python
Pandas
NumPy
Machine Learning
Scikit-learn
XGBoost
SHAP
Deployment
FastAPI
Uvicorn
Streamlit
Model Management
Joblib
Development Tools
Jupyter Notebook
PyCharm
Git
GitHub
📁 Project Structure
Credit-Risk-Assessment-using-SHAP/
│
├── Credit_Risk.ipynb
├── credit_risk_dataset.csv
├── credit_risk_model.pkl
├── best_threshold.pkl
│
├── app.py
├── main.py
│
├── static/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── requirements.txt
├── runtime.txt
├── render.yaml
├── .gitignore
└── README.md
💻 How to Run Locally
1. Clone the Repository
git clone <your-repository-url>
cd Credit-Risk-Assessment-using-SHAP
2. Create a Virtual Environment
python -m venv .venv
3. Activate the Environment

For Windows:

.venv\Scripts\activate
4. Install Dependencies
pip install -r requirements.txt
5. Run Streamlit
streamlit run app.py

The application will open in your browser.

⚡ Run FastAPI

Start the FastAPI backend using:

uvicorn main:app --reload

The API will be available at:

http://127.0.0.1:8000
🎓 Key Learning Outcomes

This project provided practical experience with:

End-to-end ML project development
Exploratory data analysis
Data cleaning and validation
Feature preprocessing
Imbalanced classification
XGBoost
Hyperparameter optimization
Cross-validation
Model evaluation
Probability calibration
Decision thresholding
SHAP explainability
Model serialization
REST API development
Streamlit application development
ML deployment concepts
Git and GitHub workflow
🔮 Future Scope

The project can be further enhanced into a more robust production-oriented credit-risk platform.

1. Advanced Threshold Optimization

Optimize the decision threshold based on business costs, expected financial loss, and risk appetite instead of relying only on model evaluation metrics.

2. Model Monitoring

Monitor:

Prediction distributions
Model performance
Data drift
Changes in applicant profiles
3. Automated Model Retraining

Build an automated pipeline that periodically retrains the model using newly available loan-performance data.

4. Fairness & Bias Analysis

Evaluate model performance across different applicant groups and identify potential sources of unintended bias.

5. Advanced Feature Engineering

Introduce additional financial indicators such as:

Debt-to-income ratio
Income stability
Credit utilization
Repayment history
Existing debt indicators
6. Interactive Explainability Dashboard

Integrate SHAP visualizations directly into the Streamlit application so users can understand the factors influencing individual predictions.

7. Database Integration

Store prediction requests and results in a database for:

Historical analysis
Auditing
Monitoring
Reporting
8. Cloud Deployment

Deploy the application on cloud infrastructure with proper environment management, monitoring, and scalability.

9. CI/CD Pipeline

Implement automated testing, model validation, and deployment using GitHub Actions.

10. Model Versioning & Experiment Tracking

Integrate tools such as MLflow for experiment tracking, model versioning, and reproducibility.

⚠️ Disclaimer

This project is intended for educational and demonstration purposes.

The predictions generated by this model should not be used as the sole basis for real-world lending or financial decisions without appropriate validation, regulatory review, monitoring, and human oversight.

👩‍💻 Author

Rashi Ladvikar

Data Analytics & Machine Learning Enthusiast
