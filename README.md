#### 💰 Loan Approval Prediction System


#### 📘 Overview

This project predicts whether a loan application will be approved or rejected based on applicant details such as income, credit history, and loan amount.
It uses a Machine Learning model (Random Forest Classifier) trained on historical loan data, integrated into a Gradio web app for interactive predictions.

#### 🧠 Project Workflow

Data Loading – Import and inspect dataset (loan_data.csv)

Data Cleaning – Handle missing values and duplicates

Exploratory Data Analysis (EDA) – Visualize distributions and correlations

#### Feature Engineering –

Log transformation on skewed columns (person_income)

Categorical encoding using pd.get_dummies()

Model Building – Train a Random Forest Classifier

#### Model Evaluation – Evaluate using:

Accuracy

ROC-AUC Score

Classification Report

ROC Curve visualization

Deployment – Deploy the final trained model via a Gradio interface in app.py

#### 🧾 Dataset Information

File: loan_data.csv
Each record represents a loan applicant with the following features:

#### Feature	Description

person_age	Age of applicant
person_income	Annual income (log transformed)
person_home_ownership	Type of home ownership
loan_intent	Purpose of the loan (education, business, etc.)
loan_grade	Credit grade
loan_amnt	Loan amount requested
loan_int_rate	Interest rate on the loan
loan_percent_income	Loan amount as a % of income
cb_person_default_on_file	Previous default status
cb_person_cred_hist_length	Length of credit history
loan_status	Target variable – 1 (Approved) / 0 (Rejected)


#### ⚙️ Model Details

Algorithm Used: Random Forest Classifier

Hyperparameters:

RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)


#### Performance Metrics:

Accuracy: ~0.9 (approx.)

ROC-AUC Score: High (>0.85)

Robust to class imbalance and feature correlation

#### 📊 Exploratory Data Analysis

The notebook (Loan.ipynb) includes:

Distribution Plots for numeric features (using seaborn.histplot)

Correlation Heatmap for understanding feature relationships

Skewness Correction using log transformation (np.log1p)

#### 🚀 Deployment (Gradio App)

The model is integrated into a Gradio-based web interface via app.py.
Users can input feature values through a friendly UI and instantly get predictions.


#### 🧩 Folder Structure
Loan_Approval_Prediction/
│
├── Loan.ipynb              # Jupyter Notebook (data analysis + model training)
├── app.py                  # Gradio web app for deployment
├── loan_data.csv           # Dataset used for training
├── loan_model.pkl          # Trained Random Forest model
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

#### 🧰 Installation & Usage
1️⃣ Clone the Repository
git clone https://github.com/<your-username>/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction

#### 2️⃣ Install Dependencies
pip install -r requirements.txt

#### 🧮 Requirements
pandas
numpy
matplotlib
seaborn
scikit-learn
gradio

#### 🌟 Future Improvements

Add hyperparameter tuning (GridSearchCV) for model optimization

Implement cross-validation for better generalization      

But remind it that using gridsearch and all would take mus=ch more time...

###### APP LINK

https://huggingface.co/spaces/nandha-01/LoanApprovalPrediction


