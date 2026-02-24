# DiaPred — Diabetes Prediction (Streamlit App)

DiaPred is an interactive **Diabetes Prediction** web application built with **Streamlit**. It includes an end-to-end workflow: **EDA → missing value handling → model comparison → ensembling → prediction UI**.

## 🚀 Features

- **Interactive EDA Dashboard**
  - Univariate + multivariate analysis
  - Correlation heatmap & pairplots
  - Distribution plots (healthy vs diabetic)
  - Scatter plots across highly correlated feature pairs
  - HiPlot interactive exploration for selected columns

- **Missing Value Handling**
  - Converts `0` values to `NaN` for selected medical features
  - Imputes missing values using **median by target class (Outcome)**

- **Model Comparison**
  - Compare multiple ML models
  - Visual outputs:
    - Confusion Matrix
    - ROC Curve
    - Precision–Recall Curve
    - Metrics Bar Graph
  - Metrics reported:
    - Accuracy, Precision, Recall, F1 Score

- **Ensembling**
  - Majority voting ensemble evaluation using `ensembling.csv`
  - Confusion matrix + accuracy for ensemble output

- **Prediction**
  - User enters health measurements and gets a diabetes prediction

---

## 🧠 Models Included

- Logistic Regression  
- Random Forest  
- Support Vector Machine
- K-Nearest Neighbors  
- Naive Bayes  
- Gradient Boosting Classifier  
- Decision Tree 
- XGBoost  
- LightGBM

## 📊 Dataset

This project uses the **Pima Indians Diabetes dataset** with the following features:

- `Pregnancies` — Number of times pregnant  
- `Glucose` — Plasma glucose concentration (2 hours in an oral glucose tolerance test)  
- `BloodPressure` — Diastolic blood pressure (mm Hg)  
- `SkinThickness` — Triceps skin-fold thickness (mm)  
- `Insulin` — 2-hour serum insulin (mu U/ml)  
- `BMI` — Body mass index (kg/m²)  
- `DiabetesPedigreeFunction` — Diabetes pedigree function  
- `Age` — Age in years  
- `Outcome` — 0 (Non-diabetic), 1 (Diabetic)

## ⚙️ Setup & Installation

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2️⃣ Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
streamlit run Diabetes.py
```

Streamlit will display a local URL (typically `http://localhost:8501`) — open it in your browser.
