# Multi-disease-predictor
Machine Learning model,that can predict heart disease,diabetes and breast cancer from your medical data, this project is one of my task of summer internship at codeAlpha.
# 🏥 Multi Disease Prediction System

A Machine Learning-based web application that predicts the likelihood of **Heart Disease**, **Diabetes**, and **Breast Cancer** using trained classification models. The application is built with **Python**, **Scikit-learn**, **XGBoost**, and **Streamlit**.

---

## 📌 Project Overview

This project aims to provide a simple and interactive interface for predicting three common diseases using machine learning models trained on publicly available datasets.

Users can enter medical information through the Streamlit interface and instantly receive predictions.

---

## ✨ Features

- ❤️ Heart Disease Prediction
- 🩸 Diabetes Prediction
- 🎗 Breast Cancer Prediction
- Interactive Streamlit Web Interface
- Data Preprocessing using Scikit-learn Pipelines
- Multiple Machine Learning Models Compared
- Trained Models Saved using Joblib
- Easy-to-use Prediction Dashboard

---

## 🛠 Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Joblib
- Matplotlib
- Seaborn

---

## 📊 Datasets Used

### ❤️ Heart Disease
- UCI Heart Disease Dataset

### 🩸 Diabetes
- Diabetes Prediction Dataset

### 🎗 Breast Cancer
- Breast Cancer Wisconsin (Diagnostic) Dataset

---

## 🤖 Machine Learning Models

The following models were trained and compared for each dataset:

- Logistic Regression
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- XGBoost

The best-performing model was selected for deployment.

---

## 📈 Model Performance

| Disease | Selected Model | Accuracy |
|----------|---------------|----------|
| ❤️ Heart Disease | Support Vector Machine (SVM) | **93.44%** |
| 🩸 Diabetes | XGBoost | **97.05%** |
| 🎗 Breast Cancer | Logistic Regression | **97.37%** |

---

## 📂 Project Structure

```
Multi-disease-predictor/
│
├── app.py
├── Models/
│   ├── heart_disease.pkl
│   ├── diabetes.pkl
│   └── breast_cancer.pkl
│
├── Datasets/
│
├── Notebooks/
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/fenildevganiya77/Multi-disease-predictor.git
```

Move into the project folder:

```bash
cd Multi-disease-predictor
```

---

### Create a Virtual Environment (Optional)

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at:

```
http://localhost:8501
```

---

## 📷 Screenshots

Add screenshots of your application here after deployment.

Example:

- Home Page
- Heart Disease Prediction
- Diabetes Prediction
- Breast Cancer Prediction

---

## 📌 Future Improvements

- Add more disease prediction models
- Deploy the application on Streamlit Community Cloud
- Improve UI/UX with custom themes
- Add SHAP Explainable AI
- Store prediction history
- User authentication system

---

## 👨‍💻 Author

**Fenil Ganiya**

B.Tech – Data Science & Artificial Intelligence  
Indian Institute of Technology Bhilai

GitHub:
https://github.com/fenildevganiya77

---

## 📄 License

This project is developed for educational purposes and as part of the **CodeAlpha Machine Learning Internship**.
