
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Multi Disease Prediction", layout="wide")

heart_model = joblib.load("Models/heart_disease.pkl")
diabetes_model = joblib.load("Models/diabetes.pkl")
breast_model = joblib.load("Models/breast_cancer.pkl")

st.sidebar.title("🏥 Multi Disease Prediction")
page = st.sidebar.radio("Select Disease",["Heart Disease","Diabetes","Breast Cancer"])

def show_result(model, df, positive_label):
    pred = model.predict(df)[0]
    try:
        prob = model.predict_proba(df).max()*100
        st.metric("Confidence", f"{prob:.2f}%")
    except Exception:
        pass
    if pred==1:
        st.error(positive_label)
    else:
        st.success("No disease detected")

if page=="Heart Disease":
    st.title("❤️ Heart Disease Prediction")
    cols=["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]
    vals={}
    for c in cols:
        vals[c]=st.number_input(c,value=0.0) if c=="oldpeak" else st.number_input(c,value=0)
    if st.button("Predict Heart Disease"):
        show_result(heart_model,pd.DataFrame([vals]),"Heart Disease Detected")

elif page=="Diabetes":
    st.title("🩸 Diabetes Prediction")
    vals={}
    vals["gender"]=st.selectbox("Gender",["Male","Female","Other"])
    vals["age"]=st.number_input("Age",0.0,120.0)
    vals["hypertension"]=st.selectbox("Hypertension",[0,1])
    vals["heart_disease"]=st.selectbox("Heart Disease",[0,1])
    vals["smoking_history"]=st.selectbox("Smoking History",["No Info","never","former","current","ever","not current"])
    vals["bmi"]=st.number_input("BMI",0.0,100.0)
    vals["HbA1c_level"]=st.number_input("HbA1c Level",0.0,15.0)
    vals["blood_glucose_level"]=st.number_input("Blood Glucose",0,400)
    if st.button("Predict Diabetes"):
        show_result(diabetes_model,pd.DataFrame([vals]),"Diabetes Detected")

else:
    st.title("🎗 Breast Cancer Prediction")
    feats=['radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1','symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2','symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3','symmetry3','fractal_dimension3']
    vals={}
    c1,c2=st.columns(2)
    for i,f in enumerate(feats):
        with (c1 if i%2==0 else c2):
            vals[f]=st.number_input(f,value=0.0,format="%.5f")
    if st.button("Predict Breast Cancer"):
        show_result(breast_model,pd.DataFrame([vals]),"Malignant Tumor Predicted")
