import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Multi Disease Prediction System",
    page_icon="🏥",
    layout="wide"
)

# ---------------------------------------------------------
# Load Models
# ---------------------------------------------------------

heart_model = joblib.load("Models/heart_disease.pkl")
diabetes_model = joblib.load("Models/diabetes.pkl")
breast_model = joblib.load("Models/breast_cancer.pkl")

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------

st.markdown("""
<style>

.main{
    background-color:#f6f8fb;
}

.block-container{
    padding-top:2rem;
}

h1{
    color:#0E5CAD;
    font-weight:700;
}

.stButton>button{
    width:100%;
    background:#0E5CAD;
    color:white;
    border:none;
    border-radius:10px;
    padding:12px;
    font-size:17px;
    font-weight:bold;
}

.stButton>button:hover{
    background:#084B8A;
}

.result-box{
    padding:20px;
    border-radius:12px;
    background:white;
    border-left:6px solid #0E5CAD;
    box-shadow:0px 2px 12px rgba(0,0,0,0.1);
    margin-top:20px;
}

.footer{
    margin-top:60px;
    text-align:center;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

st.sidebar.title("🏥 Multi Disease Prediction")

page = st.sidebar.radio(
    "Choose Disease",
    [
        "❤️ Heart Disease",
        "🩸 Diabetes",
        "🎗 Breast Cancer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This application predicts the likelihood of three diseases using trained Machine Learning models."
)

# ---------------------------------------------------------
# Prediction Function
# ---------------------------------------------------------

def show_result(model, df, disease_name):

    prediction = model.predict(df)[0]

    confidence = None

    try:
        confidence = model.predict_proba(df).max() * 100
    except:
        pass

    st.markdown("<div class='result-box'>", unsafe_allow_html=True)

    if prediction == 1:

        st.error(f"⚠ {disease_name} Detected")

    else:

        st.success("✅ No Disease Detected")

    if confidence is not None:

        st.metric(
            label="Prediction Confidence",
            value=f"{confidence:.2f}%"
        )

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# HEART DISEASE PAGE
# =========================================================

if page == "❤️ Heart Disease":

    st.title("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    values = {}

    with col1:

        values["age"] = st.slider(
            "Age",
            18,
            100,
            40
        )

        sex = st.selectbox(
            "Sex",
            [
                "Female",
                "Male"
            ]
        )

        values["sex"] = 1 if sex == "Male" else 0

        chest_pain = st.selectbox(
            "Chest Pain Type",
            [
                "Typical Angina",
                "Atypical Angina",
                "Non-anginal Pain",
                "Asymptomatic"
            ]
        )

        values["cp"] = {
            "Typical Angina":0,
            "Atypical Angina":1,
            "Non-anginal Pain":2,
            "Asymptomatic":3
        }[chest_pain]

        values["trestbps"] = st.slider(
            "Resting Blood Pressure (mm Hg)",
            80,
            220,
            120
        )

        values["chol"] = st.slider(
            "Serum Cholesterol (mg/dL)",
            100,
            600,
            200
        )

        fasting = st.selectbox(
            "Fasting Blood Sugar >120 mg/dL",
            [
                "No",
                "Yes"
            ]
        )

        values["fbs"] = 1 if fasting == "Yes" else 0

        ecg = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST-T Wave Abnormality",
                "Left Ventricular Hypertrophy"
            ]
        )

        values["restecg"] = {
            "Normal":0,
            "ST-T Wave Abnormality":1,
            "Left Ventricular Hypertrophy":2
        }[ecg]
    with col2:

        values["thalach"] = st.slider(
            "Maximum Heart Rate Achieved",
            60,
            220,
            150
        )

        exercise = st.selectbox(
            "Exercise Induced Angina",
            [
                "No",
                "Yes"
            ]
        )

        values["exang"] = 1 if exercise == "Yes" else 0

        values["oldpeak"] = st.slider(
            "ST Depression",
            0.0,
            7.0,
            1.0,
            0.1
        )

        slope = st.selectbox(
            "Slope of Peak Exercise ST Segment",
            [
                "Upsloping",
                "Flat",
                "Downsloping"
            ]
        )

        values["slope"] = {
            "Upsloping":0,
            "Flat":1,
            "Downsloping":2
        }[slope]

        values["ca"] = st.selectbox(
            "Number of Major Vessels",
            [0,1,2,3,4]
        )

        thal = st.selectbox(
            "Thalassemia Test",
            [
                "Normal",
                "Fixed Defect",
                "Reversible Defect"
            ]
        )

        values["thal"] = {
            "Normal":3,
            "Fixed Defect":6,
            "Reversible Defect":7
        }[thal]

    if st.button("Predict Heart Disease"):

        df = pd.DataFrame([values])

        show_result(
            heart_model,
            df,
            "Heart Disease"
        )

# =========================================================
# DIABETES PAGE
# =========================================================

elif page == "🩸 Diabetes":

    st.title("🩸 Diabetes Prediction")

    col1, col2 = st.columns(2)

    values = {}

    with col1:

        values["gender"] = st.selectbox(
            "Gender",
            [
                "Male",
                "Female",
                "Other"
            ]
        )

        values["age"] = st.slider(
            "Age",
            1,
            100,
            30
        )

        hypertension = st.selectbox(
            "Hypertension",
            [
                "No",
                "Yes"
            ]
        )

        values["hypertension"] = 1 if hypertension == "Yes" else 0

        heart = st.selectbox(
            "Previous Heart Disease",
            [
                "No",
                "Yes"
            ]
        )

        values["heart_disease"] = 1 if heart == "Yes" else 0

    with col2:

        smoking = st.selectbox(
            "Smoking History",
            [
                "No Info",
                "Never",
                "Former",
                "Current",
                "Ever",
                "Not Current"
            ]
        )

        values["smoking_history"] = {
            "No Info":"No Info",
            "Never":"never",
            "Former":"former",
            "Current":"current",
            "Ever":"ever",
            "Not Current":"not current"
        }[smoking]

        values["bmi"] = st.slider(
            "Body Mass Index (BMI)",
            10.0,
            60.0,
            25.0,
            0.1
        )

        values["HbA1c_level"] = st.slider(
            "HbA1c Level (%)",
            3.0,
            15.0,
            5.5,
            0.1
        )

        values["blood_glucose_level"] = st.slider(
            "Blood Glucose Level (mg/dL)",
            50,
            400,
            120
        )

    if st.button("Predict Diabetes"):

        df = pd.DataFrame([values])

        show_result(
            diabetes_model,
            df,
            "Diabetes"
        )

# =========================================================
# BREAST CANCER PAGE
# =========================================================

else:

    st.title("🎗 Breast Cancer Prediction")

    feature_labels = {

        "radius1":"Mean Radius",
        "texture1":"Mean Texture",
        "perimeter1":"Mean Perimeter",
        "area1":"Mean Area",
        "smoothness1":"Mean Smoothness",
        "compactness1":"Mean Compactness",
        "concavity1":"Mean Concavity",
        "concave_points1":"Mean Concave Points",
        "symmetry1":"Mean Symmetry",
        "fractal_dimension1":"Mean Fractal Dimension",

        "radius2":"Radius Standard Error",
        "texture2":"Texture Standard Error",
        "perimeter2":"Perimeter Standard Error",
        "area2":"Area Standard Error",
        "smoothness2":"Smoothness Standard Error",
        "compactness2":"Compactness Standard Error",
        "concavity2":"Concavity Standard Error",
        "concave_points2":"Concave Points Standard Error",
        "symmetry2":"Symmetry Standard Error",
        "fractal_dimension2":"Fractal Dimension Standard Error",

        "radius3":"Worst Radius",
        "texture3":"Worst Texture",
        "perimeter3":"Worst Perimeter",
        "area3":"Worst Area",
        "smoothness3":"Worst Smoothness",
        "compactness3":"Worst Compactness",
        "concavity3":"Worst Concavity",
        "concave_points3":"Worst Concave Points",
        "symmetry3":"Worst Symmetry",
        "fractal_dimension3":"Worst Fractal Dimension"
    }

    feats = list(feature_labels.keys())

    values = {}

    st.subheader("Mean, Standard Error & Worst Measurements")

    c1, c2 = st.columns(2)
    for i, feature in enumerate(feats):

        with (c1 if i % 2 == 0 else c2):

            values[feature] = st.number_input(
                feature_labels[feature],
                min_value=0.0,
                value=0.0,
                format="%.5f"
            )

    if st.button("Predict Breast Cancer"):

        df = pd.DataFrame([values])

        show_result(
            breast_model,
            df,
            "Malignant Tumor"
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        <h4>🏥 Multi Disease Prediction System</h4>
        <p>
            Developed using <b>Streamlit</b> and <b>Machine Learning</b>.<br>
            This tool is intended for educational purposes only and should not replace professional medical advice.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
