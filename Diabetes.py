import streamlit as st
import pandas as pd
import pickle

# Title
st.title("Health Risk Prediction App")

# Load the model
@st.cache_data
def load_model():
    with open("deva.pkl", "rb") as file:  # replace with your .pkl file path
        model = pickle.load(file)
    return model

model = load_model()

# Sidebar inputs
st.sidebar.header("Input Features")

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
smoking_history = st.sidebar.selectbox("Smoking History", ["never","No Info", "current", "former", "ever", "not current"])
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
HbA1c_level = st.sidebar.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=5.5)
blood_glucose_level = st.sidebar.number_input("Blood Glucose Level", min_value=50, max_value=500, value=100)

# Map categorical variables if needed
gender_map = {"Male": 0, "Female": 1, "Other": 2}
smoking_map = {"never": 4, "No Info":0, "current":1, "former":3, "ever":2, "not current":5}

input_data = pd.DataFrame({
    "gender": [gender_map[gender]],
    "age": [age],
    "hypertension": [hypertension],
    "heart_disease": [heart_disease],
    "smoking_history": [smoking_map[smoking_history]],
    "bmi": [bmi],
    "HbA1c_level": [HbA1c_level],
    "blood_glucose_level": [blood_glucose_level]
})

st.subheader("Input Data")
st.write(input_data)

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data) if hasattr(model, "predict_proba") else None

    st.subheader("Prediction")
    st.write(prediction[0])

    if prediction_proba is not None:
        st.subheader("Prediction Probability")
        st.write(prediction_proba)
