
import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))

st.title("Advanced Heart Disease Predictor")

age = st.number_input("Age", 18, 100)
sex = st.selectbox("Sex", [0, 1])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200)
chol = st.number_input("Cholesterol", 100, 400)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = true; 0 = false)", [0, 1])
restecg = st.selectbox("Resting ECG", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 200)
exang = st.selectbox("Exercise Induced Angina (1 = yes; 0 = no)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0)
slope = st.selectbox("Slope", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thal", [0, 1, 2, 3])

if st.button("Predict"):
    input_data = [[age, sex, cp, trestbps, chol, fbs, restecg,
                   thalach, exang, oldpeak, slope, ca, thal]]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("🫀 The person is likely to have heart disease.")
    else:
        st.info("💚 The person is unlikely to have heart disease.")
