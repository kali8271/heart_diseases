import streamlit as st
import numpy as np
import pandas as pd
import joblib, os
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "Cardiovascular_Disease_Dataset.csv"
MODEL_FILE = "best_model.pkl"
TIME_FILE = "last_retrain.txt"
RETRAIN_INTERVAL_HOURS = 72   # retrain every 72 hours
MIN_DATA_SIZE = 150           # need at least 150 rows

# --- Function to check if retraining needed ---
def should_retrain():
    # condition 1: enough data
    if not os.path.exists(DATA_FILE):
        return False
    df = pd.read_csv(DATA_FILE)
    if len(df) < MIN_DATA_SIZE:
        return False
    
    # condition 2: time-based
    if not os.path.exists(TIME_FILE):
        return True
    with open(TIME_FILE, "r") as f:
        last_time = datetime.fromisoformat(f.read().strip())
    return datetime.now() - last_time >= timedelta(hours=RETRAIN_INTERVAL_HOURS)

# --- Train and save model ---
def train_model():
    df = pd.read_csv(DATA_FILE)
    X = df.drop(["patientid", "target"], axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump((model, scaler), MODEL_FILE)
    with open(TIME_FILE, "w") as f:
        f.write(datetime.now().isoformat())
    st.success("✅ Model retrained and saved!")

# --- Load or retrain ---
if should_retrain() or not os.path.exists(MODEL_FILE):
    train_model()

model, scaler = joblib.load(MODEL_FILE)

# --- Streamlit UI ---
st.title("❤️ Heart Disease Prediction (Scheduled + Data-Based Retrain)")
st.write(f"Model auto-retrains every {RETRAIN_INTERVAL_HOURS} hours **only if dataset has ≥ {MIN_DATA_SIZE} rows**.")

# Example inputs
age = st.number_input("Age", min_value=1, max_value=120, value=40)
gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
chestpain = st.selectbox("Chest Pain Type (0–3)", [0,1,2,3])
restingBP = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
serumcholestrol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fastingbloodsugar = st.selectbox("Fasting Blood Sugar >120 mg/dl", [0,1])
restingrelectro = st.selectbox("Resting Electrocardiographic Results (0–2)", [0,1,2])
maxheartrate = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exerciseangia = st.selectbox("Exercise Induced Angina", [0,1])
oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
slope = st.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0,1,2])
noofmajorvessels = st.selectbox("Number of Major Vessels (0–3)", [0,1,2,3])

if st.button("Predict"):
    input_data = np.array([[age, gender, chestpain, restingBP, serumcholestrol,
                            fastingbloodsugar, restingrelectro, maxheartrate,
                            exerciseangia, oldpeak, slope, noofmajorvessels]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease!")
    else:
        st.success("✅ Low Risk of Heart Disease")
