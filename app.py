import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import io
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

# ---------------- Settings ----------------
DATA_FILE = "Cardiovascular_Disease_Dataset.csv"
MODEL_FILE = "best_model.pkl"
TIME_FILE = "last_retrain.txt"
RETRAIN_INTERVAL_HOURS = 48
MIN_DATA_SIZE = 150

# ---------------- Retraining Logic ----------------
def should_retrain():
    if not os.path.exists(DATA_FILE):
        return False
    df = pd.read_csv(DATA_FILE)
    if len(df) < MIN_DATA_SIZE:
        return False
    if not os.path.exists(TIME_FILE):
        return True
    with open(TIME_FILE, "r") as f:
        last_time = datetime.fromisoformat(f.read().strip())
    return datetime.now() - last_time >= timedelta(hours=RETRAIN_INTERVAL_HOURS)

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

if should_retrain() or not os.path.exists(MODEL_FILE):
    train_model()

model, scaler = joblib.load(MODEL_FILE)

# ---------------- Streamlit UI ----------------
st.title("❤️ Heart Disease Prediction & Awareness")
st.write(f"ℹ️ The model retrains every **{RETRAIN_INTERVAL_HOURS} hours** (only if dataset has ≥ {MIN_DATA_SIZE} rows).")

patient_name = st.text_input("👤 Enter your name", "Anonymous")

# Simplified inputs for normal users
age = st.slider("Age", 18, 100, 40)
gender = st.radio("Gender", ["Male", "Female"])
gender = 1 if gender == "Male" else 0

chestpain = st.radio("Do you feel chest pain?", ["No", "Mild", "Moderate", "Severe"])
chestpain = ["No", "Mild", "Moderate", "Severe"].index(chestpain)

restingBP = st.slider("Blood Pressure (mmHg)", 80, 200, 120)
serumcholestrol = st.slider("Cholesterol (mg/dl)", 120, 350, 200)

fastingbloodsugar = st.radio("Blood Sugar > 120 mg/dl?", ["No", "Yes"])
fastingbloodsugar = 1 if fastingbloodsugar == "Yes" else 0

maxheartrate = st.slider("Heart Rate (bpm)", 60, 200, 150)

# Technical values (hidden from normal user)
restingrelectro = 1
exerciseangia = 0
oldpeak = 1.0
slope = 1
noofmajorvessels = 0

# ---------------- Prediction ----------------
if st.button("🔍 Predict"):
    input_data = np.array([[age, gender, chestpain, restingBP, serumcholestrol,
                            fastingbloodsugar, restingrelectro, maxheartrate,
                            exerciseangia, oldpeak, slope, noofmajorvessels]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        risk_msg = "⚠️ High Risk of Heart Disease!"
        st.error(risk_msg)
    else:
        risk_msg = "✅ Low Risk of Heart Disease"
        st.success(risk_msg)

    # ---------------- Awareness Tips ----------------
    tips, medicines = [], []

    if restingBP > 140:
        tips.append("Reduce salt intake and monitor your blood pressure regularly.")
        medicines.append("Doctor may suggest antihypertensives (e.g., Amlodipine).")
    else:
        tips.append("Maintain your healthy BP with regular checkups and a balanced diet.")

    if serumcholestrol > 240:
        tips.append("Limit fried and oily food. Add more fiber (oats, vegetables).")
        medicines.append("Statins may be prescribed (consult your doctor).")
    else:
        tips.append("Continue maintaining a cholesterol-friendly diet.")

    if fastingbloodsugar == 1:
        tips.append("Avoid excessive sugar. Prefer whole grains and fruits.")
        medicines.append("Metformin may be prescribed if diabetes is detected.")
    else:
        tips.append("Keep your sugar levels under control with a balanced diet.")

    if maxheartrate < 60:
        tips.append("Low heart rate detected. Consult a doctor if dizziness occurs.")
    elif maxheartrate > 180:
        tips.append("Very high heart rate. Avoid over-exertion and seek medical advice.")
    else:
        tips.append("Your heart rate is within a safe range.")

    tips.append("🏃 Exercise 30 minutes daily (walking, cycling, or yoga).")
    tips.append("🥦 Eat more vegetables, fruits, and lean proteins.")
    tips.append("🚭 Avoid smoking and limit alcohol.")

    st.subheader("📊 Personalized Health Tips")
    for t in tips:
        st.write("- " + t)

    st.subheader("💊 Medicine Awareness (Educational Only)")
    for m in medicines:
        st.write("- " + m)

    st.warning("⚠️ This is an AI-based awareness tool, not a substitute for professional medical advice.")

    # ---------------- PDF Report ----------------
    st.subheader("📝 Heart Health Report")
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 770, "❤️ Heart Health Report")

    report_date = datetime.now().strftime("%d %B %Y, %I:%M %p")
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, f"👤 Patient: {patient_name}")
    c.drawString(350, 750, f"🗓️ Date: {report_date}")

    c.drawString(50, 720, "📌 Patient Inputs:")
    data = [
        ["Age", str(age)],
        ["Gender", "Male" if gender==1 else "Female"],
        ["Blood Pressure (mmHg)", str(restingBP)],
        ["Cholesterol (mg/dl)", str(serumcholestrol)],
        ["Chest Pain", ["No","Mild","Moderate","Severe"][chestpain]],
        ["Blood Sugar >120", "Yes" if fastingbloodsugar==1 else "No"],
        ["Heart Rate (bpm)", str(maxheartrate)],
    ]
    table = Table(data, colWidths=[200, 200])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 1, colors.grey),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
    ]))
    table.wrapOn(c, 50, 680)
    table.drawOn(c, 50, 630)

    
        # --- Risk Status with colors + badge ---
    c.setFont("Helvetica-Bold", 14)
    if prediction == 1:
        c.setFillColor(colors.red)
        badge = "⚠️"
    else:
        c.setFillColor(colors.green)
        badge = "✅"
        
    c.drawString(50, 600, f"Risk Status: {badge} {risk_msg}")
    c.setFillColor(colors.black)  # reset back to normal
        # --- Recommendations Section ---
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor(colors.darkblue)
    c.drawString(50, 570, "💡 Recommendations:")
    c.setFillColor(colors.black)
    # --- Personalized Advice System ---

# Diet advice
    if restingBP > 140:
        diet_advice = "Reduce salt intake, avoid processed foods, eat more fruits and leafy greens."
    elif serumcholestrol > 240:
        diet_advice = "Avoid fried/junk foods, prefer whole grains, oats, beans, and nuts."
    elif fastingbloodsugar == 1:
        diet_advice = "Limit sugar, eat complex carbs (brown rice, whole wheat), add vegetables."
    else:
        diet_advice = "Maintain a balanced diet with vegetables, fruits, lean proteins, and whole grains."

# Exercise advice
    if maxheartrate < 100:
        exercise_advice = "Avoid heavy workouts, start with light walking and consult a doctor."
    elif exerciseangia == 1:
        exercise_advice = "Avoid intense exercise, focus on yoga, meditation, and breathing exercises."
    else:
        exercise_advice = "Do 30 minutes of moderate exercise daily like walking, cycling, or swimming."

# Medicine advice
    if prediction == 1:  # high risk
        medicine_advice = "Consult a cardiologist, follow prescribed medicines, and schedule regular checkups."
    else:
        medicine_advice = "No immediate medication required, but monitor BP, sugar, and cholesterol regularly."

    # Diet advice
    c.setFont("Helvetica", 12)
    c.setFillColor(colors.green)
    c.drawString(70, 540, f"🥗 Diet: {diet_advice}")
    
    # Exercise advice
    c.setFillColor(colors.orange)
    c.drawString(70, 520, f"🏃 Exercise: {exercise_advice}")
    
    # Medicine advice
    c.setFillColor(colors.red)
    c.drawString(70, 500, f"💊 Medicine: {medicine_advice}")
    
    # Reset to black
    c.setFillColor(colors.black)

    
