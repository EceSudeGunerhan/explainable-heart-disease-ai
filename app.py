import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap

# --------------------
# Ayarlar
# --------------------
st.set_page_config(
    page_title="Kalp Hastalığı Risk Tahmini",
    layout="centered"
)

THRESHOLD = 0.35
MODEL_PATH = "model.pkl"

# --------------------
# Model yükle
# --------------------
model = joblib.load(MODEL_PATH)

st.title("Kalp Hastalığı Risk Tahmin Sistemi")
st.write("Bu sistem klinik karar destek amaçlıdır, teşhis koymaz.")

# --------------------
# Kullanıcı girişi
# --------------------
age = st.slider("Yaş", 20, 80, 50)
sex = st.selectbox("Cinsiyet", [0, 1], format_func=lambda x: "Kadın" if x == 0 else "Erkek")
cp = st.selectbox("Göğüs ağrısı tipi (cp)", [0, 1, 2, 3])
trestbps = st.slider("Dinlenme kan basıncı", 90, 200, 130)
chol = st.slider("Kolesterol (mg/dl)", 120, 400, 240)
fbs = st.selectbox("Açlık kan şekeri > 120", [0, 1])
restecg = st.selectbox("EKG sonucu", [0, 1, 2])
thalach = st.slider("Maksimum kalp atım hızı", 70, 210, 150)
exang = st.selectbox("Egzersizle anjina", [0, 1])
oldpeak = st.slider("ST depresyonu", 0.0, 6.0, 1.0)
slope = st.selectbox("ST segment eğimi", [0, 1, 2])
ca = st.selectbox("Ana damar sayısı", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia", [0, 1, 2, 3])

input_df = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
}])

# --------------------
# Tahmin
# --------------------
if st.button("Riski Hesapla"):
    proba = model.predict_proba(input_df)[0][1]
    prediction = int(proba >= THRESHOLD)

    st.subheader("Sonuç")
    st.write(f"Risk Olasılığı: **{proba:.2%}**")

    if prediction == 1:
        st.error("🔴 Yüksek Kalp Hastalığı Riski")
    else:
        st.success("🟢 Düşük Kalp Hastalığı Riski")

    # --------------------
    # SHAP açıklama
    # --------------------
    X_background = model.named_steps["preprocessor"].transform(input_df)
    explainer = shap.Explainer(
        model.named_steps["model"],
        X_background
    )
    shap_values = explainer(X_background)

    st.subheader("Tahmin Açıklaması (SHAP)")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(bbox_inches="tight")
