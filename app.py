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
st.write("Bu sistem klinik karar destek amaçlıdır, **teşhis koymaz**.")

# --------------------
# Kullanıcı girişi
# --------------------

age = st.slider(
    "Yaş",
    20,
    80,
    50,
    help="Hastanın yaşı (20–80 arası). Yaş arttıkça kalp hastalığı riski artabilir."
)

sex = st.selectbox(
    "Cinsiyet",
    [0, 1],
    format_func=lambda x: "Kadın" if x == 0 else "Erkek",
    help="Kadın = 0, Erkek = 1. Erkeklerde kalp hastalığı riski genellikle daha yüksektir."
)

st.markdown("""
**Göğüs ağrısı tipi (cp):**  
- **0:** Tipik anjina (kalp kaynaklı klasik göğüs ağrısı)  
- **1:** Atipik anjina  
- **2:** Anjinal olmayan ağrı  
- **3:** Asemptomatik (ağrı yok)
""")
cp = st.selectbox("Göğüs ağrısı tipi (cp)", [0, 1, 2, 3])

trestbps = st.slider(
    "Dinlenme kan basıncı (mmHg)",
    90,
    200,
    130,
    help="Hastanın dinlenme halindeki sistolik kan basıncı. Yüksek değerler risk göstergesidir."
)

chol = st.slider(
    "Kolesterol (mg/dl)",
    120,
    400,
    240,
    help="Serum kolesterol seviyesi. Yüksek kolesterol kalp hastalığı riskini artırır."
)

st.markdown("""
**Açlık kan şekeri (fbs):**  
- **0:** ≤ 120 mg/dl (normal)  
- **1:** > 120 mg/dl (yüksek)
""")
fbs = st.selectbox("Açlık kan şekeri > 120", [0, 1])

st.markdown("""
**EKG sonucu (restecg):**  
- **0:** Normal  
- **1:** ST-T dalga anormalliği  
- **2:** Sol ventrikül hipertrofisi
""")
restecg = st.selectbox("EKG sonucu", [0, 1, 2])

thalach = st.slider(
    "Maksimum kalp atım hızı",
    70,
    210,
    150,
    help="Egzersiz sırasında ulaşılan maksimum kalp atım hızı."
)

st.markdown("""
**Egzersizle anjina (exang):**  
- **0:** Hayır  
- **1:** Evet
""")
exang = st.selectbox("Egzersizle anjina", [0, 1])

oldpeak = st.slider(
    "ST depresyonu",
    0.0,
    6.0,
    1.0,
    help="Egzersize bağlı ST segment depresyonu. Yüksek değerler anormal kabul edilir."
)

st.markdown("""
**ST segment eğimi (slope):**  
- **0:** Yükselen  
- **1:** Düz  
- **2:** Azalan
""")
slope = st.selectbox("ST segment eğimi", [0, 1, 2])

st.markdown("""
**Ana damar sayısı (ca):**  
Floroskopi ile görülen ana damar sayısı (0–3).
""")
ca = st.selectbox("Ana damar sayısı", [0, 1, 2, 3])

st.markdown("""
**Thalassemia (thal):**  
- **0:** Normal  
- **1:** Sabit defekt  
- **2:** Tersinir defekt  
- **3:** Bilinmiyor
""")
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
