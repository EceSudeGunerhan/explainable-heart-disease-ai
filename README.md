# Explainable Heart Disease AI  
**Kalp Hastalığı Risk Tahmin ve Açıklama Sistemi**

Explainable Heart Disease AI, klinik karar destek amacıyla geliştirilmiş, bireylerin kalp hastalığı riskini tahmin eden ve bu tahminlerin nedenlerini açıklayan yapay zeka tabanlı bir web uygulamasıdır.  
Uygulama **teşhis koymaz**, yalnızca **destekleyici risk analizi** sunar.

Model tahminleri; istatistiksel öğrenme, açıklanabilir yapay zeka (SHAP) ve etkileşimli bir Streamlit arayüzü ile kullanıcıya sunulmaktadır.

**Canlı Uygulama (Streamlit):**  
https://explainable-heart-disease-ai.streamlit.app

---

## Proje Özellikleri

- Kalp hastalığı riski tahmini (Binary Classification)
- Logistic Regression, Decision Tree ve Random Forest karşılaştırması
- Optimum threshold analizi
- SHAP ile özellik bazlı açıklanabilirlik
- Streamlit tabanlı kullanıcı arayüzü
- Gerçek zamanlı tahmin ve görselleştirme

---

## Veri Seti

- **Kaynak:** `heart.csv`
- **Gözlem Sayısı:** 1025 (duplicate temizliği sonrası 302)
- **Özellik Sayısı:** 13
- **Hedef Değişken:**  
  - `target = 1` → Kalp hastalığı riski var  
  - `target = 0` → Risk yok

### Kullanılan Özellikler
- age, sex
- chest pain type (cp)
- resting blood pressure (trestbps)
- cholesterol (chol)
- fasting blood sugar (fbs)
- resting ECG (restecg)
- max heart rate (thalach)
- exercise induced angina (exang)
- oldpeak
- slope, ca, thal

---

## Makine Öğrenmesi Süreci

### Ön İşleme (Preprocessing)
- Duplicate kayıtların temizlenmesi
- IQR yöntemi ile aykırı değer baskılama (clipping)
- Train / Test ayrımı (%80 / %20)

### Kullanılan Modeller
- **Logistic Regression** (final model)
- Decision Tree
- Random Forest

### Final Model Performansı (Logistic Regression)
- **Accuracy:** ~0.85  
- **Precision:** ~0.88  
- **Recall:** ~0.85  
- **F1-score:** ~0.86  
- **ROC-AUC:** ~0.91  

---

## Threshold Analizi

Farklı karar eşikleri test edilerek precision–recall dengesi incelenmiştir.  
Model için **en dengeli eşik değeri ~0.50** olarak belirlenmiştir.

---

## Açıklanabilir Yapay Zeka (SHAP)

- Her tahmin için özellik katkıları hesaplanır
- Global feature importance görselleştirilir
- Modelin “neden bu kararı verdiği” kullanıcıya açıkça gösterilir

Örnek:
- `cp`, `thalach`, `oldpeak` gibi özelliklerin risk üzerindeki etkileri görselleştirilir

---

## Streamlit Arayüzü

Uygulama kullanıcıdan klinik verileri alır ve:

1. **Risk Tahmini**
   - Kalp hastalığı riski var / yok
2. **Olasılık Skoru**
   - Modelin tahmin güveni
3. **SHAP Açıklaması**
   - Özellik bazlı katkı grafiği

Modern, sade ve etkileşimli bir tasarım kullanılmıştır.

---

## Proje Dosya Yapısı

heard_disease_ml/
├── data/
│ └── heart.csv
├── notebooks/
│ └── 01_eda.ipynb
├── src/
│ ├── preprocessing.py
│ ├── train_logistic.py
│ ├── train_decision_tree.py
│ ├── train_random_forest.py
│ ├── validate_logistic.py
│ ├── threshold_analysis.py
│ ├── shap_explain.py
│ └── save_model.py
├── app.py
├── model.pkl
├── requirements.txt
└── README.md


---

## Uyarı

Bu uygulama **tıbbi teşhis amacıyla kullanılamaz**.  
Sadece **akademik ve karar destek** amaçlıdır.

---

## Geliştirici

**Ece Sude Günerhan**  
Süleyman Demirel Üniversitesi  
Bilgisayar Mühendisliği  

İlgi Alanları:
- Makine Öğrenmesi
- Açıklanabilir Yapay Zeka (XAI)
- Veri Bilimi
- Klinik karar destek sistemleri
