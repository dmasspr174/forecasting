# 📊 Airport Passenger Forecasting — LSTM Web App

A data-driven web application that predicts airport passenger volume using a Long Short-Term Memory (LSTM) model, enabling better trend analysis and decision-making.

---

## 🚀 Live Demo
[(Add your Streamlit / deployment link here)](https://forecastingpassenger.streamlit.app/)

---

## 👨‍💻 My Role
Full-Stack Developer (Machine Learning + Web App)

---

## 🧠 Overview
This project leverages time-series forecasting using LSTM to predict passenger numbers based on historical data. The application provides an interactive interface for input, prediction, and visualization, making complex forecasting accessible to end users.

---

## ⚙️ Key Features
- 📥 Input passenger data (12 months)
- 🤖 Automatic prediction using LSTM model
- 📈 Interactive visualization (charts & tables)
- ✅ Input validation to prevent errors
- 💾 Data storage using Firebase (Firestore)
- 🌐 Web-based interface using Streamlit

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras (LSTM)  
- Streamlit  
- Firebase (Firestore)  
- Pandas, NumPy, Matplotlib  

---

## 🧪 Model Details
- Model: Long Short-Term Memory (LSTM)
- Type: Time-Series Forecasting
- Input: Historical passenger data (monthly)
- Output: Predicted passenger volume
- Model saved as `.h5` for efficient reuse

---

## 📸 Preview
forecasting page
<img width="463" height="218" alt="image" src="https://github.com/user-attachments/assets/65143f28-02ce-45c0-b326-683afe25d36d" />
history forecasting page
<img width="491" height="229" alt="image" src="https://github.com/user-attachments/assets/8a864b84-f208-4dcf-acb5-4352b3787c49" />


## 🔄 Workflow
1. User inputs historical passenger data  
2. Data is preprocessed and validated  
3. LSTM model generates prediction  
4. Results are displayed in charts and tables  
5. Data optionally stored in Firebase  

---

## 📈 Impact
- Enables data-driven passenger trend analysis  
- Reduces manual forecasting effort  
- Improves efficiency in decision-making processes  

---

## 🎯 Project Value
This project demonstrates my ability to integrate machine learning models into a web application, including data processing, model deployment, and interactive visualization.

---

## 📦 Installation

```bash
git clone https://github.com/dmasspr174/forecasting.git
cd forecasting
pip install -r requirements.txt
streamlit run app.py
