# 📊 Temporal Fusion Transformer for FMCG Sales Forecasting

This project implements a Flask-based backend that powers an intelligent, interpretable, multi-horizon forecasting system for FMCG sales. It uses the Temporal Fusion Transformer (TFT) architecture to provide demand predictions across various time horizons (7-day, 14-day, 30-day), helping retailers optimize inventory and reduce waste.

> Developed as part of our capstone project at Bennett University (Team 54), this system is integrated with a mobile app built in React Native for real-time interaction.

---

## 🎯 Problem Statement

In the volatile FMCG domain, inaccurate forecasting leads to overstocking, wastage, and lost sales. Traditional statistical models fail to capture the nonlinear, seasonal, and promotion-driven patterns in sales. Our system solves this by applying state-of-the-art deep learning (TFT) to forecast demand while remaining interpretable and scalable.

---

## 🚀 System Highlights

- 🧠 **Temporal Fusion Transformer (TFT)** for deep sequence learning
- 📈 **Multi-horizon forecasting** with attention-driven interpretability
- ⚙️ **Flask API** for mobile integration
- 🔐 **JWT Authentication** and user-specific forecast saving
- 📁 Handles 120M+ records across 200K+ SKUs from the Favorita dataset

---

## 🛠️ Technologies Used

- Python 3.9+
- Flask
- PyTorch (TFT implementation via `pytorch-forecasting`)
- Pandas, NumPy, Scikit-learn
- Flask-CORS, JWT for API and auth
- Render for backend hosting

---

## 📦 Project Structure
forecast-backend/
├── app.py # Main Flask API
├── model/ # TorchScript model + scaler
├── utils/ # Data preprocessing and validation
├── requirements.txt
└── README.md

## 🔗 Related Repo
Frontend (React Native + Expo): https://github.com/athira526/Sales-Forecast-App

## 📌 Notable Features
⚡ Model supports 7/14/30-day multi-step forecasts

🔍 Visual insights via attention weights

🔒 Secure login with JWT-based auth

📂 Upload Excel sales data (via mobile app)

## 👥 Contributors
Athira Ravi Pillai

Lakshit Gupta

Krish Jain

Mentored by Dr. Riti Kushwaha (CS Dept., Bennett University)

## 📚 References
Temporal Fusion Transformer Paper

McKinsey, Gartner, Accenture reports on AI in FMCG

Favorita Grocery Sales dataset


## Screenshots
![Screenshot 2025-06-24 161844](https://github.com/user-attachments/assets/be2c321c-c6ee-4f47-acbe-cec239c74037)


![Screenshot 2025-06-24 161852](https://github.com/user-attachments/assets/ced41418-54c7-47fd-9fb7-ae34de11c25b)

![Screenshot 2025-06-24 161710](https://github.com/user-attachments/assets/4659c6f4-183a-42e2-9f24-915ef1fd2706)


![WhatsApp Image 2025-04-19 at 16 08 17_0d68dbce](https://github.com/user-attachments/assets/0f32a180-eadb-4995-a4d7-e357a05a706c)
