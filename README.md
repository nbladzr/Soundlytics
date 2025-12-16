# 🎧 Spotify Song Popularity Prediction using ANN

Project ini bertujuan untuk memprediksi popularitas lagu Spotify menggunakan Artificial Neural Network (ANN) berdasarkan fitur audio lagu.

## 📌 Dataset
Dataset terdiri dari dua kelas:
- Lagu dengan popularitas rendah (label 0)
- Lagu dengan popularitas tinggi (label 1)

Dataset diambil dari Spotify audio features.

## ⚙️ Metode
- Data preprocessing (handling missing values & scaling)
- Artificial Neural Network (MLPClassifier)
- Evaluasi menggunakan accuracy, classification report, dan confusion matrix
- Deploy menggunakan Streamlit

## 🧠 Model
- ANN dengan 3 hidden layer (128, 64, 32 neuron)
- Aktivasi ReLU
- Optimizer Adam

## 🚀 Cara Menjalankan Aplikasi
```bash
pip install -r requirements.txt
streamlit run app.py
