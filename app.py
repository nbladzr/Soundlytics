import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# Load model & scaler
# =========================
model = pickle.load(open("ann_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# Load dataset
# =========================
df = pd.read_csv("dataset/spotify_dataset.csv")  # gabungan low + high

# =========================
# UI
# =========================
st.set_page_config(page_title="TuneScope", page_icon="🎧")

st.title("🎧 TuneScope")
st.caption("Prediksi Popularitas Lagu Spotify dengan ANN")

# =========================
# Input judul lagu
# =========================
song_title = st.text_input("🔍 Masukkan Judul Lagu")

if st.button("Prediksi"):
    if song_title.strip() == "":
        st.warning("Masukkan judul lagu terlebih dahulu")
    else:
        # Cari lagu
        song = df[df["track_name"].str.lower() == song_title.lower()]

        if song.empty:
            st.error("Lagu tidak ditemukan di dataset")
        else:
            # Ambil fitur numerik
            features = song.select_dtypes(include=np.number).drop(columns=["popularity_label"])

            # Scaling
            features_scaled = scaler.transform(features)

            # Prediksi
            pred = model.predict(features_scaled)[0]

            if pred == 1:
                st.success("🔥 Lagu ini DIPREDIKSI POPULER")
            else:
                st.warning("🎼 Lagu ini DIPREDIKSI TIDAK POPULER")
