import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# KONFIGURASI HALAMAN
# =========================
st.set_page_config(
    page_title="Soundlytics",
    page_icon="🎧",
    layout="centered"
)

# =========================
# LOAD DATASET (AMAN)
# =========================
DATA_PATH = "spotify_dataset.csv"  # CSV HARUS ADA DI ROOT REPO

if not os.path.exists(DATA_PATH):
    st.error("❌ File spotify_dataset.csv tidak ditemukan di repository.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# =========================
# LOAD MODEL & SCALER
# =========================
if not os.path.exists("ann_model.pkl") or not os.path.exists("scaler.pkl"):
    st.error("❌ Model atau scaler tidak ditemukan.")
    st.stop()

model = pickle.load(open("ann_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# =========================
# UI
# =========================
st.title("🎧 TuneScope")
st.caption("Prediksi Popularitas Lagu Spotify menggunakan Artificial Neural Network (ANN)")

st.markdown("---")

# =========================
# INPUT JUDUL LAGU
# =========================
song_title = st.text_input("🔍 Masukkan Judul Lagu")

if st.button("Prediksi"):
    if song_title.strip() == "":
        st.warning("Masukkan judul lagu terlebih dahulu.")
    else:
        # Cari lagu (tidak case-sensitive)
        song = df[df["track_name"].str.lower() == song_title.lower()]

        if song.empty:
            st.error("❌ Lagu tidak ditemukan di dataset.")
        else:
            # Ambil fitur numerik
            features = song.select_dtypes(include=np.number)

            # Hapus target label
            if "popularity_label" in features.columns:
                features = features.drop(columns=["popularity_label"])

            # Scaling
            features_scaled = scaler.transform(features)

            # Prediksi
            pred = model.predict(features_scaled)[0]

            # Output hasil
            st.markdown("### 🎯 Hasil Prediksi")
            if pred == 1:
                st.success("🔥 Lagu ini DIPREDIKSI POPULER")
            else:
                st.warning("🎼 Lagu ini DIPREDIKSI TIDAK POPULER")

            # Info lagu
            st.markdown("### ℹ️ Informasi Lagu")
            st.write(song[["track_name", "track_artist", "playlist_genre"]])
