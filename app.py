import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =========================
# Load model & scaler
# =========================
with open("ann_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# =========================
# Load dataset (AMAN)
# =========================
DATA_PATH = os.path.join("dataset", "spotify_dataset.csv")

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset tidak ditemukan. Pastikan folder 'dataset/' ada di GitHub.")
    st.stop()

df = pd.read_csv(DATA_PATH)

# =========================
# UI
# =========================
st.set_page_config(page_title="TuneScope", page_icon="🎧")

st.title("🎧 TuneScope")
st.caption("Prediksi Popularitas Lagu Spotify menggunakan Artificial Neural Network (ANN)")

# =========================
# Input judul lagu
# =========================
song_title = st.text_input("🔍 Masukkan Judul Lagu")

if st.button("Prediksi"):
    if song_title.strip() == "":
        st.warning("Masukkan judul lagu terlebih dahulu")
    else:
        # Cari lagu (case-insensitive)
        song = df[df["track_name"].str.lower() == song_title.lower()]

        if song.empty:
            st.error("Lagu tidak ditemukan di dataset")
        else:
            # Ambil fitur numerik
            features = song.select_dtypes(include=np.number)

            # Pastikan kolom target tidak ikut
            if "popularity_label" in features.columns:
                features = features.drop(columns=["popularity_label"])

            # Scaling
            features_scaled = scaler.transform(features)

            # Prediksi
            pred = model.predict(features_scaled)[0]

            # Output hasil
            if pred == 1:
                st.success("🔥 Lagu ini DIPREDIKSI POPULER")
            else:
                st.warning("🎼 Lagu ini DIPREDIKSI TIDAK POPULER")

            # Info tambahan (opsional tapi cakep)
            st.markdown("### ℹ️ Informasi Lagu")
            st.write(song[["track_name", "track_artist", "playlist_genre"]])
