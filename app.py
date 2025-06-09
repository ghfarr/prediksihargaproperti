
import streamlit as st
import pickle
import numpy as np

# Load model dan label encoders
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders.pkl', 'rb') as f:
    label_encoders = pickle.load(f)

# UI Streamlit
st.set_page_config(page_title="Prediksi Harga Properti", layout="centered")
st.title("🏡 Prediksi Harga Properti")
st.write("Aplikasi cerdas untuk memprediksi harga properti berdasarkan fitur yang Anda input.")

# Input pengguna
luas_bangunan = st.number_input("Luas Bangunan (GrLivArea)", min_value=10, max_value=1000, step=1)
luas_tanah = st.number_input("Luas Tanah (LotArea)", min_value=10, max_value=20000, step=1)
kamar_tidur = st.number_input("Jumlah Kamar Tidur (BedroomAbvGr)", min_value=1, max_value=10, step=1)

# Pilihan lokasi (Neighborhood)
neighborhood = st.selectbox("Lokasi Properti (Neighborhood)", label_encoders['Neighborhood'].classes_)

# Prediksi
if st.button("Prediksi Harga"):
    try:
        neighborhood_encoded = label_encoders['Neighborhood'].transform([neighborhood])[0]
        input_data = np.array([[luas_bangunan, luas_tanah, kamar_tidur, neighborhood_encoded]])
        harga_prediksi = model.predict(input_data)[0]

       st.success(f"💰 Estimasi Harga Properti: Rp {int(harga_prediksi):,}".replace(",", ".") + ',000')
    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")
