import streamlit as st
from google.cloud import firestore
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

db = firestore.Client.from_service_account_json("key.json")

docs = db.collection("forecasting").document("test").collection("manual_input").stream()
all_berangkat = []
all_datang = []

for doc in docs:
    data = doc.to_dict()
    if "prediksi_berangkat" in data and "prediksi_datang" in data:
        # Ubah string ke list float
        datang_list = [float(i.strip()) for i in data["prediksi_datang"].split(',')]
        berangkat_list = [float(i.strip()) for i in data["prediksi_berangkat"].split(',')]
        
        all_datang.extend(datang_list)
        all_berangkat.extend(berangkat_list)

csv_path = 'data/data_penumpang-exel.csv'  # sesuaikan nama file CSV kamu
df = pd.read_csv(csv_path)
df_gabungan = pd.DataFrame({
    "datang": all_datang,
    "berangkat": all_berangkat
})

# Gabungkan langsung tanpa kolom 'bulan'
df_total = pd.concat([df[['datang', 'berangkat']], df_gabungan[['datang', 'berangkat']]], ignore_index=True)

# Tampilkan hasil gabungan sebagai grafik
st.subheader("Gabungan Data Asli dan Semua Prediksi")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_total.index, df_total["datang"], label="Datang")
ax.plot(df_total.index, df_total["berangkat"], label="Berangkat")
ax.axvline(x=len(df)-0.5, color='red', linestyle='--', label='Awal Prediksi')
ax.set_xlabel("Waktu (bulan ke-)")
ax.set_ylabel("Jumlah Penumpang")
ax.legend()
st.pyplot(fig)