import streamlit as st
from google.cloud import firestore
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

db = firestore.Client.from_service_account_json("key.json")

docs = list(db.collection("forecasting").stream())

all_berangkat = []
all_datang = []


st.sidebar.write("Halaman ini digunakan untuk menampilkan hasil prediksi jumlah penumpang datang dan berangkat dengan menampilkan data asli dan data prediksi berdasarkan data input pengguna dan hasil dari model LSTM yang telah dilatih.")

if docs:
    st.success("Data berhasil ditampilkan")
    for doc in docs:
        data = doc.to_dict()
        pred_datang = data.get("prediksi_datang")
        pred_berangkat = data.get("prediksi_berangkat")
        
        try:
            # Parsing string menjadi list float
            datang_list = [float(i.strip()) for i in pred_datang.split(',')]
            berangkat_list = [float(i.strip()) for i in pred_berangkat.split(',')]

            all_datang.extend(datang_list)
            all_berangkat.extend(berangkat_list)
        
        except Exception as e:
            st.error(f"Gagal memproses dokumen {doc.id}: {e}")

    csv_path = 'data/data_penumpang-exel.csv'  # sesuaikan nama file CSV kamu
    df = pd.read_csv(csv_path)

    df_gabungan = pd.DataFrame({
        "datang": all_datang,
        "berangkat": all_berangkat
    })

    # Gabungkan langsung tanpa kolom 'bulan'
    df_total = pd.concat([df[['datang', 'berangkat']], df_gabungan[['datang', 'berangkat']]], ignore_index=True)

    # Tampilkan hasil gabungan sebagai grafik
    st.subheader("Data Asli dan Semua Prediksi")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_total.index, df_total["datang"], label="Datang")
    ax.plot(df_total.index, df_total["berangkat"], label="Berangkat")
    ax.axvline(x=len(df)-0.5, color='red', linestyle='--', label='Awal Prediksi')
    ax.set_xlabel("Data Yang Digunakan")
    ax.set_ylabel("Jumlah Penumpang")
    ax.legend()
    st.pyplot(fig)
    st.dataframe(df_total[["datang", "berangkat"]])
else:
    st.warning("Tidak ada data yang ditemukan")



