import streamlit as st
from google.cloud import firestore
import pandas as pd
import matplotlib.pyplot as plt


key_dict = st.secrets["gcp_service_account"]
db = firestore.Client.from_service_account_info(key_dict)

# Judul halaman
st.title("Visualisasi Data Prediksi Penumpang")

# Penjelasan di sidebar
st.sidebar.write("""
**Halaman Visualisasi Data Prediksi**

Halaman ini menampilkan hasil prediksi jumlah penumpang yang disimpan di Firebase. Data prediksi berasal dari input pengguna dan hasil pemodelan LSTM.


""")

# Ambil data dari Firestore
docs = list(db.collection("forecasting").stream())

if docs:
    st.success("Data berhasil diambil dari Firebase!")
    
    # Load data asli
    try:
        csv_path = 'data/data_penumpang-exel.csv'
        df = pd.read_csv(csv_path)
        
    except Exception as e:
        st.error(f"Gagal memuat data asli: {e}")
        st.stop()

    # Tampilkan semua prediksi
    for i, doc in enumerate(docs):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.subheader(f"Prediksi #{i+1}")
        with col2:
            # Tombol delete dengan styling
            if st.button("🗑️ Hapus", key=f"delete_{doc.id}", 
                        help=f"Hapus prediksi #{i+1}",
                        use_container_width=True):
                # Hapus dokumen dari Firestore
                db.collection("forecasting").document(doc.id).delete()
                st.success(f"Prediksi #{i+1} berhasil dihapus!")
                st.rerun()  # Refresh halaman untuk update tampilan
        
        try:
            data = doc.to_dict()
            # Langsung gunakan list tanpa split
            pred_datang = data["prediksi_datang"]
            pred_berangkat = data["prediksi_berangkat"]
            
            # Buat DataFrame untuk prediksi
            pred_df = pd.DataFrame({
                "datang": pred_datang,
                "berangkat": pred_berangkat
            })
            
            # Gabungkan dengan data asli
            combined_df = pd.concat([df[['datang', 'berangkat']], pred_df], ignore_index=True)
            
            # Tampilkan grafik
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(combined_df.index, combined_df["datang"], label="Datang")
            ax.plot(combined_df.index, combined_df["berangkat"], label="Berangkat")
            ax.axvline(x=len(df)-1, color='red', linestyle='--', label='Awal Prediksi')
            ax.set_title(f"Visualisasi Prediksi #{i+1}")
            ax.set_xlabel("Periode (bulan)")
            ax.set_ylabel("Jumlah Penumpang")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
            st.pyplot(fig)
            
            # Tampilkan data dalam tabel
            with st.expander(f"Lihat Data Prediksi #{i+1}"):
                st.dataframe(pred_df)
            
        except Exception as e:
            st.error(f"Error memproses dokumen {doc.id}: {str(e)}")
    # Tampilkan semua prediksi dalam satu grafik
    st.subheader("Gabungan Semua Prediksi")
    all_datang = []
    all_berangkat = []
    
    for doc in docs:
        try:
            data = doc.to_dict()
            all_datang.extend(data["prediksi_datang"])
            all_berangkat.extend(data["prediksi_berangkat"])
        except:
            continue
    
    if all_datang and all_berangkat:
        # Gabungkan semua data
        combined_all = pd.concat([
            df[['datang', 'berangkat']],
            pd.DataFrame({
                "datang": all_datang,
                "berangkat": all_berangkat
            })
        ], ignore_index=True)
        
        # Grafik gabungan
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(combined_all.index, combined_all["datang"], label="Datang")
        ax.plot(combined_all.index, combined_all["berangkat"], label="Berangkat")
        ax.axvline(x=len(df)-1, color='red', linestyle='--', label='Awal Prediksi')
        ax.set_title("Gabungan Semua Prediksi")
        ax.set_xlabel("Periode (bulan)")
        ax.set_ylabel("Jumlah Penumpang")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig)
        
        # Statistik
        st.write("**Statistik Gabungan Prediksi:**")
        col1, col2 = st.columns(2)
        col1.metric("Total Prediksi Datang", f"{sum(all_datang):,.0f}")
        col2.metric("Total Prediksi Berangkat", f"{sum(all_berangkat):,.0f}")
    else:
        st.warning("Tidak ada data prediksi yang valid untuk ditampilkan")
        
else:
    st.warning("Belum ada data prediksi yang tersimpan di Firebase")
    st.info("Lakukan prediksi melalui halaman utama untuk menyimpan data prediksi")