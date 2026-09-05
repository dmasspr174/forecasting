import streamlit as st
from google.cloud import firestore
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi Halaman Wide
st.set_page_config(page_title="Riwayat Prediksi", layout="wide")

# Inisialisasi Firebase
db = firestore.Client.from_service_account_json("key.json")

def draw_smooth_chart(combined_df, title, split_index, figsize=(10, 4.2)):
    import numpy as np
    
    fig, ax = plt.subplots(figsize=figsize)
    x = combined_df.index.values
    y1 = combined_df["datang"].values
    y2 = combined_df["berangkat"].values
    
    def get_smooth_line(x, y, factor=15, window_size=15):
        if len(x) < 3:
            return x, y
        x_fine = np.linspace(x[0], x[-1], len(x) * factor)
        y_fine = np.interp(x_fine, x, y)
        window = np.arange(1, window_size // 2 + 1)
        window = np.concatenate([window, window[::-1]])
        window = window / window.sum()
        pad_size = len(window) // 2
        y_padded = np.pad(y_fine, pad_size, mode='edge')
        y_smooth = np.convolve(y_padded, window, mode='valid')
        y_smooth = y_smooth[:len(x_fine)]
        return x_fine, y_smooth

    x_smooth, y1_smooth = get_smooth_line(x, y1)
    _, y2_smooth = get_smooth_line(x, y2)
    
    color_blue = '#4f46e5'
    color_orange = '#f97316'
    
    ax.plot(x_smooth, y1_smooth, color=color_blue, linewidth=2.2, label="Datang")
    ax.plot(x_smooth, y2_smooth, color=color_orange, linewidth=2.2, label="Berangkat")
    ax.fill_between(x_smooth, y1_smooth, color=color_blue, alpha=0.12)
    ax.fill_between(x_smooth, y2_smooth, color=color_orange, alpha=0.08)
    
    if split_index is not None:
        ax.axvline(x=split_index, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.8, label='Awal Prediksi')
        
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.xaxis.grid(True, color='#f3f4f6', linestyle='-', linewidth=1.2)
    ax.yaxis.grid(True, color='#f3f4f6', linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Periode (Bulan)", fontsize=9, color='#4b5563')
    ax.set_ylabel("Jumlah Penumpang", fontsize=9, color='#4b5563')
    ax.tick_params(colors='#4b5563', labelsize=9)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
    plt.tight_layout()
    return fig

# Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif !important;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    div[data-testid="stMetric"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 14px 20px;
        border-radius: 10px;
    }
    
    /* Tombol Hapus Merah */
    div.stButton > button {
        background-color: #ef4444 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 600 !important;
        width: 100%;
        transition: background-color 0.2s ease !important;
    }
    div.stButton > button:hover {
        background-color: #dc2626 !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Informasi")
    st.info("Halaman ini menyajikan seluruh log hasil peramalan yang tersimpan dalam koleksi Firestore.")

# Header
st.title("Visualisasi & Riwayat Prediksi")
st.caption("Eksplorasi data hasil komputasi LSTM yang tersimpan secara terpusat di Firestore.")

docs = list(db.collection("forecasting").stream())

if docs:
    try:
        csv_path = 'data/data_penumpang-exel.csv'
        df = pd.read_csv(csv_path)
    except Exception as e:
        st.error(f"Gagal memuat data historis: {e}")
        st.stop()

    tab_sesi, tab_gabungan = st.tabs(["📌 Detail Tiap Sesi", "📊 Gabungan Seluruh Prediksi"])

    with tab_sesi:
        # Pilihan dropdown untuk menggantikan list memanjang vertikal
        list_nama_sesi = [f"Prediksi #{i+1} (ID: {doc.id[:6]}...)" for i, doc in enumerate(docs)]
        selected_index = st.selectbox("Pilih Sesi Prediksi:", range(len(docs)), format_func=lambda x: list_nama_sesi[x])
        
        doc_terpilih = docs[selected_index]
        
        try:
            data = doc_terpilih.to_dict()
            pred_datang = data["prediksi_datang"]
            pred_berangkat = data["prediksi_berangkat"]
            
            pred_df = pd.DataFrame({
                "bulan": [f"Bulan {k+1}" for k in range(len(pred_datang))],
                "datang": pred_datang,
                "berangkat": pred_berangkat
            })
            
            combined_df = pd.concat([df[['datang', 'berangkat']], pred_df[['datang', 'berangkat']]], ignore_index=True)
            
            st.write("")
            col_chart_sesi, col_detail_sesi = st.columns([1.6, 1])
            
            with col_chart_sesi:
                with st.container(border=True):
                    fig = draw_smooth_chart(combined_df, f"Tren Visualisasi Prediksi #{selected_index+1}", len(df)-1)
                    st.pyplot(fig)
            
            with col_detail_sesi:
                with st.container(border=True):
                    st.markdown(f"#### Tabel Prediksi #{selected_index+1}")
                    st.dataframe(pred_df, use_container_width=True, hide_index=True, height=270)
                    
                    st.divider()
                    if st.button("🗑️ Hapus Sesi Ini", key=f"del_{doc_terpilih.id}"):
                        db.collection("forecasting").document(doc_terpilih.id).delete()
                        st.success(f"Sesi Prediksi #{selected_index+1} berhasil dihapus.")
                        st.rerun()

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses data dokumen: {str(e)}")

    with tab_gabungan:
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
            col_m1, col_m2 = st.columns(2)
            col_m1.metric("Total Prediksi Datang (Akumulasi)", f"{sum(all_datang):,.0f}")
            col_m2.metric("Total Prediksi Berangkat (Akumulasi)", f"{sum(all_berangkat):,.0f}")
            
            st.write("")
            with st.container(border=True):
                combined_all = pd.concat([
                    df[['datang', 'berangkat']],
                    pd.DataFrame({
                        "datang": all_datang,
                        "berangkat": all_berangkat
                    })
                ], ignore_index=True)
                
                fig_all = draw_smooth_chart(combined_all, "Akumulasi Keseluruhan Riwayat Prediksi", len(df)-1, figsize=(11, 4.5))
                st.pyplot(fig_all)
        else:
            st.warning("Belum ada data prediksi yang valid untuk diakumulasikan.")
else:
    st.warning("Belum ada riwayat peramalan yang tersimpan di Firebase.")
    st.info("Jalankan peramalan pada halaman utama terlebih dahulu untuk menyimpan catatan proyeksi.")