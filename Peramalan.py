import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from google.cloud import firestore
from tensorflow.keras.models import load_model
from joblib import load 
import os

# Konfigurasi Halaman Wide
st.set_page_config(page_title="Forecasting Penumpang", layout="wide")

# Inisialisasi Firebase
db = firestore.Client.from_service_account_json("key.json")

# Load data
csv_path = 'data/data_penumpang-exel.csv'
df = pd.read_csv(csv_path)

# Load model dari file .h5
model_path = 'model/lstm_model.h5'
scaler_path = 'model/scaler.pkl'

if not os.path.exists(scaler_path) and os.path.exists(model_path):
    st.error(f"Model tidak ditemukan di: {model_path}")
    st.error(f"Scaler tidak ditemukan di: {scaler_path}")
    st.stop()

try:
    model = load_model(model_path)
    scaler = load(scaler_path)
except Exception as e:
    st.error(f"Gagal memuat: {str(e)}")
    st.stop()

# Transform data menggunakan scaler
data = df[['datang', 'berangkat']].values
data_scaled = scaler.transform(data)

look_back = 12
train_size = int(len(data_scaled) * 0.9)
train = data_scaled[:train_size]

# Custom CSS
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
    
    div.stButton > button {
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        width: 100%;
        transition: all 0.2s ease-in-out !important;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%) !important;
        box-shadow: 0 4px 12px rgba(21, 101, 192, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.title("Peramalan Jumlah Penumpang")
st.caption("Proyeksi arus penumpang datang dan berangkat 12 bulan ke depan menggunakan model LSTM.")

# Sidebar Input
st.sidebar.markdown("### ⚙️ Parameter Input")
st.sidebar.info("Masukkan 12 nilai historis terakhir yang dipisahkan koma (,).")

berangkat_input = st.sidebar.text_area(
    "12 bulan terakhir penumpang berangkat",
    placeholder="Contoh: 567, 743, 904, ...",
    height=90
)
datang_input = st.sidebar.text_area(
    "12 bulan terakhir penumpang datang",
    placeholder="Contoh: 735, 850, 977, ...",
    height=90
)
tombol_pred = st.sidebar.button('Jalankan Prediksi')

# Helper functions
def get_data(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

X_train, y_train = get_data(train, look_back)
if len(X_train) == 0:
    st.error("Data latih tidak valid. Kurangi look_back atau tambah data.")
    st.stop()

# Execution Logic
if tombol_pred:
    if not berangkat_input or not datang_input:
        st.error("Harap lengkapi input data penumpang datang dan berangkat.")
    else:
        try:
            berangkat_data = [float(i.strip()) for i in berangkat_input.split(',')]
            datang_data = [float(i.strip()) for i in datang_input.split(',')]
            
            if len(berangkat_data) != 12 or len(datang_data) != 12:
                st.error("Input harus tepat berjumlah 12 angka untuk masing-masing kategori.")
                st.stop()
                
            user_input = np.column_stack((datang_data, berangkat_data))
            user_input_scaled = scaler.transform(user_input)
            input_seq = user_input_scaled[-look_back:]
            
            user_predictions = []
            current_input = input_seq.copy()
            
            for _ in range(12):
                pred = model.predict(current_input.reshape(1, look_back, 2), verbose=0)
                user_predictions.append(pred[0])
                current_input = np.vstack((current_input[1:], pred))
            
            user_predictions = np.array(user_predictions)
            user_predictions_inv = scaler.inverse_transform(user_predictions)
            user_predictions_inv = np.round(user_predictions_inv, 2)

            months_user = [f"Bulan {i+1}" for i in range(12)]
            user_df = pd.DataFrame(user_predictions_inv, columns=["datang", "berangkat"])
            user_df["bulan"] = months_user

            # Simpan ke Firestore
            try:
                data_prediksi = {
                    "prediksi_berangkat": user_df['berangkat'].tolist(),
                    "prediksi_datang": user_df['datang'].tolist()
                }
                db.collection("forecasting").add(data_prediksi)
                st.success("Data prediksi berhasil disimpan ke Firebase!")
            except Exception as e:
                st.error(f"Gagal menyimpan data ke Firebase: {e}")

            # Ringkasan Metrik
            col_m1, col_m2, col_m3 = st.columns(3)
            total_datang = user_df['datang'].sum()
            total_berangkat = user_df['berangkat'].sum()
            col_m1.metric("Total Prediksi Datang", f"{total_datang:,.2f}")
            col_m2.metric("Total Prediksi Berangkat", f"{total_berangkat:,.2f}")
            col_m3.metric("Rata-rata Penumpang / Bulan", f"{(total_datang + total_berangkat)/24:,.2f}")

            st.write("")

            # Layout 2 Kolom (Grafik & Tabel)
            col_chart, col_table = st.columns([1.6, 1])

            with col_chart:
                with st.container(border=True):
                    st.markdown("#### Tren Peramalan vs Riwayat")
                    gabungan_df = pd.concat([df[['datang', 'berangkat']], user_df[['datang', 'berangkat']]], ignore_index=True)
                    
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

                    fig, ax = plt.subplots(figsize=(10, 5))
                    x_idx = gabungan_df.index.values
                    
                    x_smooth, y1_smooth = get_smooth_line(x_idx, gabungan_df['datang'].values)
                    _, y2_smooth = get_smooth_line(x_idx, gabungan_df['berangkat'].values)
                    
                    color_blue = '#4f46e5'
                    color_orange = '#f97316'
                    
                    ax.plot(x_smooth, y1_smooth, color=color_blue, linewidth=2.2, label="Datang")
                    ax.plot(x_smooth, y2_smooth, color=color_orange, linewidth=2.2, label="Berangkat")
                    ax.fill_between(x_smooth, y1_smooth, color=color_blue, alpha=0.12)
                    ax.fill_between(x_smooth, y2_smooth, color=color_orange, alpha=0.08)
                    ax.axvline(x=len(df)-0.5, color='#ef4444', linestyle='--', linewidth=1.5, alpha=0.8, label='Awal Prediksi')
                    
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#e5e7eb')
                    ax.spines['bottom'].set_color('#e5e7eb')
                    ax.xaxis.grid(True, color='#f3f4f6', linestyle='-', linewidth=1.2)
                    ax.yaxis.grid(True, color='#f3f4f6', linestyle='--', linewidth=0.8)
                    ax.set_axisbelow(True)
                    
                    ax.set_xlabel("Waktu (Bulan)", fontsize=10, color='#4b5563')
                    ax.set_ylabel("Jumlah Penumpang", fontsize=10, color='#4b5563')
                    ax.tick_params(colors='#4b5563', labelsize=9)
                    ax.legend(frameon=True, facecolor='white', edgecolor='#e5e7eb')
                    plt.tight_layout()
                    st.pyplot(fig)

            with col_table:
                with st.container(border=True):
                    st.markdown("#### Detail Angka 12 Bulan")
                    tabel_output = user_df[["bulan", "datang", "berangkat"]]
                    st.dataframe(tabel_output, use_container_width=True, hide_index=True, height=360)
                    st.download_button(
                        "Unduh Hasil (.csv)",
                        data=tabel_output.to_csv(index=False),
                        file_name="prediksi_penumpang.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                    
        except ValueError:
            st.error("Pastikan seluruh input berupa deret angka numerik yang dipisahkan tanda koma.")
        except Exception as e:
            st.error(f"Terjadi kendala pemrosesan: {str(e)}")
else:
    st.info("Silakan masukkan deret data 12 bulan terakhir pada sidebar di sisi kiri dan tekan tombol 'Jalankan Prediksi'.")