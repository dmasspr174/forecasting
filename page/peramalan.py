import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from google.cloud import firestore
from tensorflow.keras.models import load_model  # Import fungsi untuk load model .h5
import os

# Inisialisasi Firebase
db = firestore.Client.from_service_account_json("key.json")

# Load data
csv_path = 'data/data_penumpang-exel.csv'
df = pd.read_csv(csv_path)

# Preprocessing
data = df[['datang', 'berangkat']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

look_back = 12
train_size = int(len(data_scaled) * 0.9)
train = data_scaled[:train_size]

# UI Styles
st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;  
        padding: 10px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar UI
st.sidebar.subheader("Peramalan Jumlah Penumpang")
berangkat_input = st.sidebar.text_input(
    "12 bulan terakhir penumpang berangkat",
    placeholder='Nilai dipisahkan dengan koma'
)
datang_input = st.sidebar.text_input(
    "12 bulan terakhir penumpang datang",
    placeholder='Nilai dipisahkan dengan koma'
)
tombol_pred = st.sidebar.button('Prediksi')

# Main UI
st.header("Prediksi Jumlah Penumpang Datang & Berangkat")

# Load model dari file .h5
model_path = 'model/lstm_model.h5'  # Path ke model .h5
if not os.path.exists(model_path):
    st.error(f"Model tidak ditemukan di: {model_path}")
    st.stop()

try:
    model = load_model(model_path)
    st.success("Model berhasil dimuat")
except Exception as e:
    st.error(f"Gagal memuat model: {str(e)}")
    st.stop()

# Fungsi untuk membuat dataset dengan sliding window
def get_data(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

# Prepare training data untuk validasi
X_train, y_train = get_data(train, look_back)
if len(X_train) == 0:
    st.error("Data latih tidak valid. Kurangi look_back atau tambah data.")
    st.stop()

# Prediction logic
if tombol_pred:
    if not berangkat_input or not datang_input:
        st.error("Harap isi kedua input.")
    else:
        try:
            # Parse input
            berangkat_data = [float(i.strip()) for i in berangkat_input.split(',')]
            datang_data = [float(i.strip()) for i in datang_input.split(',')]
            
            if len(berangkat_data) != 12 or len(datang_data) != 12:
                st.error("Input harus terdiri dari 12 angka untuk masing-masing kolom")
                st.stop()
                
            # Normalize input
            user_input = np.column_stack((datang_data, berangkat_data))
            user_input_scaled = scaler.transform(user_input)
            input_seq = user_input_scaled[-look_back:]  # Ambil data sesuai look_back
            
            # Generate predictions
            user_predictions = []
            current_input = input_seq.copy()
            
            for _ in range(12):
                # Reshape untuk model: (1, look_back, 2)
                pred = model.predict(current_input.reshape(1, look_back, 2), verbose=0)
                user_predictions.append(pred[0])
                # Update input: buang data paling awal, tambahkan prediksi
                current_input = np.vstack((current_input[1:], pred))
            
            # Inverse transform predictions
            user_predictions = np.array(user_predictions)
            user_predictions_inv = scaler.inverse_transform(user_predictions)
            
            # Pembulatan ke 2 angka di belakang koma
            user_predictions_inv = np.round(user_predictions_inv, 2)
            
            # Create result DataFrame
            months_user = [f"Bulan {i+1}" for i in range(12)]
            user_df = pd.DataFrame(user_predictions_inv, columns=["datang", "berangkat"])
            user_df["bulan"] = months_user
            
            # Display results
            st.subheader("Prediksi 12 Bulan Kedepan")
            st.dataframe(user_df[["bulan", "datang", "berangkat"]])
            
            # Save to Firestore
            try:
                # Simpan sebagai list float yang sudah dibulatkan
                data_prediksi = {
                    "prediksi_berangkat": user_df['berangkat'].tolist(),
                    "prediksi_datang": user_df['datang'].tolist()
                }
                db.collection("forecasting").add(data_prediksi)
                st.success("Data berhasil disimpan di Firebase!")
            except Exception as e:
                st.error(f"Gagal menyimpan data: {e}")
            
            # Plot combined data
            gabungan_df = pd.concat([df[['datang', 'berangkat']], user_df[['datang', 'berangkat']]], ignore_index=True)
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(gabungan_df.index, gabungan_df['datang'], label='Datang')
            ax.plot(gabungan_df.index, gabungan_df['berangkat'], label='Berangkat')
            ax.axvline(x=len(df)-0.5, color='red', linestyle='--', label='Awal Prediksi')
            ax.set_xlabel('Waktu (bulan)')
            ax.set_ylabel('Jumlah Penumpang')
            ax.legend()
            st.pyplot(fig)
            
        except ValueError:
            st.error("Pastikan input berupa angka yang dipisahkan koma")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")