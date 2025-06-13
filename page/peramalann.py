import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import  ReduceLROnPlateau
import matplotlib.pyplot as plt
from google.cloud import firestore

# Fungsi untuk membuat dataset dengan sliding window
def get_data(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

# Fungsi untuk membangun model
def build_model(input_shape, units=64, dropout_rate=0.3, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(2))
    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

#inisialisasi database firebase 
db = firestore.Client.from_service_account_json("key.json")

# Load CSV dari folder lokal
csv_path = 'data/data_penumpang-exel.csv'  # sesuaikan nama file CSV kamu
df = pd.read_csv(csv_path)

data = df[['datang', 'berangkat']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

look_back = 12
train_size = int(len(data_scaled) * 0.9)
train = data_scaled[:train_size]

X_train, y_train = get_data(train, look_back)
X_train = X_train.reshape(X_train.shape[0], look_back, 2)

# Dapatkan data train
X_train, y_train = get_data(train, look_back)

# Validasi hasil get_data
if len(X_train) == 0 or len(y_train) == 0:
    st.error("Data train tidak valid. Cek data input dan parameter look_back.")
    st.stop()

# Reshape untuk LSTM
X_train = X_train.reshape(X_train.shape[0], look_back, 2)

model = build_model((look_back, 2))
model.fit(
    X_train, y_train,
    epochs=400,
    batch_size=16,
    verbose=0,
    callbacks=[
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-6),
    ]
)
# Input manual dari pengguna (12 data terakhir)
st.sidebar.subheader("Peramalan Jumlah Penumpang yang datang dan berangkat")
berangkat_input = st.sidebar.text_input("Masukkan 12 bulan penumpanh berangkat")
datang_input = st.sidebar.text_input("Masukkan 12 bulan penumpanh berangkat datang")

st.markdown("""
    <style>
    div.stButton > button {
        width: 100%;  
        padding: 10px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

tombol_pred = st.sidebar.button('Prediksi')


st.header("Prediksi Jumlah Penumpang Datang & Berangkat")
if berangkat_input and datang_input:
    try:
        berangkat_data = [float(i.strip()) for i in berangkat_input.split(',')]
        datang_data = [float(i.strip()) for i in datang_input.split(',')]
    except:
        st.error("Masukan tidak valid. Pastikan semua input adalah angka.")
        st.stop()

    if len(berangkat_data) != 12 or len(datang_data) != 12:
        st.error("Input harus terdiri dari tepat 12 angka untuk masing-masing datang dan berangkat.")
        st.stop()

    if tombol_pred :
        # Normalisasi input manual menggunakan scaler dari data latih
        user_input_scaled = scaler.transform(np.column_stack((datang_data, berangkat_data)))
        input_seq = user_input_scaled[-12:]  # urutan [datang, berangkat]

        user_predictions = []
        current_input = input_seq.copy()
        for _ in range(12):
            pred = model.predict(current_input.reshape(1, 12, 2), verbose=0)
            user_predictions.append(pred[0])
            current_input = np.vstack((current_input[1:], pred))

        user_predictions = np.array(user_predictions)
        user_predictions_inv = scaler.inverse_transform(user_predictions)
        months_user = [f"Bulan {i+1}" for i in range(12)]
        user_df = pd.DataFrame(user_predictions_inv, columns=["datang", "berangkat"])
        user_df["bulan"] = months_user
        st.subheader("Prediksi Berdasarkan Input Manual")
        st.dataframe(user_df[["bulan", "datang", "berangkat"]])
        pred_berangkat = ",".join(map(str,user_df['berangkat'].tolist()))
        pred_datang = ",".join(map(str, user_df['datang'].tolist()))
        print(user_df[['datang','berangkat']])
        print(pred_berangkat)
        print(pred_datang)
        fig_user, ax_user = plt.subplots(figsize=(10, 4))
        ax_user.plot(months_user, user_df["datang"], label="Datang", marker='o')
        ax_user.plot(months_user, user_df["berangkat"], label="Berangkat", marker='o')
        ax_user.set_title("Prediksi Penumpang yang berangkat dan datang")
        ax_user.set_xlabel("Bulan")
        ax_user.set_ylabel("Jumlah Penumpang")
        ax_user.legend()
        st.pyplot(fig_user)
        
        # Buat data yang akan disimpan dalam firebase
        data_prediksi = {
            "prediksi_berangkat": pred_berangkat,
            "prediksi_datang": pred_datang
        }

        # Simpan ke koleksi 'forecasting' > dokumen 'test' > subkoleksi 'manual_input'
        db.collection("forecasting").document("test").collection("manual_input").add(data_prediksi)
        st.success("Prediksi berhasil disimpan ke Firestore.")

        pred_df = user_df
        # Langsung tampilkan gabungan data dan grafik setelah prediksi
        gabungan_df = pd.concat([df[['datang', 'berangkat']], pred_df[['datang', 'berangkat']]], ignore_index=True)
        st.subheader("Gabungan Data Asli dan Prediksi")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(gabungan_df.index, gabungan_df['datang'], label='Datang')
        ax.plot(gabungan_df.index, gabungan_df['berangkat'], label='Berangkat')
        ax.axvline(x=len(df)-0.5, color='red', linestyle='--', label='Awal Prediksi')
        ax.set_xlabel('Waktu (bulan)')
        ax.set_ylabel('Jumlah Penumpang')
        ax.legend()
        st.pyplot(fig)


