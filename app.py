import streamlit as st
# JUDUL =================================================
st.set_page_config(page_title = "Analisis Tren Berdasarkan Nilai Close dan Volatilitas Harga Saham", layout="wide")
st.title("Analisis Tren Berdasarkan Nilai Close dan Volatilitas Harga Saham : Studi Kasus Telkom Indonesia (TLKM.JK)")

from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model


# Load Model =========================================
@st.cache_resource
def load_pred_model():
    return load_model("14D-forecasting.keras")
model = load_pred_model()

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


# Define Variable =======================================

start_time = datetime(year=2024, month=4, day=1)
end_time = datetime.now()
stock = "TLKM.JK"

df = yf.download(
                  tickers = stock,
                  start = start_time,
                  end = end_time,
                  interval="1d", #Interval 1 Day
                  group_by="ticker",
                  auto_adjust = True,
                  progress = False
)

st.success(f"Pembaruan Terakhir / Waktu Server (UTC+0): {datetime.now().strftime('%d-%m-%Y %H:%M')}")
st.info(f"Data Kemarin (sebelum hari ini) yang tersedia di Yfinance : {(df.index[-1]).date()}")

st.subheader("Informasi hari ini")
# ======================= CONTAINER 1 ============================
card1, card2, card3, card4, card5 = st.columns(5)

with card1:
    st.metric(
              "Open",
              df[("TLKM.JK", "Open")].iloc[-1]
              )

with card2:
    st.metric(
              "High",
              round(df[("TLKM.JK", "High")].iloc[-1], 2)
             )

with card3:
    st.metric(
              "Low",
              df[("TLKM.JK", "Low")].iloc[-1]
              )
    
with card4:
    st.metric(
              "Volatilitas 30 Hari Terakhir",
              f"{round(df[('TLKM.JK', 'Close')].pct_change().rolling(30).std().iloc[-1] * 100, 2)}%" 
            )

with card5:
    st.metric(
              "Return Harian",
              f"{round(df[('TLKM.JK', 'Close')].pct_change().iloc[-1] * 100,2)}%"
            )
st.divider()

# ======================== GRAFIK =============================

st.subheader("Grafik Moving Average")
plt.figure(figsize=(12,5))
  
sns.lineplot(
  data=df[(stock, 'Close')],
  label=stock,
  color="red",
  )

sns.lineplot(
  data= df[(stock, 'Close')].rolling(20).mean(),
  label="MA_20",
  color="green",
  )

sns.lineplot(
  data= df[(stock, 'Close')].rolling(50).mean(),
  label="MA_50",
  color="black",
  )

sns.lineplot(
  data= df[(stock, 'Close')].rolling(200).mean(),
  label="MA_200",
  color="purple",
  )

plt.xlim(pd.Timestamp('2024-04-01'))

plt.title("Close Price History")
plt.xlabel("Tanggal")
plt.ylabel("Close Price")
st.pyplot(plt.gcf())

plt.figure(figsize=(12,5))  
close_price = df[[(stock, "Close")]].values
scaled_data = scaler.transform(close_price)
pred_14 = scaled_data[-14:].reshape((1, 14, 1))

pred_result = []

for i in range(14):
  pred = model.predict(pred_14, verbose=0)[0][0]
  pred_result.append(pred)

  pred_14 = np.append(
                      pred_14[:, 1:, :],
                      np.array([[[pred]]]),
                      axis=1
                      )


# Grafik Prediksi ==========================================
prediction_result_14D = scaler.inverse_transform(np.array(pred_result).reshape(-1, 1))
last_date = df.index[-1]
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq="D")
df_last_forecast = pd.DataFrame(prediction_result_14D, index=forecast_dates, columns=[(stock, "Close_Pred")])

st.subheader("Grafik Prediksi 14 Hari Ke Depan")

plt.figure(figsize=(10,5))
plot_pred = sns.lineplot(
                        data=df_last_forecast[(stock, "Close_Pred")],
                        color="blue",
                        marker="s",
                        )
plt.title("Prediksi")
plt.xlabel("Tanggal")
plt.ylabel("Close Price")
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt.gcf())
st.divider()

st.subheader("Galat (Akurasi) Prediksi dengan Aktual Kemarin")
st.info(
    f"Tanggal Kemarin : {(datetime.now() - pd.Timedelta(days=1)).date()}  \n"
    f"Data terakhir yang tersedia di YFinance : {df.index[-1].date()}"
)

card5, card6, card7 = st.columns(3)

with card5:
    st.metric(
              "Nilai Aktual YFinance Close Kemarin",
              df[('TLKM.JK', 'Low')].iloc[-1]
              )

with card6:
    st.metric(
              "Nilai Prediksi Close Kemarin",
              round(df_last_forecast[(stock, 'Close_Pred')].iloc[0],2)
              )

with card7:
    st.metric(
              "Galat (%)",
              round(((df_last_forecast[(stock, 'Close_Pred')].iloc[0] - df[('TLKM.JK', 'Low')].iloc[-1]) * 100 /df[('TLKM.JK', 'Low')].iloc[-2]),2)
              )

st.warning("""
Data YFinance TIDAK DAPAT menampilkan data saat akhir perkan / hari libur nasional, sehingga data di atas dapat tidak akurat jika kemarin atau kemarin lusa adalah hari libur!
\nSilahkan lihat informasi tanggal kemarin dengan tanggal kemarin yang terdapat pada Yfinance di atas.            
           """)

st.divider()
st.subheader("Volatilitas dan Exploratory Data Analysis (EDA)")
st.success(f"Tanggal Mulai : {(df.index[0]).date()}")
st.error(f"Tanggal Berakhir : {(df.index[-1]).date()}")
left, right = st.columns([3, 1])
with left:
    vol = df[("TLKM.JK", "Close")].pct_change().rolling(30).std()
    plt.figure(figsize=(6,5))
    plot_pred = sns.lineplot(
                        x = vol.index,
                        y = vol.values,
                        color="blue",
                        )
    plt.title("Volatilitas")
    plt.xlabel("Tanggal")
    plt.ylabel("Volatilitas")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(plt.gcf())
with right:
    ret = df[('TLKM.JK', 'Close')].pct_change() * 100
    st.metric(
              "Rata-Rata Close Tahunan",
              round(df[('TLKM.JK', 'Close')].mean(), 2) 
            )
    st.metric(
              "Rata-Rata Open Tahunan",
              round(df[('TLKM.JK', 'Open')].mean(), 2) 
            )
    st.metric(
              "Return Tertinggi",
              f"{round(ret.max(), 2)}%" 
            )
    st.metric(
              "Return Terendah",
              f"{round(ret.min(), 2)}%" 
            )   
