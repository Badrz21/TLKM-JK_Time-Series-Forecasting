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

st.write(f"Pembaruan Terakhir: {datetime.now().date()}")

st.subheader("Informasi hari ini")
# ======================= CONTAINER 1 ============================
card1, card2, card3, card4, card5 = st.columns(4)

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
              "Volatilitas 30 Hari",
              round(df[("TLKM.JK", "Close")].rolling(30).std().iloc[-30], 2))

with card5:
    st.metric(
              "Close Kemarin",
              round(df[("TLKM.JK", "Close")].rolling(30).std().iloc[-2], 2))
st.divider()

# ======================== GRAFIK =============================

st.subheader("Grafik Moving Average 15 Hari")
plt.figure(figsize=(12,5))
  
sns.lineplot(
  data=df[(stock, 'Close')],
  label=stock,
  color="red",
  marker="s"
  )

sns.lineplot(
  data= df[(stock, 'Close')].rolling(15).mean(),
  label="Moving Average",
  color="green",
  marker="s"
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
