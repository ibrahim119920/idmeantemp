import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# Membaca data dengan kolom 'Year' dan 'Mean'
url = "https://raw.githubusercontent.com/ibrahim119920/idmeantemp/b58834964ecd8664a999d26506c5835d16bc373d/temperature_data.csv"
data = pd.read_csv(url)

# Menjadikan 'Year' sebagai index dan memastikan format index berupa datetime untuk analisis time series
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Split data menjadi train dan test (10 tahun terakhir untuk test)
train = data.iloc[:-20]  # 112 data untuk train
test = data.iloc[-20:]   # 10 data untuk test

# Model ARIMA (karena data tahunan, tidak ada faktor musiman yang kuat, jadi tetap gunakan ARIMA biasa)
arima_model = ARIMA(train, order=(100,1,1))  # order disesuaikan jika hasil kurang memuaskan
arima_result = arima_model.fit()

# Forecast ARIMA
arima_forecast = arima_result.forecast(steps=20)

# Evaluasi hasil ARIMA
mse_arima = mean_squared_error(test, arima_forecast)
mae_arima = mean_absolute_error(test, arima_forecast)

print(f"ARIMA - Mean Squared Error: {mse_arima}")
print(f"ARIMA - Mean Absolute Error: {mae_arima}")

# Model SARIMAX (seasonal_order bisa diubah atau dihilangkan karena data tahunan tidak selalu musiman)
sarimax_model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,25))  # tanpa musiman
sarimax_result = sarimax_model.fit()

# Forecast SARIMAX
sarimax_forecast = sarimax_result.forecast(steps=20)

# Evaluasi hasil SARIMAX
mse_sarimax = mean_squared_error(test, sarimax_forecast)
mae_sarimax = mean_absolute_error(test, sarimax_forecast)

print(f"SARIMAX - Mean Squared Error: {mse_sarimax}")
print(f"SARIMAX - Mean Absolute Error: {mae_sarimax}")

# Plotting hasil
plt.figure(figsize=(10, 5))
plt.plot(train.index, train['Mean'], label='Train')
plt.plot(test.index, test['Mean'], label='Test', color='orange')
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='green', linestyle='--')
plt.plot(test.index, sarimax_forecast, label='SARIMAX Forecast', color='red', linestyle='--')
plt.legend()
plt.title('ARIMA and SARIMAX Forecasting')
plt.xlabel('Year')
plt.ylabel('Mean Temperature')
plt.show()
