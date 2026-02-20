import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# === Load Dataset ===
file_path = '/content/drive/MyDrive/DATA/Co2.csv'
df_raw = pd.read_csv(file_path, header=[0, 1])
df_raw.columns = ['_'.join(col).strip() for col in df_raw.columns.values]
df_raw.rename(columns={'Country_Year': 'Year'}, inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

countries = ['Canada', 'India', 'Japan', 'Russia']
markers = ['o', 's', '^', 'D']

n_lags = 5
n_test = 6          # calibration window
n_future = 12

plt.figure(figsize=(15, 8))

# === Identify CO2 Columns ===
co2_columns = {}
for col in df_raw.columns:
    for country in countries:
        if country in col and 'CO2' in col:
            co2_columns[country] = col

# === Loop Through Countries ===
for i, country in enumerate(countries):

    column = co2_columns[country]
    ts = df_raw[['Year', column]].dropna()
    ts.columns = ['Year', 'CO2']
    ts.set_index('Year', inplace=True)

    # === Scaling ===
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(ts[['CO2']])

    # === Supervised Data ===
    X, y = [], []
    for j in range(n_lags, len(scaled)):
        X.append(scaled[j-n_lags:j])
        y.append(scaled[j])
    X, y = np.array(X), np.array(y)

    # === Train / Calibration Split ===
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_cal, y_cal = X[-n_test:], y[-n_test:]

    # === GRU Model ===
    model = Sequential([
        GRU(32, input_shape=(n_lags, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=200, verbose=0)

    # === Calibration Predictions ===
    y_cal_pred = model.predict(X_cal)
    y_cal_true = scaler.inverse_transform(y_cal)
    y_cal_pred = scaler.inverse_transform(y_cal_pred)

    # === Conformal Residuals ===
    residuals = np.abs(y_cal_true - y_cal_pred).flatten()

    q80 = np.quantile(residuals, 0.90)
    q95 = np.quantile(residuals, 0.975)

    # === Retrain on Full Data ===
    model.fit(X, y, epochs=200, verbose=0)

    # === Future Forecast ===
    input_seq = scaled[-n_lags:].reshape(1, n_lags, 1)
    future_scaled = []

    for _ in range(n_future):
        pred = model.predict(input_seq)[0, 0]
        future_scaled.append(pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[pred]]], axis=1)

    future_mean = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    future_years = np.arange(ts.index.max() + 1, ts.index.max() + n_future + 1)

    # === Prediction Intervals ===
    lower_80 = future_mean - q80
    upper_80 = future_mean + q80

    lower_95 = future_mean - q95
    upper_95 = future_mean + q95

    # === Plot ===
    plt.plot(ts.index, ts['CO2'], marker=markers[i], label=f'{country} Actual')
    plt.plot(future_years, future_mean, linestyle='--', label=f'{country} Forecast')

    plt.fill_between(future_years, lower_95, upper_95, alpha=0.15)
    plt.fill_between(future_years, lower_80, upper_80, alpha=0.30)

    # === Forecast Table ===
    forecast_table = pd.DataFrame({
        'Year': future_years,
        'Mean Forecast': future_mean,
        'Lower 80%': lower_80,
        'Upper 80%': upper_80,
        'Lower 95%': lower_95,
        'Upper 95%': upper_95
    })

    print(f"\nForecast with Prediction Intervals for {country}:")
    print(forecast_table.to_string(index=False))

# === Final Plot Formatting ===
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions")
plt.title("GRU Forecast with 80% and 95% Prediction Intervals (Conformal Prediction)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
