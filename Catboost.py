# =========================================================
# CatBoost CO2 Forecasting with 80% & 95% Prediction Intervals
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import (
    mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
    explained_variance_score,
    max_error
)

import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Parameters
# ---------------------------
file_path = '/content/drive/MyDrive/DATA/Co2.csv'

countries = ['Canada', 'India', 'Japan', 'Russia']
markers = ['o', 's', '^', 'D']

n_test = 12
n_future = 12
n_lags = 5

n_bootstrap = 1000
pi_levels = {
    "80": (10, 90),
    "95": (2.5, 97.5)
}

metrics_dict = {}

# ---------------------------
# Load Dataset
# ---------------------------
df_raw = pd.read_csv(file_path, header=[0, 1])
df_raw.columns = ['_'.join(col).strip() for col in df_raw.columns.values]
df_raw.rename(columns={'Country_Year': 'Year'}, inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# ---------------------------
# Identify CO2 Columns
# ---------------------------
co2_columns = {}
for col in df_raw.columns:
    for country in countries:
        if country in col and 'CO2' in col:
            co2_columns[country] = col

# ---------------------------
# Lag Feature Creator
# ---------------------------
def create_lag_features(series, n_lags):
    df = pd.DataFrame({'y': series})
    for i in range(1, n_lags + 1):
        df[f'lag_{i}'] = df['y'].shift(i)
    return df.dropna()

# ---------------------------
# Plot Setup
# ---------------------------
plt.figure(figsize=(15, 7))

# =========================================================
# Main Loop
# =========================================================
for i, country in enumerate(countries):

    if country not in co2_columns:
        continue

    print(f"\n================ {country} =================")

    # ---------------------------
    # Extract Series
    # ---------------------------
    ts = df_raw[['Year', co2_columns[country]]].dropna()
    ts.columns = ['Year', 'CO2']
    ts.set_index('Year', inplace=True)

    # ---------------------------
    # Supervised Transformation
    # ---------------------------
    df_lag = create_lag_features(ts['CO2'], n_lags)

    train = df_lag.iloc[:-n_test]
    test = df_lag.iloc[-n_test:]

    X_train, y_train = train.drop(columns='y'), train['y']
    X_test, y_test = test.drop(columns='y'), test['y']

    # ---------------------------
    # CatBoost Model
    # ---------------------------
    model = CatBoostRegressor(
        iterations=500,
        depth=6,
        learning_rate=0.05,
        loss_function='RMSE',
        random_seed=42,
        verbose=False
    )
    model.fit(X_train, y_train)

    # ---------------------------
    # Test Prediction
    # ---------------------------
    y_pred = model.predict(X_test)

    # ---------------------------
    # Residuals (Bootstrap)
    # ---------------------------
    train_pred = model.predict(X_train)
    residuals = y_train.values - train_pred

    # ---------------------------
    # Bootstrap Prediction Intervals (Test)
    # ---------------------------
    boot_preds_test = []
    for _ in range(n_bootstrap):
        noise = np.random.choice(residuals, size=len(y_pred), replace=True)
        boot_preds_test.append(y_pred + noise)
    boot_preds_test = np.array(boot_preds_test)

    test_pi = {}
    for level, (lq, uq) in pi_levels.items():
        test_pi[level] = (
            np.percentile(boot_preds_test, lq, axis=0),
            np.percentile(boot_preds_test, uq, axis=0)
        )

    # ---------------------------
    # Recursive Future Forecast
    # ---------------------------
    last_lags = X_test.iloc[-1].values
    future_preds = []

    for _ in range(n_future):
        pred = model.predict(last_lags.reshape(1, -1))[0]
        future_preds.append(pred)
        last_lags = np.roll(last_lags, 1)
        last_lags[0] = pred

    future_preds = np.array(future_preds)
    future_years = np.arange(ts.index.max() + 1,
                             ts.index.max() + n_future + 1)

    # ---------------------------
    # Bootstrap PI (Future)
    # ---------------------------
    boot_preds_future = []
    for _ in range(n_bootstrap):
        noise = np.random.choice(residuals, size=n_future, replace=True)
        boot_preds_future.append(future_preds + noise)
    boot_preds_future = np.array(boot_preds_future)

    future_pi = {}
    for level, (lq, uq) in pi_levels.items():
        future_pi[level] = (
            np.percentile(boot_preds_future, lq, axis=0),
            np.percentile(boot_preds_future, uq, axis=0)
        )

    # ---------------------------
    # Plot
    # ---------------------------
    plt.plot(ts.index, ts['CO2'], marker=markers[i],
             label=f'{country} Actual')

    plt.plot(test.index, y_pred, linestyle='--',
             marker=markers[i], label=f'{country} Test Forecast')

    plt.fill_between(test.index,
                     test_pi["95"][0], test_pi["95"][1],
                     color='gray', alpha=0.15)

    plt.fill_between(test.index,
                     test_pi["80"][0], test_pi["80"][1],
                     color='gray', alpha=0.30)

    plt.plot(future_years, future_preds, linestyle='dotted',
             marker=markers[i], label=f'{country} Future Forecast')

    plt.fill_between(future_years,
                     future_pi["95"][0], future_pi["95"][1],
                     color='orange', alpha=0.15)

    plt.fill_between(future_years,
                     future_pi["80"][0], future_pi["80"][1],
                     color='orange', alpha=0.30)

    # ---------------------------
    # Metrics
    # ---------------------------
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    metrics_dict[country] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'R2': r2_score(y_test, y_pred),
        'Explained Variance': explained_variance_score(y_test, y_pred),
        'Max Error': max_error(y_test, y_pred),
        'Median AE': median_absolute_error(y_test, y_pred),
        'Accuracy (%)': 100 - mape
    }

    # ---------------------------
    # Forecast Table
    # ---------------------------
    forecast_df = pd.DataFrame({
        'Year': future_years,
        'Point Forecast': future_preds,
        'PI80 Lower': future_pi["80"][0],
        'PI80 Upper': future_pi["80"][1],
        'PI95 Lower': future_pi["95"][0],
        'PI95 Upper': future_pi["95"][1]
    })

    print("\nFuture Forecast with Prediction Intervals:")
    print(forecast_df.round(3).to_string(index=False))

# ---------------------------
# Final Plot
# ---------------------------
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions")
plt.title("CatBoost CO₂ Forecasting with 80% & 95% Prediction Intervals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Metrics
# ---------------------------
for country, metrics in metrics_dict.items():
    print(f"\nPerformance Metrics – {country}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
