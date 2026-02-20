# =========================================================
# Gradient Boosting CO2 Forecasting with 80% & 95% PIs
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
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

n_lags = 5
n_test = 12
n_future = 12

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
    # Extract Time Series
    # ---------------------------
    ts = df_raw[['Year', co2_columns[country]]].dropna()
    ts.columns = ['Year', 'CO2']
    ts.set_index('Year', inplace=True)

    # ---------------------------
    # Scaling
    # ---------------------------
    scaler = MinMaxScaler()
    scaled_co2 = scaler.fit_transform(ts[['CO2']])

    # ---------------------------
    # Create Lagged Features
    # ---------------------------
    X, y = [], []
    for j in range(n_lags, len(scaled_co2)):
        X.append(scaled_co2[j - n_lags:j].flatten())
        y.append(scaled_co2[j][0])

    X = np.array(X)
    y = np.array(y)

    # ---------------------------
    # Train/Test Split
    # ---------------------------
    X_train, y_train = X[:-n_test], y[:-n_test]
    X_test, y_test = X[-n_test:], y[-n_test:]

    # ---------------------------
    # Gradient Boosting Model
    # ---------------------------
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ---------------------------
    # Test Prediction
    # ---------------------------
    y_pred_scaled = model.predict(X_test).reshape(-1, 1)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    test_years = ts.index[-n_test:]

    # ---------------------------
    # Training Residuals
    # ---------------------------
    train_pred = scaler.inverse_transform(
        model.predict(X_train).reshape(-1, 1)
    )
    train_true = scaler.inverse_transform(
        y_train.reshape(-1, 1)
    )

    residuals = train_true.flatten() - train_pred.flatten()

    # ---------------------------
    # Bootstrap PIs (Test)
    # ---------------------------
    boot_preds_test = []
    for _ in range(n_bootstrap):
        sampled_residuals = np.random.choice(
            residuals, size=len(y_pred), replace=True
        )
        boot_preds_test.append(y_pred.flatten() + sampled_residuals)

    boot_preds_test = np.array(boot_preds_test)

    test_pi = {
        level: (
            np.percentile(boot_preds_test, lq, axis=0),
            np.percentile(boot_preds_test, uq, axis=0)
        )
        for level, (lq, uq) in pi_levels.items()
    }

    # ---------------------------
    # Future Forecast
    # ---------------------------
    last_input = scaled_co2[-n_lags:].flatten()
    future_scaled = []

    for _ in range(n_future):
        next_val = model.predict(last_input.reshape(1, -1))[0]
        future_scaled.append(next_val)
        last_input = np.roll(last_input, -1)
        last_input[-1] = next_val

    y_future = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).flatten()

    future_years = np.arange(
        ts.index.max() + 1,
        ts.index.max() + n_future + 1
    )

    # ---------------------------
    # Bootstrap PIs (Future)
    # ---------------------------
    boot_preds_future = []
    for _ in range(n_bootstrap):
        noise = np.random.choice(residuals, size=n_future, replace=True)
        boot_preds_future.append(y_future + noise)

    boot_preds_future = np.array(boot_preds_future)

    future_pi = {
        level: (
            np.percentile(boot_preds_future, lq, axis=0),
            np.percentile(boot_preds_future, uq, axis=0)
        )
        for level, (lq, uq) in pi_levels.items()
    }

    # =====================================================
    # FUTURE FORECAST TABLE (ADDED)
    # =====================================================
    forecast_df = pd.DataFrame(
        {
            'Year': future_years,
            'Point Forecast': y_future,
            'PI80 Lower': future_pi["80"][0],
            'PI80 Upper': future_pi["80"][1],
            'PI95 Lower': future_pi["95"][0],
            'PI95 Upper': future_pi["95"][1]
        }
    )

    print("\nFuture Forecast with Prediction Intervals:")
    print(forecast_df.round(3).to_string(index=False))

    # ---------------------------
    # Plot
    # ---------------------------
    plt.plot(ts.index, ts['CO2'], marker=markers[i],
             label=f'{country} Actual')

    plt.plot(test_years, y_pred.flatten(), linestyle='--',
             marker=markers[i], label=f'{country} Test Forecast')

    plt.fill_between(test_years,
                     test_pi["95"][0], test_pi["95"][1],
                     color='gray', alpha=0.15)

    plt.fill_between(test_years,
                     test_pi["80"][0], test_pi["80"][1],
                     color='gray', alpha=0.30)

    plt.plot(future_years, y_future, linestyle='dotted',
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
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics_dict[country] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'R2': r2_score(y_true, y_pred),
        'Explained Variance': explained_variance_score(y_true, y_pred),
        'Max Error': max_error(y_true, y_pred),
        'Median AE': median_absolute_error(y_true, y_pred),
        'Accuracy (%)': 100 - mape
    }

# ---------------------------
# Final Plot
# ---------------------------
plt.xlabel("Year")
plt.ylabel("CO₂ Emissions")
plt.title("Gradient Boosting CO₂ Forecasting with 80% & 95% Prediction Intervals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Display Metrics
# ---------------------------
for country, metrics in metrics_dict.items():
    print(f"\nPerformance Metrics – {country}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
