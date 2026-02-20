# =========================================================
# BSTS CO2 Forecasting with 80% & 95% Prediction Intervals
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
sts = tfp.sts

# ---------------------------
# Parameters
# ---------------------------
file_path = '/content/drive/MyDrive/DATA/Co2.csv'  # Replace with your path
countries = ['Canada', 'India', 'Japan', 'Russia']
markers = ['o', 's', '^', 'D']
n_future = 12
pi_levels = {"80": (10, 90), "95": (2.5, 97.5)}

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

# ---------------------------
# Metrics Dictionary
# ---------------------------
metrics_dict = {}

# ---------------------------
# Main Loop for Each Country
# ---------------------------
for i, country in enumerate(countries):

    if country not in co2_columns:
        continue

    print(f"\n================ {country} =================")

    # ---------------------------
    # Prepare Time Series
    # ---------------------------
    ts = df_raw[['Year', co2_columns[country]]].dropna().copy()
    ts.columns = ['Year', 'CO2']
    ts.set_index('Year', inplace=True)
    observed_time_series = ts['CO2'].values.astype(np.float32)

    # ---------------------------
    # Define BSTS Model
    # ---------------------------
    trend = sts.LocalLinearTrend(observed_time_series=observed_time_series)
    seasonal = sts.Seasonal(num_seasons=5, observed_time_series=observed_time_series)
    model = sts.Sum([trend, seasonal], observed_time_series=observed_time_series)

    # ---------------------------
    # Variational Inference
    # ---------------------------
    surrogate_posterior = tfp.sts.build_factored_surrogate_posterior(model)
    optimizer = tf.optimizers.Adam(learning_rate=0.1)

    tfp.vi.fit_surrogate_posterior(
        target_log_prob_fn=model.joint_distribution(observed_time_series).log_prob,
        surrogate_posterior=surrogate_posterior,
        optimizer=optimizer,
        num_steps=500
    )

    # ---------------------------
    # In-Sample Fitted Values
    # ---------------------------
    posterior_samples = surrogate_posterior.sample(50)
    fitted_dist = tfp.sts.one_step_predictive(
        model,
        observed_time_series=observed_time_series,
        parameter_samples=posterior_samples
    )
    y_fitted = np.mean(fitted_dist.mean().numpy(), axis=0)  # 1D array

    # ---------------------------
    # Compute In-Sample Metrics
    # ---------------------------
    mse = np.mean((observed_time_series - y_fitted) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((observed_time_series - y_fitted) / observed_time_series)) * 100

    metrics_dict[country] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAPE (%)': mape
    }

    # ---------------------------
    # Forecast Future
    # ---------------------------
    forecast_dist = tfp.sts.forecast(
        model,
        observed_time_series=observed_time_series,
        parameter_samples=posterior_samples,
        num_steps_forecast=n_future
    )

    y_forecast_mean = forecast_dist.mean().numpy().flatten()
    y_forecast_samples = forecast_dist.sample(1000).numpy()
    future_years = np.arange(ts.index.max() + 1, ts.index.max() + n_future + 1)

    # ---------------------------
    # Prediction Intervals
    # ---------------------------
    future_pi = {}
    for level, (lq, uq) in pi_levels.items():
        lower = np.percentile(y_forecast_samples, lq, axis=0).flatten()
        upper = np.percentile(y_forecast_samples, uq, axis=0).flatten()
        future_pi[level] = (lower, upper)

    # ---------------------------
    # Plotting
    # ---------------------------
    plt.plot(ts.index, ts['CO2'], marker=markers[i], label=f'{country} Actual')
    plt.plot(future_years, y_forecast_mean, linestyle='dotted',
             marker=markers[i], label=f'{country} Forecast (BSTS)')

    plt.fill_between(future_years, future_pi["95"][0], future_pi["95"][1],
                     color='orange', alpha=0.15)
    plt.fill_between(future_years, future_pi["80"][0], future_pi["80"][1],
                     color='orange', alpha=0.30)

    # ---------------------------
    # Forecast Table
    # ---------------------------
    forecast_df = pd.DataFrame({
        'Year': future_years,
        'Point Forecast': y_forecast_mean,
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
plt.title("BSTS CO₂ Forecasting with 80% & 95% Prediction Intervals")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------
# Display In-Sample Metrics
# ---------------------------
for country, metrics in metrics_dict.items():
    print(f"\nPerformance Metrics – {country}")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
