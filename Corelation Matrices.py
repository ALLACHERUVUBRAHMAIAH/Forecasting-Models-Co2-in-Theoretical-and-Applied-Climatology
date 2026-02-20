# === Imports ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, r2_score, max_error, explained_variance_score

# === Load Dataset ===
file_path = '/content/drive/MyDrive/DATA/Co2.csv'
df_raw = pd.read_csv(file_path, header=[0, 1])
df_raw.columns = ['_'.join(col).strip() for col in df_raw.columns.values]
df_raw.rename(columns={'Country_Year': 'Year'}, inplace=True)
df_raw['Year'] = df_raw['Year'].astype(int)

# === Parameters ===
countries = ['Canada', 'India', 'Japan', 'Russia']
n_lags = 5
n_test = 12
metrics_dict = {}

# === Identify CO2 Columns ===
co2_columns = {}
for col in df_raw.columns:
    for country in countries:
        if country in col and 'CO2' in col:
            co2_columns[country] = col

# === Loop Through Countries ===
for country in countries:
    if country not in co2_columns:
        continue

    column = co2_columns[country]
    ts = df_raw[['Year', column]].dropna().copy()
    ts.columns = ['Year', 'CO2']
    ts.set_index('Year', inplace=True)

    # === Scaling ===
    scaler = MinMaxScaler()
    scaled_co2 = scaler.fit_transform(ts[['CO2']])

    # === Prepare Sequence Data ===
    X, y = [], []
    for j in range(n_lags, len(scaled_co2) - n_test):
        X.append(scaled_co2[j - n_lags:j])
        y.append(scaled_co2[j])
    X = np.array(X)
    y = np.array(y)

    # === Simple Forecasting Model (Baseline: Last Value Prediction) ===
    # You can replace this with your GRU/LSTM model predictions
    y_pred = X[:, -1, 0]  # naive forecast: last observed value
    y_true = y[:, 0]

    # === Compute Metrics ===
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    maxerr = max_error(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    accuracy = 100 - mape

    metrics_dict[country] = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'Median AE': medae,
        'R2 Score': r2,
        'Max Error': maxerr,
        'Explained Variance': evs,
        'MAPE (%)': mape,
        'Accuracy (%)': accuracy
    }

# === Convert Metrics to DataFrame ===
metrics_df = pd.DataFrame(metrics_dict).T
print("Metrics Table:\n", metrics_df)

# === Compute Correlation Matrix ===
corr = metrics_df.corr()
print("\nCorrelation Matrix:\n", corr)

# === Visualize Correlation Matrix ===
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Forecasting Metrics")
plt.show()

# === Suggest Redundant Metrics Based on High Correlation (>0.9) ===
print("\nRedundant Metrics Suggestion (correlation > 0.9):")
threshold = 0.9
for i in corr.columns:
    for j in corr.columns:
        if i != j and corr.loc[i, j] > threshold:
            print(f"{i} and {j} are highly correlated (r = {corr.loc[i,j]:.2f})")
