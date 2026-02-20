import matplotlib.pyplot as plt

# === Data ===
time_lags = ['t-5', 't-4', 't-3', 't-2', 't-1']
importances = [0.011, 0.020, 0.034, 0.052, 0.062]

# === Plot ===
plt.figure(figsize=(8, 5))
plt.bar(time_lags, importances)

# === Labels and Title ===
plt.xlabel('Time Lag')
plt.ylabel('Integrated Gradient Importance')
plt.title('Temporal Importance')

# === Grid ===
plt.grid(True, axis='y', linestyle='--', alpha=0.6)

# === Value labels on bars (small improvement) ===
for i in range(len(importances)):
    plt.text(i, importances[i] + 0.001,
             round(importances[i], 3),
             ha='center', fontsize=10)

# === Layout ===
plt.tight_layout()

# === Show ===
plt.show()
