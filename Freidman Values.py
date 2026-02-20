import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata

# Accuracy (%) for each country
accuracy_india = [95.47, 96.21, 97.94, 67.49, 76.54, 75.85, 78.39, 88.11, 86.37, 31.27]
accuracy_canada = [59.62, 90.76, 69.85, 77.25, 74.79, 75.12, 74.46, 91.62, 87.56, 75.19]
accuracy_japan = [95.47, 93.07, 91.67, 93.14, 92.84, 75.12, 94.31, 97.83, 97.59, 90.11]
accuracy_russia = [89.13, 86.12, 74.09, 88.06, 86.26, 86.10, 85.65, 92.24, 93.05, 79.88]

models = ['Holt-Winters', 'GPR', 'BSTS', 'CatBoost', 'RF', 'Decision Tree',
          'Gradient Boosting', 'GRU', 'CNN-LSTM', 'N-BEATS']

# Combine data into 2D array (rows=countries, columns=models)
data = np.array([accuracy_india, accuracy_canada, accuracy_japan, accuracy_russia])

# Compute ranks per country (higher accuracy = lower rank number)
ranks = np.array([rankdata(-row, method='average') for row in data])

# Average rank for each model across countries
avg_ranks = ranks.mean(axis=0)

print("Average Ranks:", dict(zip(models, avg_ranks)))
# Sort models by average rank (optional)
sorted_indices = np.argsort(avg_ranks)
sorted_models = [models[i] for i in sorted_indices]
sorted_avg_ranks = avg_ranks[sorted_indices]

# Plot bar chart
plt.figure(figsize=(12,6))
bars = plt.bar(sorted_models, sorted_avg_ranks, color='skyblue')
plt.ylabel('Average Rank (Friedman Test)')
plt.title('Friedman Test Ranking of Forecasting Models for CO₂ Emissions')
plt.ylim(0, max(sorted_avg_ranks)+1)

# Annotate rank values on bars
for bar, rank in zip(bars, sorted_avg_ranks):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{rank:.2f}', ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
