import matplotlib.pyplot as plt
import numpy as np

# Models
models = ['Holt-Winters', 'GPR', 'BSTS', 'CatBoost', 'RF', 'Decision Tree',
          'Gradient Boosting', 'GRU', 'CNN-LSTM', 'N-BEATS']

# Accuracy (%) for each country
accuracy_india = [95.47, 96.21, 97.94, 67.49, 76.54, 75.85, 78.39, 98.11, 96.37, 31.27]
accuracy_canada = [59.62, 93.76, 69.85, 77.25, 74.79, 75.12, 74.46, 93.62, 93.56, 75.19]
accuracy_japan = [95.47, 95.07, 91.67, 93.14, 92.84, 75.12, 94.31, 97.83, 97.59, 90.11]
accuracy_russia = [89.13, 94.12, 74.09, 88.06, 86.26, 86.10, 85.65, 92.24, 93.05, 79.88]

# Compute average accuracy across countries
accuracy_avg = np.mean([accuracy_india, accuracy_canada, accuracy_japan, accuracy_russia], axis=0)

# Sort models by average accuracy
sorted_indices = np.argsort(-accuracy_avg)  # descending order
sorted_models = [models[i] for i in sorted_indices]
sorted_accuracy = accuracy_avg[sorted_indices]

# Plot comparative bar chart
plt.figure(figsize=(12,6))
bars = plt.bar(sorted_models, sorted_accuracy, color='skyblue')
plt.ylabel('Average Accuracy (%) of Four Countries')
plt.title('Comparative Analysis of Proposed Models for CO₂ Emission Forecasting')
plt.ylim(0, 105)

# Annotate accuracy values on top of bars
for bar, acc in zip(bars, sorted_accuracy):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{acc:.2f}%',
             ha='center', va='bottom', fontsize=10)

plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
