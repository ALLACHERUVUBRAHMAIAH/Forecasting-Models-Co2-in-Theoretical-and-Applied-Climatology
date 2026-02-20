import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import rankdata

# Accuracy data
accuracy_india = [95.47, 96.21, 97.94, 67.49, 76.54, 75.85, 78.39, 88.11, 86.37, 31.27]
accuracy_canada = [59.62, 90.76, 69.85, 77.25, 74.79, 75.12, 74.46, 91.62, 87.56, 75.19]
accuracy_japan = [95.47, 93.07, 91.67, 93.14, 92.84, 75.12, 94.31, 97.83, 97.59, 90.11]
accuracy_russia = [89.13, 86.12, 74.09, 88.06, 86.26, 86.10, 85.65, 92.24, 93.05, 79.88]

models = ['Holt-Winters', 'GPR', 'BSTS', 'CatBoost', 'RF',
          'Decision Tree', 'Gradient Boosting', 'GRU',
          'CNN-LSTM', 'N-BEATS']

# Convert to DataFrame (required for post-hoc)
df = pd.DataFrame({
    'India': accuracy_india,
    'Canada': accuracy_canada,
    'Japan': accuracy_japan,
    'Russia': accuracy_russia
}, index=models)

# Transpose: rows=datasets, columns=models
df = df.T

# Nemenyi post-hoc test
nemenyi_results = sp.posthoc_nemenyi_friedman(df)

print("Nemenyi Post-hoc p-values:")
print(nemenyi_results)
