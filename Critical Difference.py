import matplotlib.pyplot as plt
from math import sqrt

k = len(models)   # number of models
N = data.shape[0] # number of datasets

# Critical Difference for alpha=0.05
q_alpha = 3.314  # Nemenyi critical value for k=10
CD = q_alpha * sqrt((k * (k + 1)) / (6 * N))

# Sort models by rank
sorted_idx = np.argsort(avg_ranks)
sorted_ranks = avg_ranks[sorted_idx]
sorted_models = np.array(models)[sorted_idx]

plt.figure(figsize=(12, 3))
plt.hlines(1, min(sorted_ranks)-0.2, max(sorted_ranks)+0.2)
plt.plot(sorted_ranks, [1]*k, 'o')

for i, model in enumerate(sorted_models):
    plt.text(sorted_ranks[i], 1.02, model, rotation=45,
             ha='right', va='bottom', fontsize=10)

# CD bar
plt.plot([max(sorted_ranks)-CD, max(sorted_ranks)], [0.95, 0.95], lw=3)
plt.text(max(sorted_ranks)-CD/2, 0.92, f'CD = {CD:.2f}',
         ha='center', fontsize=10)

plt.yticks([])
plt.xlabel('Average Rank (Lower is Better)')
plt.title('Critical Difference Diagram (Nemenyi Post-ho,Mape)')
plt.tight_layout()
plt.show()
