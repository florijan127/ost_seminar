import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
# Create mock data
data = pd.DataFrame({
    'A': np.random.rand(100),
    'B': np.random.rand(100),
    'C': np.random.rand(100),
    'D': np.random.rand(100)
})


# Calculate correlation matrix
correlation_matrix = data.corr()
print(correlation_matrix)

# Print correlation matrix
print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')

plt.colorbar()
plt.title('Correlation Matrix', pad=20)
plt.show()
import scipy.stats as stats

# Calculate correlation coefficients and p-values
correlation_coefficients = data.corr()
p_values = pd.DataFrame(np.zeros((data.shape[1], data.shape[1])), columns=data.columns, index=data.columns)

for row in data.columns:
    for col in data.columns:
        if row != col:
            corr_test = stats.pearsonr(data[row], data[col])
            p_values.loc[row, col] = corr_test[1]
        else:
            p_values.loc[row, col] = np.nan

print("Correlation Coefficients:\n", correlation_coefficients)
print("\nP-values:\n", p_values)