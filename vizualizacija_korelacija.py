import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

znacajke = pd.read_csv("znacajke.csv")

data = znacajke[['EDAmean', 'EDAstd', 'EDAslope', 'EDArange', 'RESPrate', 'RESPamplitude', 'RESPstd', 'RESPrange', 'mae']]

correlation_matrix = data.corr()

# Print correlation matrix
# print(correlation_matrix)

# Plot correlation matrix
plt.figure(figsize=(10, 8))
plt.matshow(correlation_matrix, fignum=1)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
# for i in range(len(correlation_matrix.columns)):
#     for j in range(len(correlation_matrix.columns)):
#         plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', ha='center', va='center', color='black')
for i, row in enumerate(data.columns):
    for j, col in enumerate(data.columns):
        if row != col:
            corr_test = stats.pearsonr(data[row], data[col])
            plt.text(i, j, f'{corr_test[1]:.4f}', ha='center', va='center', color='black')
            # plt.text(i, j, 0.99, ha='center', va='center', color='black')
        else:
            plt.text(i, j, 1, ha='center', va='center', color='black')

plt.colorbar()
plt.title('Correlation Matrix', pad=20)
plt.show()

