import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

znacajke = pd.read_csv("znacajke.csv")
print(len(znacajke))

plt.figure(figsize=(20, 10))

for i, feature in enumerate(['EDAmean', 'EDAstd', 'EDAslope', 'EDArange', 'RESPrate', 'RESPamplitude', 'RESPstd', 'RESPrange']):
    plt.subplot(2, 4, i + 1)
    plt.scatter(znacajke[feature], znacajke['mae'], color='red')
    plt.xlabel(feature)
    plt.ylabel('mae')
    plt.legend()
    plt.grid(True)
        
plt.show()
