import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

znacajke = pd.read_csv("znacajke.csv")
print(len(znacajke))

plt.figure(figsize=(20, 10))

for i, feature in enumerate(['EDAmean', 'EDAstd', 'EDAslope', 'EDArange', 'RESPrate', 'RESPamplitude', 'RESPstd', 'mae']):
    plt.subplot(1, 8, i + 1)
    plt.title(feature)
    for id in znacajke['id'].unique():
        y = []
        x = ['pre', 'startle', 'post']
        for phase in x:
            print(znacajke[(znacajke['id'] == id) & (znacajke['phase'] == phase) & (znacajke['startle_index'] == 1)][feature])
            y.append(znacajke[(znacajke['id'] == id) & (znacajke['phase'] == phase) & (znacajke['startle_index'] == 1)][feature])
        plt.plot(x, y, color='red', marker='o')
        plt.scatter(x, y, color='red') 
    plt.legend()
        
plt.show()
