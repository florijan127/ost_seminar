import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

znacajke = pd.read_csv("znacajke.csv")
print(len(znacajke))

plt.figure(figsize=(20, 10))

# for i, feature in enumerate(['EDAmean', 'EDAstd', 'EDAslope', 'EDArange', 'RESPrate', 'RESPamplitude', 'RESPstd', 'RESPrange']):
for i, feature in enumerate(['EDAmean', 'EDAslope', 'RESPrate', 'RESPamplitude']):
    plt.subplot(2, 2, i + 1)
    plt.title(feature)
    data = []
    labels = []
    x = ['pre', 'startle', 'post']
    for phase in x:
        for phase_index in range(1, 4):
            data.append(znacajke[(znacajke['phase'] == phase) & (znacajke['startle_index'] == phase_index)][feature])
            labels.append(f"{phase} {phase_index}")
    
    plt.boxplot(data, labels=labels)
    plt.legend()
        
plt.show()
