import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plt.scatter(['a', 'b', 'c'], [1, 2, 1], color='red')
plt.plot(['a', 'b', 'c'], [1, 2, 1], marker='o')
plt.scatter(['a', 'b', 'c'], [1, 3, 1], color='red')
plt.plot(['a', 'b', 'c'], [1, 3, 1], marker='o')
plt.scatter(['a', 'b', 'c'], [1, 4, 1], color='red')
plt.plot(['a', 'b', 'c'], [1, 4, 1], marker='o')

plt.show()