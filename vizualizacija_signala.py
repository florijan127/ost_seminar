import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# offset = pd.read_excel("offset.xlsx")
# print(offset)
# offset.columns = ["id", "offset", "startle1", "startle2", "startle3", "duration1", "nan"]
# offset.to_csv("offset.csv", index=False, float_format='%.3f')

# Učitavanje podataka
eda = pd.read_csv("EDA/eda_01.csv")
resp = pd.read_csv("RESPIRATION/respiration_01.csv")
time = np.arange(0, len(eda)/1000, 0.001)

offset = pd.read_csv("offset.csv")

# Vizualizacija signala
plt.figure(figsize=(20, 10))
plt.subplot(2, 1, 1)
plt.plot(time, eda, label="EDA")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time, resp, label="RESP")
plt.legend()

for i in range(1, 4):
    plt.subplot(2, 1, 1).axvline(x=offset["offset"][0] + offset[f"startle{i}"][0], color="red", linestyle="--")
    plt.subplot(2, 1, 2).axvline(x=offset["offset"][0] + offset[f"startle{i}"][0], color="red", linestyle="--")

plt.show()
plt.subplot(2, 1, 1).legend()
plt.subplot(2, 1, 2).legend()