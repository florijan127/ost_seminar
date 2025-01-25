import pandas as pd
import numpy as np
import os

for i in range(1, 21):
    if os.path.exists(f'EDA/eda_{i:02}.csv'):
        eda = pd.read_csv(f'EDA/eda_{i:02}.csv')
        len_eda = len(eda)
    if os.path.exists(f'RESPIRATION/respiration_{i:02}.csv'):
        resp = pd.read_csv(f'RESPIRATION/respiration_{i:02}.csv')
        len_resp = len(resp)
    if os.path.exists(f'TIME/time_{i:02}.csv'):
        time = pd.read_csv(f'TIME/time_{i:02}.csv')
        len_time = len(time)
    
    if len_eda == len_resp == len_time:
        print(f'equal for {i}')
    else:
        print(f'not equal for {i}')
        print(f'len({i} = {len(eda)}')
        print(f'len({i} = {len(resp)}')
        print(f'len({i} = {len(time)}')

