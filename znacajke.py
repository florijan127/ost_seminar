import pandas as pd 
import numpy as np

offset = pd.read_csv("offset.csv")

for i in range(1, 4):
    offset[f'startle{i}'] = offset[f'startle{i}'] + offset['offset']
    offset[f'pre{i}'] = offset[f'startle{i}'] - 10
    offset[f'post{i}'] = offset[f'startle{i}'] + 10

data = {'id':[], 'startle_index':[], 'phase':[], 'EDAmean':[], 'EDAstd':[], 'EDAslope':[], 'EDArange':[], 'RESPrate':[], 'RESPamplitude':[], 'RESPstd':[], 'RESPrange':[], 'mae':[]}
mae = pd.read_excel("mae_startle.xlsx")

for i in range(1, 4):
    for id in offset['id']:
        eda = pd.read_csv(f"EDA/eda_{id:02}.csv")
        resp = pd.read_csv(f"RESPIRATION/respiration_{id:02}.csv")
        for phase in ['pre', 'startle', 'post']:
            data['id'].append(id)
            data['startle_index'].append(i)
            data['phase'].append(phase)
            start_index = int(offset[offset['id'] == id][f'{phase}{i}'].values[0] * 1000)
            eda_window = eda.iloc[start_index:start_index + 10000]
            resp_window = resp.iloc[start_index:start_index + 10000]
            data['EDAmean'].append(eda_window.mean().iloc[0])
            data['EDAstd'].append(eda_window.std().iloc[0])
            data['EDAslope'].append(np.polyfit(np.arange(len(eda_window)), eda_window, 1)[0][0])
            data['EDArange'].append(eda_window.max().iloc[0] - eda_window.min().iloc[0])
            data['RESPrate'].append(resp_window.mean().iloc[0])
            data['RESPamplitude'].append(resp_window.max().iloc[0] - resp_window.min().iloc[0])
            data['RESPstd'].append(resp_window.std().iloc[0])
            data['RESPrange'].append(resp_window.max().iloc[0] - resp_window.min().iloc[0])
            if phase == 'startle':
                mae_phase = f'startle {i}'
            else:
                mae_phase = f'{phase}-startle {i}' 
            
            data['mae'].append(mae[mae['ID'] == id][mae_phase.upper()].values[0])

# print(data)
df = pd.DataFrame(data)
df.to_csv("znacajke.csv", index=False, float_format='%.6f')

