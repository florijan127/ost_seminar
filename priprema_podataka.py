import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_all_csv(directory, prefix):
    file_paths = glob.glob(os.path.join(directory, f"{prefix}_*.csv"))
    print(f"\nUčitavanje datoteka iz {directory} s prefiksom '{prefix}':")
    print(f"Pronađene datoteke: {file_paths}")
    
    combined_data = pd.concat(
        (pd.read_csv(file, header=None, names=['value']) for file in file_paths),
        axis=0, 
        ignore_index=True
    )
    print(f"Učitano {len(combined_data)} redaka podataka")
    return combined_data['value']

# Definiranje direktorija
current_dir = os.getcwd()
respiration_dir = os.path.join(current_dir, "RESPIRATION")
time_dir = os.path.join(current_dir, "TIME")
offset_file = os.path.join(current_dir, "offset.xlsx")
mae_startle_file = os.path.join(current_dir, "mae_startle.xlsx")

# Učitavanje podataka
print("\nUčitavanje podataka...")
respiration_data = load_all_csv(respiration_dir, "respiration")
time_data = load_all_csv(time_dir, "time")
offset_data = pd.read_excel(offset_file)
mae_startle_data = pd.read_excel(mae_startle_file)

# Primjena offseta na mae_startle podatke
offset = offset_data["Offset (simulator_start - fiziologija_start)"].iloc[0]
print(f"\nPrimjena offseta {offset} na sve vremenske oznake...")

# Dodavanje offseta na sve vremenske stupce u mae_startle
time_columns = ['PRE-STARTLE 1', 'STARTLE 1', 'POST-STARTLE 1',
                'PRE-STARTLE 2', 'STARTLE 2', 'POST-STARTLE 2',
                'PRE-STARTLE 3', 'STARTLE 3', 'POST-STARTLE 3']

for col in time_columns:
    mae_startle_data[col] = mae_startle_data[col] + offset

# Stvaranje DataFrame s vremenskim oznakama
df = pd.DataFrame({
    'respiration': respiration_data,
    'time': time_data + offset
})

print("\nVremenski rasponi nakon primjene offseta:")
print(f"Podaci disanja: {df['time'].min():.2f} do {df['time'].max():.2f}")
print(f"MAE Startle (PRE-1): {mae_startle_data['PRE-STARTLE 1'].min():.2f} do {mae_startle_data['PRE-STARTLE 1'].max():.2f}")
print(f"MAE Startle (POST-1): {mae_startle_data['POST-STARTLE 1'].min():.2f} do {mae_startle_data['POST-STARTLE 1'].max():.2f}")

def extract_features_with_sliding_window(data, window_size=500, step_size=250):
    features = []
    times = []
    
    for start in range(0, len(data) - window_size + 1, step_size):
        window = data.iloc[start:start + window_size]
        
        features.append({
            'mean': window['respiration'].mean(),
            'std': window['respiration'].std(),
            'min': window['respiration'].min(),
            'max': window['respiration'].max(),
            'median': window['respiration'].median()
        })
        
        times.append(window['time'].iloc[window_size // 2])
    
    return pd.DataFrame(features), pd.Series(times)

# Generiranje značajki
print("\nGeneriranje značajki...")
X, times = extract_features_with_sliding_window(df)
print(f"Generirano {len(X)} prozora sa značajkama")

# Generiranje ciljnih vrijednosti
print("\nGeneriranje ciljnih vrijednosti...")
y = []
matches_found = 0

for i, t in enumerate(times):
    if i % 1000 == 0:
        print(f"Obrađeno {i}/{len(times)} vremenskih točaka")
    
    # Provjera svih startle perioda (1, 2 i 3)
    for period in range(1, 4):
        pre_col = f'PRE-STARTLE {period}'
        post_col = f'POST-STARTLE {period}'
        
        match = mae_startle_data[
            (mae_startle_data[pre_col] <= t) &
            (mae_startle_data[post_col] >= t)
        ]
        
        if not match.empty:
            matches_found += 1
            y.append(match[post_col].iloc[0] - match[pre_col].iloc[0])  # Trajanje perioda
            break
    else:  # Ako nije pronađeno podudaranje ni u jednom periodu
        y.append(np.nan)

print(f"\nUkupno pronađeno {matches_found} podudaranja od {len(times)} vremenskih točaka")

# Uklanjanje redaka s NaN vrijednostima
print("\nČišćenje podataka...")
y = np.array(y)
valid_indices = ~np.isnan(y)
X = X[valid_indices]
y = y[valid_indices]

if len(X) == 0:
    raise ValueError("Nema preostalih podataka nakon čišćenja. Provjerite vremenske raspone u podacima.")

print(f"\nPreostalo {len(X)} uzoraka nakon čišćenja")

# Podjela na trening i test skupove
print("\nPodjela na trening i test skupove...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Spremanje podataka
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv', index=False)
pd.Series(y_train).to_csv('y_train.csv', index=False)
pd.Series(y_test).to_csv('y_test.csv', index=False)

print("\nPodaci su uspješno pripremljeni i podijeljeni:")
print(f"Trening skup: {len(X_train)} redaka")
print(f"Test skup: {len(X_test)} redaka")

# Statistika ciljnih vrijednosti
print("\nStatistika ciljnih vrijednosti:")
print(f"Min: {y.min():.2f}")
print(f"Max: {y.max():.2f}")
print(f"Mean: {y.mean():.2f}")
print(f"Std: {y.std():.2f}")