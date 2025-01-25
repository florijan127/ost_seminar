import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Učitavanje spremljenog modela
checkpoint = torch.load('improved_startle_model.pth')
model_state = checkpoint['model_state_dict']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# Definicija modela (mora biti ista kao kod treniranja)
class ImprovedStartleModel(torch.nn.Module):
    def __init__(self, input_size):
        super(ImprovedStartleModel, self).__init__()
        self.block1 = torch.nn.Sequential(
            torch.nn.Linear(input_size, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3)
        )
        self.block2 = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        self.block3 = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        )
        self.output = torch.nn.Sequential(
            torch.nn.Linear(32, 1),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x

# Inicijalizacija modela
input_size = 5  # Broj značajki mora odgovarati broju značajki u podacima
model = ImprovedStartleModel(input_size)
model.load_state_dict(model_state)
model.eval()

# Učitavanje podataka za testiranje
X_test = pd.read_csv("X_test.csv").values
y_test = pd.read_csv("y_test.csv").values.flatten()

# Normalizacija ulaznih podataka
X_test_scaled = scaler_X.transform(X_test)

# Pretvaranje u PyTorch tenzore
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Predikcije
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()

# Denormalizacija podataka
predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
y_test_orig = scaler_y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Izračun metrika
mse = mean_squared_error(y_test_orig, predictions_orig)
r2 = r2_score(y_test_orig, predictions_orig)

print("Vizualizacija predikcija:")
print(f"MSE: {mse:.4f}")
print(f"R2 Score: {r2:.4f}")

# Graf predikcija vs stvarne vrijednosti
plt.figure(figsize=(14, 7))
plt.plot(y_test_orig, label="Stvarne vrijednosti", color="blue", linestyle="--", alpha=0.7)
plt.plot(predictions_orig, label="Predikcije modela", color="red", alpha=0.7)
plt.title("Usporedba predikcija modela i stvarnih vrijednosti", fontsize=16)
plt.xlabel("Primjeri", fontsize=14)
plt.ylabel("Vrijeme izdržljivosti (sekunde)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('predictions_vs_actuals_file.png')
plt.show()

# Scatter plot predikcija vs stvarne vrijednosti
plt.figure(figsize=(8, 8))
plt.scatter(y_test_orig, predictions_orig, alpha=0.6, color="green")
plt.plot([min(y_test_orig), max(y_test_orig)], [min(y_test_orig), max(y_test_orig)], color="red", linestyle="--", label="Idealna linija")
plt.title("Scatter plot: Stvarne vrijednosti vs Predikcije", fontsize=16)
plt.xlabel("Stvarne vrijednosti", fontsize=14)
plt.ylabel("Predikcije", fontsize=14)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('scatter_predictions_file.png')
plt.show()

# Histogram pogrešaka
errors = y_test_orig - predictions_orig
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color="purple", alpha=0.7)
plt.title("Histogram pogrešaka (Stvarne - Predikcije)", fontsize=16)
plt.xlabel("Pogreška (sekunde)", fontsize=14)
plt.ylabel("Broj primjera", fontsize=14)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('error_histogram_file.png')
plt.show()