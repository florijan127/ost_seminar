import torch
import pandas as pd
import matplotlib.pyplot as plt

# Definicija modela (mora biti ista kao kod treniranja)
class StartlePredictionModel(torch.nn.Module):
    def __init__(self, input_size):
        super(StartlePredictionModel, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(64, 32)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(32, 1)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Učitavanje spremljenog modela
model = StartlePredictionModel(input_size=5)  # Promijeni broj ulaznih značajki ako je potrebno
model.load_state_dict(torch.load("startle_prediction_model.pth"))
model.eval()

# Učitavanje podataka
X_test = pd.read_csv("X_test.csv").values
y_test = pd.read_csv("y_test.csv").values.flatten()

print("Stvarne vrijednosti (y_test):")
print(y_test[:10])  # Ispis prvih 10 vrijednosti

# Pretvaranje u PyTorch tenzore
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

# Predikcije
with torch.no_grad():
    predictions = model(X_test_tensor).numpy().flatten()

# Vizualizacija predikcija u odnosu na stvarne vrijednosti
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Stvarne vrijednosti (trajanje u sekundama)", color="blue", linestyle="--", alpha=0.7)
plt.plot(predictions, label="Predikcije modela (trajanje u sekundama)", color="red", alpha=0.7)

# Dodavanje margina i prilagodba osi za jasniju usporedbu
plt.ylim(min(min(y_test), min(predictions)) - 1, max(max(y_test), max(predictions)) + 1)

# Oznake i naslov
plt.title("Usporedba predikcija i stvarnih vrijednosti", fontsize=14)
plt.xlabel("Primjer", fontsize=12)
plt.ylabel("Vrijednost trajanja u sekundama", fontsize=12)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()