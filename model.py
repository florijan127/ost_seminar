import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm  # Za progress bar

# Učitavanje podataka
X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values.flatten()
y_test = pd.read_csv("y_test.csv").values.flatten()

# Pretvaranje podataka u PyTorch tenzore
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Dataset i DataLoader za treniranje i evaluaciju
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Definicija modela
class StartlePredictionModel(nn.Module):
    def __init__(self, input_size):
        super(StartlePredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)  # Izlaz je skalarna vrijednost
        self.dropout = nn.Dropout(0.3)  # Regularizacija za izbjegavanje prenaučenosti

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Inicijalizacija modela
input_size = X_train.shape[1]  # Broj značajki
model = StartlePredictionModel(input_size)

# Definicija gubitka i optimizatora
criterion = nn.MSELoss()  # Gubitak za regresiju
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Funkcija za treniranje
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_X, batch_y in progress_bar:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": loss.item()})
        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(train_loader):.4f}")

# Funkcija za evaluaciju
def evaluate_model(model, test_loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for batch_X, batch_y in progress_bar:
            outputs = model(batch_X)
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())
    predictions = np.array(predictions).flatten()
    actuals = np.array(actuals).flatten()
    mse = mean_squared_error(actuals, predictions)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")
    return mse

# Treniranje modela
train_model(model, train_loader, criterion, optimizer, epochs=50)

# Evaluacija modela
evaluate_model(model, test_loader)

torch.save(model.state_dict(), "startle_prediction_model.pth")