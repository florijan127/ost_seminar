import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Učitavanje i normalizacija podataka
X_train = pd.read_csv("X_train.csv").values
X_test = pd.read_csv("X_test.csv").values
y_train = pd.read_csv("y_train.csv").values.flatten()
y_test = pd.read_csv("y_test.csv").values.flatten()

# Normalizacija ulaznih podataka
scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Normalizacija ciljnih vrijednosti
scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Pretvaranje u PyTorch tenzore
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

# Poboljšani Dataset i DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class ImprovedStartleModel(nn.Module):
    def __init__(self, input_size):
        super(ImprovedStartleModel, self).__init__()
        
        # Prvi blok
        self.block1 = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Drugi blok
        self.block2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Treći blok
        self.block3 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Izlazni sloj
        self.output = nn.Sequential(
            nn.Linear(32, 1),
            nn.Tanh()  # Za stabilizaciju izlaza
        )
        
        # Inicijalizacija težina
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.fill_(0.01)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.output(x)
        return x

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler, epochs=100):
    early_stopping = EarlyStopping(patience=10)
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(epochs):
        # Treniranje
        model.train()
        train_loss = 0
        for batch_X, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validacija
        model.eval()
        val_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Spremanje najboljeg modela
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Ažuriranje learning rate
        scheduler.step(val_loss)
        
        # Ispis metrika
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        
        # Denormalizacija za računanje stvarnog MSE
        predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        actuals_orig = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
        
        mse = mean_squared_error(actuals_orig, predictions_orig)
        r2 = r2_score(actuals_orig, predictions_orig)
        
        print(f"Epoch {epoch+1}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")
        print("-" * 50)
        
        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Vraćanje najboljih težina
    model.load_state_dict(best_model_state)
    return train_losses, val_losses

# Inicijalizacija modela i treniranje
input_size = X_train.shape[1]
model = ImprovedStartleModel(input_size)

# Loss funkcija s težinama
criterion = nn.MSELoss()

# Optimizator s weight decay za regularizaciju
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Treniranje modela
train_losses, val_losses = train_and_validate(model, train_loader, test_loader, criterion, optimizer, scheduler)

# Spremanje modela
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
}, 'improved_startle_model.pth')

# Vizualizacija rezultata treniranja
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.close()

# Finalna evaluacija
model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(batch_y.cpu().numpy())

predictions = np.array(predictions).flatten()
actuals = np.array(actuals).flatten()

# Denormalizacija predikcija
predictions_orig = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
actuals_orig = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()

# Završne metrike
final_mse = mean_squared_error(actuals_orig, predictions_orig)
final_rmse = np.sqrt(final_mse)
final_r2 = r2_score(actuals_orig, predictions_orig)

print("\nZavršne metrike:")
print(f"MSE: {final_mse:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"R2 Score: {final_r2:.4f}")