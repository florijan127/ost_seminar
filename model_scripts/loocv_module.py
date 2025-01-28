import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm  # Za progress bar
import matplotlib.pyplot as plt
from model import prepare_dataloaders, StartlePredictionModel, train_model, evaluate_model

def loocv(X_path, y_path, epochs=100, input_size=8):
    # Učitavanje podataka
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path)
    
    mse_values = []
    actuals = []
    predictions = []
    for id in X['id'].unique():
        X_train = X[X['id'] != id].drop(columns='id').values
        y_train = y[y['id'] != id].drop(columns='id').values.flatten()
        X_test = X[X['id'] == id].drop(columns='id').values
        y_test = y[y['id'] == id].drop(columns='id').values.flatten()
        
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
        
        model = StartlePredictionModel(input_size)

        # Definicija gubitka i optimizatora
        criterion = nn.MSELoss()  # Gubitak za regresiju
        optimizer = optim.Adam(model.parameters(), lr=0.002)
        
        train_model(model, train_loader, criterion, optimizer, epochs=epochs)

        # Evaluacij modela
        mse, a, p = evaluate_model(model, test_loader, plot=False)
        mse_values.append(mse)
        actuals.extend(a)
        predictions.extend(p)

        # torch.save(model.state_dict(), "startle_prediction_model.pth")
    mse = np.mean(mse_values)
    root_mse = np.sqrt(mse)
    print(f"Mean Squared Error on Test Set: {mse:.4f}")
    print(f"Standard Deviation of MSE: {np.std(mse_values):.4f}")
    print(f"Root Mean Squared Error on Test Set: {root_mse:.4f}")
    print(f"Standard Deviation of RMSE: {np.std(np.sqrt(mse_values)):.4f}")
    plt.scatter(range(len(actuals)), actuals, label="Actual")
    plt.scatter(range(len(predictions)), predictions, color='red', label="Predicted")
    for i in range(len(actuals)):
        plt.plot([i, i], [actuals[i], predictions[i]], color='black')
    plt.legend()
    plt.show()

