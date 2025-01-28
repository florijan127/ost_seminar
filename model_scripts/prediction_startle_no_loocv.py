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





# Inicijalizacija modela
x_train_path = "X_prediction_startle.csv"
y_train_path = "y_prediction_startle.csv"

input_size = 8  # Broj značajki
model = StartlePredictionModel(input_size)

# Definicija gubitka i optimizatora
criterion = nn.MSELoss()  # Gubitak za regresiju
optimizer = optim.Adam(model.parameters(), lr=0.002)

train_loader, test_loader = prepare_dataloaders(x_train_path, y_train_path, x_train_path, y_train_path)

# Treniranje modela
train_model(model, train_loader, criterion, optimizer, epochs=10000)

# Evaluacija modela
evaluate_model(model, train_loader)

torch.save(model.state_dict(), "prediction_pre_model.pth")