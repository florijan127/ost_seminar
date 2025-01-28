import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from tqdm import tqdm  # Za progress bar
import matplotlib.pyplot as plt
# from model import prepare_dataloaders, StartlePredictionModel, train_model, evaluate_model
from loocv_module import loocv  # Ensure loocv_module.py contains the definition of loocv function





# Inicijalizacija modela
x_path = "X_prediction_startle_loocv.csv"
y_path = "y_prediction_startle_loocv.csv"

loocv(x_path, y_path, epochs=2000)
