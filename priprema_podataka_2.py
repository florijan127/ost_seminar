import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

znacajke = pd.read_csv("znacajke.csv")

X_estimation = znacajke[znacajke['phase'] == 'startle'][['EDAmean', 'EDAstd', 'EDAslope', 'EDArange', 'RESPrate', 'RESPamplitude', 'RESPstd', 'RESPrange', 'id']]
y_estimation = znacajke[znacajke['phase'] == 'startle'][['mae', 'id']]

X_prediction_startle = znacajke[znacajke['phase'] == 'pre'][['EDAmean', 'EDAstd', 'EDAslope', 'EDArange', 'RESPrate', 'RESPamplitude', 'RESPstd', 'RESPrange', 'id']]
y_prediction_startle = y_estimation.copy()


X_prediction_post = X_estimation.copy()
for column in X_prediction_startle.columns:
    if column != 'id':
        X_prediction_post[column + '_pre'] = X_prediction_startle[column].values

X_prediction_post['mae'] = y_estimation['mae']
y_prediction_post = znacajke[znacajke['phase'] == 'post'][['mae', 'id']]

X_estimation.to_csv("X_estimation_loocv.csv", index=False, float_format='%.6f')
y_estimation.to_csv("y_estimation_loocv.csv", index=False, float_format='%.6f')
X_prediction_startle.to_csv("X_prediction_startle_loocv.csv", index=False, float_format='%.6f')
y_prediction_startle.to_csv("y_prediction_startle_loocv.csv", index=False, float_format='%.6f')
X_prediction_post.to_csv("X_prediction_post_loocv.csv", index=False, float_format='%.6f')
y_prediction_post.to_csv("y_prediction_post_loocv.csv", index=False, float_format='%.6f')

# X_prediction_post.drop('id', inplace=True, axis=1)

X_prediction_post.drop('id', axis=1).to_csv("X_prediction_post.csv", index=False, float_format='%.6f')
y_prediction_post.drop('id', axis=1).to_csv("y_prediction_post.csv", index=False, float_format='%.6f')

