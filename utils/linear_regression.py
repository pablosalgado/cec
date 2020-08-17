import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn
import numpy as np

TRIALS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BATCH_SIZE = [2, 4, 8, 16, 32]
TIME_STEPS = [6, 12, 24]

data = pd.DataFrame()

for trial in TRIALS:
    for batch_size in BATCH_SIZE:
        for time_steps in TIME_STEPS:
            log_file = f"../models/trial-{trial:02}/{batch_size}/{time_steps}/log.csv"

            if not os.path.isfile(log_file):
                continue

            print(f"Loading log file: {log_file}")

            data_n = pd.read_csv(log_file)

            data_n.insert(0, 'trial', trial)
            data_n.insert(1, 'batch_size', batch_size)
            data_n.insert(2, 'time_steps', time_steps)

            # Validation accuracy linear regression
            X = data_n.iloc[:, 3].values.reshape(-1, 1)
            Y = data_n.iloc[:, 6].values.reshape(-1, 1)
            linear_regressor = LinearRegression()
            linear_regressor.fit(X, Y)
            data_n['val_acc_slope'] = linear_regressor.coef_.max()

            # plt.scatter(X, Y)
            # plt.plot(X, linear_regressor.predict(X), color='red')
            # plt.title("Validation accuracy linear regression ")
            # plt.ylabel('Accuracy')
            # plt.xlabel('Epoch')
            # plt.grid(True)
            # plt.savefig(f"val_acc_slope-{trial:02}-{batch_size:02}-{time_steps:02}.png")
            # plt.close()

            # Validation loss linear regression
            X = data_n.iloc[:, 3].values.reshape(-1, 1)
            Y = data_n.iloc[:, 7].values.reshape(-1, 1)
            linear_regressor = LinearRegression()
            linear_regressor.fit(X, Y)
            data_n['val_loss_slope'] = linear_regressor.coef_.max()

            # plt.scatter(X, Y)
            # plt.plot(X, linear_regressor.predict(X), color='red')
            # plt.title("Validation loss linear regression ")
            # plt.ylabel('Loss')
            # plt.xlabel('Epoch')
            # plt.grid(True)
            # plt.savefig(f"val_loss_slope-{trial:02}-{batch_size:02}-{time_steps:02}.png")
            # plt.close()

            # Validation loss linear regression normalization
            scaler = sklearn.preprocessing.MinMaxScaler()
            N = scaler.fit_transform(Y)
            linear_regressor = LinearRegression()
            linear_regressor.fit(X, N)
            data_n['val_loss_scaled'] = N

            # plt.scatter(X, N)
            # plt.plot(X, linear_regressor.predict(X), color='red')
            # plt.title("Validation scaled loss linear regression ")
            # plt.ylabel('Accuracy')
            # plt.xlabel('Epoch')
            # plt.grid(True)
            # plt.savefig(f"val_loss_n_slope-{trial:02}-{batch_size:02}-{time_steps:02}.png")
            # plt.close()

            data = data.append(data_n)

data = data.reset_index(drop=True)
data.to_csv('cec-trials.csv')