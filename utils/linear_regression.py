import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

TRIALS = [1, 2, 3, 4, 5, 6]
BATCH_SIZE = [2, 4, 8, 16, 32]
TIME_STEPS = [6, 12, 24]

data = pd.DataFrame()

for trial in TRIALS:
    for batch_size in BATCH_SIZE:
        for time_steps in TIME_STEPS:
            log_file = f"../models/trial-0{trial}/{batch_size}/{time_steps}/log.csv"

            if not os.path.isfile(log_file):
                continue

            data_n = pd.read_csv(log_file)

            data_n.insert(0, 'trial', trial)
            data_n.insert(1, 'batch_size', batch_size)
            data_n.insert(2, 'time_steps', time_steps)

            X = data_n.iloc[:, 3].values.reshape(-1, 1)
            Y = data_n.iloc[:, 6].values.reshape(-1, 1)
            linear_regressor = LinearRegression()
            linear_regressor.fit(X, Y)
            data_n['val_acc_slope'] = linear_regressor.coef_.max()

            # plt.scatter(X, Y)
            # plt.plot(X, linear_regressor.predict(X), color='red')
            # plt.show()

            X = data_n.iloc[:, 3].values.reshape(-1, 1)
            Y = data_n.iloc[:, 7].values.reshape(-1, 1)
            linear_regressor = LinearRegression()
            linear_regressor.fit(X, Y)
            data_n['val_loss_slope'] = linear_regressor.coef_.max()

            data = data.append(data_n)

data.to_csv('cec-trials.csv')