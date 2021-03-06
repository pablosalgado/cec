import tensorflow as tf
import pandas as pd
import common

class History:
    history = {}


history = History()

df = pd.read_csv('../models/trial-final/32/12/log.csv')
df = df[['accuracy', 'loss', 'val_accuracy', 'val_loss']]
d = df.to_dict()

for x in d:
    history.history[x] = [v for k,v in d[x].items()]

common.plot_acc_loss(history, '../models/trial-final/32/12/plot.png')