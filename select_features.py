__author__ = "Koren Gast"

__author__ = "Koren Gast"

from utils.utils import join_assets, prepare_data
import pathlib
from models.random_forest import RandomForest
from models.MLP import MLP
from models.lstm import LSTM_classifier, LSTM_regressor
from models.conv1d import conv1D_model
from models.adaBoost import AdaBoost
import pandas as pd
import numpy as np
from datetime import datetime
from joblib import Parallel, delayed

CUTOFF = 0.15  # in percents. The minimal value of ascending
# N_ESTIMATORS = [50, 100, 300]
N_ESTIMATORS = [100]
EPOCHS = 10
MOUNTH_DATA_ROWS = int(30 * 24 * (60 / 5))
# s_date = '31 Jan, 2017'
s_date = '01 Jan, 2018'
# e_date = '31 Jan, 2019'
e_date = '31 Jan, 2018'
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
# symbols = ['NEOUSDT']
pull_interval = '5M'
data_interval = '30M'
data_intervals = pull_interval + '_' + data_interval
# symbols_to_predict = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
symbols_to_predict = ['NEOUSDT']
merging = 6  # Should be equal to data_interval/pull_interval
models = dict()

class_weight = {
    -5: 50,
    -2: 20,
    -1: 10,
    0: 1,
    1: 10,
    2: 20,
    5: 50,
}

for n_est in N_ESTIMATORS:
    # pass
    models['RandomForest_' + str(n_est)] = RandomForest(n_est, class_weight)
    # models['AdaBoost_' + str(n_est)] = AdaBoost(n_est, class_weight)
# models['MLP'] = MLP()

kl_file_names = []
for symbol in symbols:
    print(symbol)
    kl_f_name = symbol + '_' + s_date + '_TO_' + e_date + '.csv'
    kl_file_names.append(kl_f_name)
features_file_names = []

for kl_f in kl_file_names:
    print(kl_f)
    features_f_name = kl_f
    features_file_names.append(features_f_name)

features = join_assets(symbols, features_file_names, data_intervals, is_features=True)

s2pred = 'NEOUSDT'

# First - BTCUSDT_volume (measure = 1.09)
# Second - BNBUSDT_close_ratio (measure = 1.16)
# Third - LTCUSDT_close_ratio (measure = 1.175)
# Forth - LTCUSDT_volume (measure = 1.191)
# Fifth - 'LTCUSDT_R^2' (measure = 1.265)
base_cols = ['timestamp', 'NEOUSDT_close_ratio', 'NEOUSDT_R^2']
base_cols = base_cols + ['BTCUSDT_volume', 'BNBUSDT_close_ratio',
                         'LTCUSDT_close_ratio', 'LTCUSDT_volume', 'LTCUSDT_R^2']

model = models['RandomForest_100']

# for c in features.columns:
def loop(c):
    if c not in base_cols:
        cols = base_cols + [c]
        x_df = features[cols]
        x_df = x_df.dropna()
        X_train, X_valid, y_train, y_valid, df_train_y, df_valid, df_valid_y = \
            prepare_data(x_df, CUTOFF, s2pred, merging, is_features=True)

        model.fit(X=X_train, y=y_train)
        df_valid_y['predictions'] = model.predict(df_valid.drop('timestamp', axis=1))

        df_valid_y.to_csv('predictions/' + data_intervals + '/feature_selection/' + c + '.csv', index=False)


results = Parallel(n_jobs=10)(delayed(loop)(e) for e in features.columns)
print(features.columns)
print(features.shape)