__author__ = "Koren Gast"

from utils.utils import merge_assets, prepare_data
import pathlib
from models.random_forest import RandomForest
from models.MLP import MLP
from models.lstm import LSTM_model
from models.adaBoost import AdaBoost
import pandas as pd
import numpy as np

CUTOFF = 0.15  # in percents. The minimal value of ascending
N_ESTIMATORS = [50, 100, 300]
s_date = '31 Jan, 2018'
e_date = '31 Jan, 2019'
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
    pass
    # models['RandomForest_' + str(n_est)] = RandomForest(n_est, class_weight)
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

print('a')
features = merge_assets(symbols, features_file_names, data_intervals)
print('b')
# TODO: Prepare data for LSTM (no feature, just merging raw data)
models['LSTM'] = LSTM_model(features.shape[1]-1, 6)

for s2pred in symbols_to_predict:
    print('c')
    X_train, X_valid, y_train, y_valid, df_valid, df_valid_y = prepare_data(features, CUTOFF, s2pred, merging)
    print('d')
    for model_name in models.keys():
        model = models[model_name]
        model.fit(X=X_train, y=y_train)
        df_valid_y['predictions'] = model.predict(df_valid.drop('timestamp', axis=1))
        f_imp = [None] + list(model.get_feture_importances(X_train.shape[1])) + [None] * 6
        df_valid_y.loc['feature_importance'] = f_imp

        avg_inc_pred = np.mean(df_valid_y[df_valid_y['y_bins'] > 0]['predictions'])
        measure = np.mean(df_valid_y[df_valid_y['predictions'] > avg_inc_pred]['y_bins']) - np.mean(df_valid_y['y_bins'])
        measure = round(float(measure)*100, 2)

        pathlib.Path('predictions/' + data_intervals).mkdir(exist_ok=True)
        pred_file_name = model_name + '_' + s2pred
        df_valid_y.to_csv('predictions/' + data_intervals + '/' + pred_file_name +
                          '_' + str(measure) + '.csv', index=False)

print('Done')

