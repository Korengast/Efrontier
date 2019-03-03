__author__ = "Koren Gast"

from utils.utils import join_assets, prepare_data, add_ys
import pathlib
from models.random_forest import RandomForest
from models.MLP import MLP
from models.lstm import LSTM_classifier, LSTM_regressor
from models.conv1d import conv1D_model
from models.adaBoost import AdaBoost
from models.features_selector import Selector
import pandas as pd
import numpy as np
import datetime as dt
import copy

CUTOFF = 0.15  # in percents. The minimal value of ascending
N_ESTIMATORS = [100]
EPOCHS = 10
MOUNTH_DATA_ROWS = int(30 * 24 * (60 / 5))
s_date = '31 Jan, 2017'
# s_date = '01 Jan, 2019'
e_date = '31 Jan, 2019'
# e_date = '02 Jan, 2019'
SYMBOLS_TO_USE = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
# SYMBOLS_TO_USE = ['NEOUSDT']
pull_interval = '5M'
data_interval = '30M'
data_intervals = pull_interval + '_' + data_interval
# SYMBOLS_TO_PREDICT = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
SYMBOLS_TO_PREDICT = ['NEOUSDT']
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
for symbol in SYMBOLS_TO_USE:
    print(symbol)
    kl_f_name = symbol + '_' + s_date + '_TO_' + e_date + '.csv'
    kl_file_names.append(kl_f_name)
features_file_names = []

for kl_f in kl_file_names:
    print(kl_f)
    features_f_name = kl_f
    features_file_names.append(features_f_name)

# base_cols = ['timestamp', 'NEOUSDT_close_ratio', 'NEOUSDT_R^2']
# base_cols = base_cols + ['BTCUSDT_volume', 'BNBUSDT_close_ratio',
#                          'LTCUSDT_close_ratio', 'LTCUSDT_volume', 'LTCUSDT_R^2']


print('a')
# features = join_assets(symbols, features_file_names, data_intervals, is_features=True)
features = join_assets(SYMBOLS_TO_USE, features_file_names, data_intervals, is_features=True)
# features = features[base_cols]
features = features.dropna()
# features = []
# jklines = join_assets(SYMBOLS_TO_USE, kl_file_names, data_intervals, is_features=False)
jklines = []

# models['LSTMc_' + str(EPOCHS)] = LSTM_classifier(jklines.shape[1] - 1, 7)
# models['LSTMr_' + str(EPOCHS)] = LSTM_regressor(jklines.shape[1] - 1, 7)
# TODO:
# models['conv1D'] = conv1D_model(jklines.shape[1] - 1, 10, 7)

TOTAL_DATA_ROWS = features.shape[0]
# cross_data_endpoints = list(range(6 * MOUNTH_DATA_ROWS, TOTAL_DATA_ROWS, MOUNTH_DATA_ROWS))
cross_data_endpoints = [TOTAL_DATA_ROWS]
y_cols = ['y', 'y_R^2', 'y%', 'y*r2', 'y_bins']

for s2pred in SYMBOLS_TO_PREDICT:
    for model_name in models.keys():
        is_keras = 'LSTM' in model_name or 'conv1D' in model_name
        result = {
            'model_symbol': [model_name + '_' + s2pred],
            'train_s_date': [[]],
            'valid_s_date': [[]],
            'valid_e_date': [[]],
            'measure': [[]],
            'avg_y%': [[]],
            'value_to_buy': [[]]
        }
        model = models[model_name]
        n_round = 0
        for ep in cross_data_endpoints:
            n_round += 1
            if not is_keras:
                cross_data = features.iloc[:ep]
                is_features = True
            else:
                cross_data = jklines.iloc[:ep]
                is_features = False

            cross_data = add_ys(cross_data, CUTOFF, s2pred, merging, is_features)
            # df_train = features.iloc[:-2 * MOUNTH_DATA_ROWS]
            # df_valid = features.iloc[-2 * MOUNTH_DATA_ROWS:-MOUNTH_DATA_ROWS]
            # df_test = features.iloc[-MOUNTH_DATA_ROWS:]

            df_train_y = cross_data.iloc[:-MOUNTH_DATA_ROWS]
            df_valid_y = cross_data.iloc[-int(1.5 * MOUNTH_DATA_ROWS):-MOUNTH_DATA_ROWS]
            df_test_y = cross_data.iloc[-MOUNTH_DATA_ROWS:]

            # l = cross_data.shape[0]
            # df_to_selector = cross_data.iloc[int((0.9*l)-(MOUNTH_DATA_ROWS / 2)):int(0.9*l)]
            n_est = 0
            if 'AdaBoost' in model_name:
                n_est = int(model_name.replace('AdaBoost_', ''))
            if 'RandomForest' in model_name:
                n_est = int(model_name.replace('RandomForest_', ''))

            feature_selector = Selector(model_name, df_valid_y, CUTOFF, s2pred, merging, n_est, class_weight)
            # l = ['soft_upper_dist', 'soft_lower_dist', 'soft_upper_broke', 'soft_lower_broke']
            # l = ['NEOUSDT_' + i for i in l]
            feature_selector.execute()
            df_train_y = df_train_y[feature_selector.get_cols() + y_cols]
            df_valid_y = df_valid_y[feature_selector.get_cols() + y_cols]
            df_test_y = df_test_y[feature_selector.get_cols() + y_cols]
            X_train = np.array(df_train_y.drop(['timestamp'] + y_cols, axis=1))
            X_test = np.array(df_test_y.drop(['timestamp'] + y_cols, axis=1))

            y_train = np.array(df_train_y['y_bins'])
            y_test = np.array(df_test_y['y_bins'])

            print('d')
            if not is_keras:
                model.fit(X=X_train, y=y_train)
            else:
                model.fit(X=X_train, y=np.array(df_train_y['y']), epochs=EPOCHS)
            df_valid_y['predictions'] = model.predict(df_valid_y.drop(['timestamp'] + y_cols, axis=1))
            df_test_y['predictions'] = model.predict(df_test_y.drop(['timestamp'] + y_cols, axis=1))
            if not is_keras:
                f_imp = [None] + list(model.get_feture_importances(X_train.shape[1])) + [None] * 6
                df_valid_y.loc['feature_importance'] = f_imp

            df_to_zero = df_valid_y[df_valid_y['predictions'] > 0.0]
            measure = np.mean(df_to_zero['y'])

            pathlib.Path('predictions/' + data_intervals).mkdir(exist_ok=True)
            pred_file_name = model_name + '_' + s2pred
            l = cross_data.shape[0]

            dates = dt.date.fromtimestamp(df_test_y['timestamp'].iloc[0]/1000).__str__() + '_' + \
                    dt.date.fromtimestamp(df_test_y['timestamp'].iloc[-1] / 1000).__str__()
            # dates = str(cross_data.iloc[int(0.9 * l)]['timestamp']) + '_' + str(cross_data.iloc[l - 1]['timestamp'])
            pathlib.Path('predictions/' + data_intervals + '/qualitative/' + model_name).mkdir(exist_ok=True)
            df_train_y.to_csv('predictions/5M_30M/qualitative/' + model_name + '/' + pred_file_name +
                              '_' + str(measure) + '_' + dates + '_TRAIN.csv', index=False)
            df_test_y.to_csv('predictions/5M_30M/qualitative/' + model_name + '/' + pred_file_name +
                             '_' + str(measure) + '_' + dates + '.csv', index=False)
print('Done')
