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

CUTOFF = 0.15  # in percents. The minimal value of ascending
# N_ESTIMATORS = [50, 100, 300]
N_ESTIMATORS = [10, 50, 100]
EPOCHS = 10
MOUNTH_DATA_ROWS = int(30 * 24 * (60 / 5))
s_date = '31 Jan, 2017'
# s_date = '01 Jan, 2019'
e_date = '31 Jan, 2019'
# e_date = '02 Jan, 2019'
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

base_cols = ['timestamp', 'NEOUSDT_close_ratio', 'NEOUSDT_R^2']
base_cols = base_cols + ['BTCUSDT_volume', 'BNBUSDT_close_ratio',
                         'LTCUSDT_close_ratio', 'LTCUSDT_volume', 'LTCUSDT_R^2']


print('a')
# features = join_assets(symbols, features_file_names, data_intervals, is_features=True)
features = join_assets(symbols, features_file_names, data_intervals, is_features=True)
features = features[base_cols]
features = features.dropna()
# features = []
jklines = join_assets(symbols, kl_file_names, data_intervals, is_features=False)

# models['LSTMc_' + str(EPOCHS)] = LSTM_classifier(jklines.shape[1] - 1, 7)
# models['LSTMr_' + str(EPOCHS)] = LSTM_regressor(jklines.shape[1] - 1, 7)
# TODO:
# models['conv1D'] = conv1D_model(jklines.shape[1] - 1, 10, 7)

TOTAL_DATA_ROWS = jklines.shape[0]
cross_data_endpoints = list(range(6 * MOUNTH_DATA_ROWS, TOTAL_DATA_ROWS, MOUNTH_DATA_ROWS))
# cross_data_endpoints = [TOTAL_DATA_ROWS]

for s2pred in symbols_to_predict:
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
            X_train, X_valid, y_train, y_valid, df_train_y, df_valid, df_valid_y = \
                prepare_data(cross_data, CUTOFF, s2pred, merging, is_features=is_features)
            print('d')
            if not is_keras:
                model.fit(X=X_train, y=y_train)
            else:
                # model.fit(X=X_train, y=y_train, epochs=EPOCHS)
                model.fit(X=X_train, y=np.array(df_train_y['y']), epochs=EPOCHS)
            df_valid_y['predictions'] = model.predict(df_valid.drop('timestamp', axis=1))
            if not is_keras:
                f_imp = [None] + list(model.get_feture_importances(X_train.shape[1])) + [None] * 6
                df_valid_y.loc['feature_importance'] = f_imp

            avg_inc_pred = np.mean(df_valid_y[df_valid_y['y_bins'] > 0]['predictions'])
            measure = np.mean(df_valid_y[df_valid_y['predictions'] > avg_inc_pred]['y_bins']) - np.mean(
                df_valid_y['y_bins'])
            measure = measure / np.mean(df_valid_y['y_bins'])
            measure = round(float(measure) * 100, 2)

            pathlib.Path('predictions/' + data_intervals).mkdir(exist_ok=True)
            pred_file_name = model_name + '_' + s2pred
            l = cross_data.shape[0]

            result['train_s_date'][0].append(datetime.utcfromtimestamp(cross_data.iloc[0]['timestamp'] / 1000))
            result['valid_s_date'][0].append(datetime.utcfromtimestamp(cross_data.iloc[int(0.9 * l)]['timestamp'] / 1000))
            result['valid_e_date'][0].append(datetime.utcfromtimestamp(cross_data.iloc[l - 1]['timestamp'] / 1000))
            result['measure'][0].append(measure)
            result['avg_y%'][0].append(np.mean(df_valid_y[df_valid_y['predictions'] > avg_inc_pred]['y%']))
            result['value_to_buy'][0].append(avg_inc_pred)

            dates = str(cross_data.iloc[int(0.9 * l)]['timestamp']) + '_' + str(cross_data.iloc[l-1]['timestamp'])
            pathlib.Path('predictions/' + data_intervals + '/qualitative/' + model_name).mkdir(exist_ok=True)
            # df_valid_y.to_csv('try.csv')
            df_valid_y.to_csv('predictions/' + data_intervals + '/qualitative/' + model_name + '/' + pred_file_name +
                              '_' + str(measure) + '_' + dates + '.csv', index=False)
        # result = pd.DataFrame(result)
        # result['total_measure'] = np.mean(result.iloc[0]['measure'])
        # result['total_y%'] = np.mean(result.iloc[0]['avg_y%'][0])
        # result.to_csv('results.csv', mode='a', header=False, index=False)


print('Done')
