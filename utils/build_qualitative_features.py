__author__ = "Koren Gast"
import pandas as pd
import copy
import numpy as np
import collections
import logging

MAX_WINDOW = 4

def Bolinger_Bands(close, window_size=420/5, num_of_std=2):

    rolling_mean = close.rolling(window=window_size).mean()
    rolling_std = close.rolling(window=window_size).std()
    upper_band = rolling_mean + (rolling_std*num_of_std)
    lower_band = rolling_mean - (rolling_std*num_of_std)

    return rolling_mean, upper_band, lower_band, rolling_std

def cal_by_condition(temp_data, condition):
    all_cols = []
    for col in temp_data:
        curr_col = temp_data[col].tolist()
        curr_col = [c if condition(c) else 0 for c in curr_col]
        all_cols.append(curr_col)

    all_cols_np = np.matrix(all_cols)
    sum_cols = all_cols_np.sum(axis=0).tolist()
    sum_cols = sum_cols[0]
    return sum_cols

def compute_RSI(momentum_size, data):
    temp_data = copy.deepcopy(data)
    temp_data['diff'] = temp_data['close'] - temp_data['close'].shift(1)
    diff_aggregation = collections.OrderedDict()
    for i in range(0, momentum_size):
        diff_aggregation['diff' + str(-1 * i)] = temp_data['diff'].shift(i)

    diff_aggregation = pd.DataFrame(diff_aggregation)
    temp_data = temp_data.join(diff_aggregation)
    del temp_data['diff']
    cols = [c for c in temp_data.columns if c.startswith('diff')]
    temp_data = temp_data[cols]

    positive_condition = lambda x: x > 0
    negative_condition = lambda x: x < 0

    temp_data['gain'] = cal_by_condition(temp_data, positive_condition)
    temp_data['loss'] = cal_by_condition(temp_data, negative_condition)

    temp_data['RS'] = -1 * (temp_data['gain'].cumsum() / temp_data['loss'].cumsum())
    RSI = 100 - 100 / (1 + temp_data['RS'])
    return RSI

def build_features(data, merging):
    data = copy.deepcopy(data)
    data['close_ratio'] = data['close'] / data['close'].shift(1)

    data['mean_420'] = data['close'].rolling(window=int(420/5)).mean()
    data['std_420'] = data['close'].rolling(window=int(420/5)).std()

    data['RSI_420'] = compute_RSI(int(420/5), data)

    return data
