from __future__ import division

import collections
import copy
import logging
import os

import numpy as np
import pandas as pd

from storage.kline import pull

FEATURES_DIRECTORY = './data/features'
FEATURES_LENGTH = 1024



def get_path_by_interval_and_symbol(interval, to_interval, symbol):
    if to_interval is not None and to_interval != interval:
        return os.path.abspath(FEATURES_DIRECTORY + '/' + interval + '/' + to_interval + '/' + symbol + '.csv').replace('web/', '')
    else:
        return os.path.abspath(FEATURES_DIRECTORY + '/' + interval + '/' + symbol + '.csv').replace('web/', '')


def append_new_line(new_data_row, interval, symbol, to_interval=None):
    features_path = get_path_by_interval_and_symbol(interval, to_interval, symbol)
    with open(features_path, 'a') as f:
        new_data_row.to_csv(f, header=False, index=False)

    # data = pd.read_csv(features_path)
    # data = pd.concat([data, new_data_row], ignore_index=True)
    # data.to_csv(features_path, index=False)


def split_data_by_interval(kline_data, split_interval):

    new_df_data = []

    for i in range(split_interval):
        new_df_data.append([])

    for index in range(kline_data.shape[0]):

        df_pos = index % split_interval
        new_df_data[df_pos].append(kline_data.iloc[index].values)

    for i in range(split_interval):
        new_df_data[i] = pd.DataFrame(new_df_data[i], columns=kline_data.columns)

    return new_df_data


def unify_all_data(splitted_data):

    unify_data = []

    for i in range(max(s.shape[0] for s in splitted_data)):
        for split in splitted_data:
            if i < split.shape[0]:
                unify_data.append(split.iloc[i])
    unify_data = pd.DataFrame(unify_data)
    unify_data = unify_data.reset_index(drop=True)
    return unify_data


interval_to_split = {
    '5m' : {
        '10m': 2,
        '15m': 3,
        '20m': 4,
        '25m': 5,
        '30m': 6,
        '35m': 7,
        '40m': 8,
        '45m': 9,
        '50m': 10,
        '55m': 11,
        '60m': 12,
    },
    '15m': {
        '30m': 2,
        '45m': 3,
        '60m': 4,
        '75m': 5,
        '90m': 6,
        '105m': 7,
        '120m': 8,
        '135m': 9,
        '150m': 10,
        '165m': 11,
        '180m': 12,
    },
    '30m': {
        '60m': 2,
        '90m': 3,
        '120m': 4,
        '150m': 5,
        '180m': 6,
        '210m': 7,
        '240m': 8,
        '270m': 9,
        '300m': 10,
        '330m': 11,
        '360m': 12,
    }
}
def convert_klines(klines_data, interval_diff):



    klines_data['volume'] = klines_data['volume'].rolling(interval_diff).sum()
    klines_data['order_book_volume'] = klines_data['order_book_volume'].rolling(interval_diff).sum()
    klines_data['Number of trades'] = klines_data['Number of trades'].rolling(interval_diff).sum()
    klines_data['high'] = klines_data['high'].rolling(interval_diff).max()
    klines_data['low'] = klines_data['low'].rolling(interval_diff).min()
    klines_data['open'] = klines_data['open'].shift(interval_diff)
    klines_data = klines_data[interval_diff:]
    return klines_data


def build_features(interval, to_interval, symbol):
    klines_path = pull.get_path_by_interval_and_symbol(interval, symbol)
    if not os.path.exists(klines_path):
        raise Exception("No klines data was found for this interval/symbol")

    features_path = get_path_by_interval_and_symbol(interval, to_interval, symbol)
    features_data = None
    kline_data = pd.read_csv(klines_path)

    if to_interval != interval:
        kline_data = convert_klines(kline_data, interval_to_split[interval][to_interval])

    if os.path.exists(features_path):
        features_data = pd.read_csv(features_path)
        if len(features_data) > 0:
            last_features_time = features_data.tail(1)['datetime'].tolist()[0]
            if last_features_time is not None:
                datetime_data_index = kline_data['datetime'].tolist().index(last_features_time)

                if to_interval != interval:
                    kline_data = kline_data[max(datetime_data_index - (FEATURES_LENGTH + 6) * interval_to_split[interval][to_interval], 0):]
                else:
                    kline_data = kline_data[max(datetime_data_index - (FEATURES_LENGTH + 6), 0):]

        else:
            features_data = None

    if to_interval != interval:
        splitted_kline_data = split_data_by_interval(kline_data, interval_to_split[interval][to_interval])
        splitted_features_data = []

        for data in splitted_kline_data:

            splitted_features_data.append(build_features_by_data(data))
        data = unify_all_data(splitted_features_data)
    else:
        data = build_features_by_data(kline_data)

    if features_data is not None:
        features_data = pd.read_csv(features_path)  # refresh (don't del)
        last_curr = features_data['datetime'].tolist()[-1]
        print("Last curr in original data is {0}".format(last_curr))

        with open(features_path, 'a') as f:
            print ("XXXXXXXXXXXX APPENDING FEATURES", features_path)

            if data.shape[0] != 0:
                data[data['datetime'] > last_curr].to_csv(f, header=False, index=False)
                print("Last curr in updated data is {0}".format(data['datetime'].tolist()[-1]))
            else:
                print("Data is {0}".format(data.shape))
    else:
        directory_path = features_path[:features_path.rfind('/')]
        try:
            os.stat(directory_path)
        except Exception as e:
            os.makedirs(directory_path)

        print ("XXXXXXXXXXXX SAVING FEATURES", features_path)
        data.to_csv(features_path, index=False)
        return list(data.columns)


def build_features_by_data(data):
    data = copy.deepcopy(data)
    data['last_close'] = data['close'].shift(1)
    data['close_diff'] = (data['close'] / data['last_close'] - 1) * 100
    data['close/high'] = data['close'] / data['high']
    data['close/low'] = data['close'] / data['low']

    def cal_agol(open, close):
        res = []
        for o, c in zip(open, close):
            res.append(int(o / 1000) - int(c / 1000))
        return res

    data['is_agol_changed'] = cal_agol(data['close'].tolist(), data['open'].tolist())

    # RSI:
    def compute_RSI(momentum_size):
        temp_data = copy.deepcopy(data)
        temp_data['diff'] = temp_data['close'] - temp_data['last_close']
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

        temp_data['gain'] = cal_by_condition(temp_data, positive_condition)
        temp_data['loss'] = cal_by_condition(temp_data, negative_condition)

        temp_data['RS'] = -1 * (temp_data['gain'].cumsum() / temp_data['loss'].cumsum())
        data['RSI_' + str(momentum_size)] = 100 - 100 / (1 + temp_data['RS'])

    # Stochastic oscillator
    def compute_stochasticOscillator(momentum_size):
        temp_data = copy.deepcopy(data)
        temp_data['L'] = temp_data['low'].rolling(momentum_size).min()
        temp_data['H'] = temp_data['high'].rolling(momentum_size).max()
        cols = ['close', 'H', 'L']
        temp_data = temp_data[cols]
        data['%K_' + str(momentum_size)] = 100 * (temp_data['close'] - temp_data['L']) / (
            temp_data['H'] - temp_data['L'])

    # Williams_R
    def compute_williamsR(momentum_size):
        temp_data = copy.deepcopy(data)
        temp_data['L'] = temp_data['low'].rolling(momentum_size).min()
        temp_data['H'] = temp_data['high'].rolling(momentum_size).max()
        cols = ['close', 'H', 'L']
        temp_data = temp_data[cols]
        data['%R_' + str(momentum_size)] = -100 * (temp_data['H'] - temp_data['close']) / (
            temp_data['H'] - temp_data['L'])

    # PROC
    def compute_PROC(momentum_size):
        temp_data = copy.deepcopy(data)
        temp_data['close' + str(-1 * momentum_size)] = temp_data['close'].shift(momentum_size)
        cols = ['close', 'close' + str(-1 * momentum_size)]
        temp_data = temp_data[cols]
        data['PROC_' + str(momentum_size)] = (temp_data['close'] - temp_data[
            'close' + str(-1 * momentum_size)]) / \
                                             temp_data['close' + str(-1 * momentum_size)]

    def enter_new_col(main_col, action, window_size, win):
        if win == '':
            col_name = main_col + '_' + action + '_' + str(window_size)
        else:
            col_name = main_col + '_' + action + '_' + str(window_size) + '_' + win
        if action == 'mean':
            if win is not '':
                data[col_name] = data[main_col].rolling(window_size, win_type=win).mean()
            else:
                data[col_name] = data[main_col].rolling(window_size).mean()
        if action == 'sum':
            if win is not '':
                data[col_name] = data[main_col].rolling(window_size, win_type=win).sum()
            else:
                data[col_name] = data[main_col].rolling(window_size).sum()
        if action == 'max':
            if win is not '':
                data[col_name] = data[main_col].rolling(window_size, win_type=win).max()
            else:
                data[col_name] = data[main_col].rolling(window_size).max()
        if action == 'min':
            if win is not '':
                data[col_name] = data[main_col].rolling(window_size, win_type=win).min()
            else:
                data[col_name] = data[main_col].rolling(window_size).min()
        # OBV
        temp_data = copy.deepcopy(data)
        obvs = []

        def calc_obv(temp_data):

            volume = temp_data['volume'].tolist()
            close = temp_data['close'].tolist()
            last_close = temp_data['last_close'].tolist()

            for v, c, l in zip(volume, close, last_close):
                obv = v if not obvs else obvs[-1]
                if c > l:
                    obv = obv + v
                if c < l:
                    obv = obv - v
                obvs.append(obv)

        calc_obv(temp_data)
        data['OBV_0'] = obvs

        if temp_data.shape[1] % 1000 == 0:
            print(temp_data.shape)

    length = FEATURES_LENGTH + 1
    momentous = [n for n in range(0, length) if bin(n).count('1') == 1]
    actions = ['mean', 'sum', 'min', 'max']
    # there is no max / min for triang - leave it empty.
    wintypes = ['']
    features = ['close/high', 'close/low', 'volume', 'close_diff']
    for momentum_size in momentous:
        compute_RSI(momentum_size)
        compute_stochasticOscillator(momentum_size)
        compute_williamsR(momentum_size)
        compute_PROC(momentum_size)
        if 'OBV_0' in data.columns:
            data['OBV_' + str(momentum_size)] = data['OBV_0'].shift(momentum_size)
        for action in actions:
            for feature in features:
                for win in wintypes:
                    enter_new_col(feature, action, momentum_size, win)

    simple_features = ['open', 'close', 'high', 'low', 'volume', 'Number of trades', 'order_book_volume']
    for feature in simple_features:
        for momentum_size in momentous:
            col_name = feature + '_diff_' + str(momentum_size)
            data[col_name] = (data[feature] / data[feature].shift(momentum_size) - 1) * 100

    data = data[max(momentous):]
    for col in data:
        if col not in ['datetime', 'symbol']:
            numpy_col = np.array(data[col].astype(np.float32))
            inf_res = np.isinf(numpy_col).tolist()
            inf_res = [ele for ele in inf_res if ele is True]

            none_res = np.isnan(numpy_col).tolist()
            none_res = [ele for ele in none_res if ele is True]

            if inf_res:
                logging.warning(
                    "Total in NP.INF OR -NP.INF are {0} at column {1}".format(len(inf_res), col))
            if none_res:
                logging.warning(
                    "Total in NANS are {0} at column {1}".format(len(none_res), col))
            if inf_res or none_res:
                numpy_col = np.nan_to_num(numpy_col)
            data[col] = numpy_col
    return data

#
# x = pd.read_csv('/Users/Avi/PycharmProjects/AlgoBitTrading/data/klines/30m/BCCUSDT.csv')
# x = x[-1030:]
#
# import datetime
#
# start = datetime.datetime.now()
# x = build_features_by_data(x)
# print(x.shape[0])
# end = datetime.datetime.now()
#
# print((end - start).total_seconds())
