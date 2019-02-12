__author__ = "Koren Gast"
from binance.client import Client
import pandas as pd
import copy
import numpy as np
import collections
import logging

FEATURES_LENGTH = 10


def get_klines(s_date, e_date, symbol, interval):
    api_public = 'LNjFxoD3e0Orsi8UxaXiVvlTxVkDCuQDrNOkft0q6sXihD3yLxuYGQ5GdEIFSx1Z'
    api_private = 'fij1o7AjRdOeksAqD42R1JAEEplPKyoD301RI3P3Chsl70ltTvL6JtHr2J2Pp68u'
    client = Client(api_public, api_private)

    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, s_date, e_date)

    filtered_klines = []
    for k in klines:
        new_list = k[1:6] + [k[-4]]
        new_list = [float(n) for n in new_list]
        filtered_klines.append(new_list)

    df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'Number of trades'],
                      data=filtered_klines)

    return df


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

    simple_features = ['open', 'close', 'high', 'low', 'volume', 'Number of trades']
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


def y_to_bins(df, y, cutoff):
    def condition(x):
        x = (x-1)*100
        bin = 0
        if x < -10 * cutoff:
            bin = -10
        elif -10 * cutoff < x < -5 * cutoff:
            bin = -5
        elif -5 * cutoff < x < cutoff:
            bin = -1
        elif cutoff < x < 5 * cutoff:
            bin = 1
        elif 5 * cutoff < x < 10 * cutoff:
            bin = 5
        elif 10 * cutoff < x:
            bin = 10
        return bin

    df['y_bins'] = df[y].apply(condition)
    df = df.drop(y, axis=1)
    return df
