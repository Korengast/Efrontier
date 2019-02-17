__author__ = "Koren Gast"
import pandas as pd
import copy
import numpy as np
import collections
import logging

MAX_WINDOW = 16


def cal_agol(open, close):
    res = []
    for o, c in zip(open, close):
        res.append(int(o / 1000) - int(c / 1000))
    return res


def compute_RSI(momentum_size, data):
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

    temp_data['gain'] = cal_by_condition(temp_data, positive_condition)
    temp_data['loss'] = cal_by_condition(temp_data, negative_condition)

    temp_data['RS'] = -1 * (temp_data['gain'].cumsum() / temp_data['loss'].cumsum())
    RSI = 100 - 100 / (1 + temp_data['RS'])
    return RSI


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

    # Stochastic oscillator


def compute_stochasticOscillator(momentum_size, data):
    temp_data = copy.deepcopy(data)
    temp_data['L'] = temp_data['low'].rolling(momentum_size).min()
    temp_data['H'] = temp_data['high'].rolling(momentum_size).max()
    cols = ['close', 'H', 'L']
    temp_data = temp_data[cols]
    K = 100 * (temp_data['close'] - temp_data['L']) / (temp_data['H'] - temp_data['L'])
    return K

    # Williams_R


def compute_williamsR(momentum_size, data):
    temp_data = copy.deepcopy(data)
    temp_data['L'] = temp_data['low'].rolling(momentum_size).min()
    temp_data['H'] = temp_data['high'].rolling(momentum_size).max()
    cols = ['close', 'H', 'L']
    temp_data = temp_data[cols]
    R = -100 * (temp_data['H'] - temp_data['close']) / (temp_data['H'] - temp_data['L'])
    return R

    # PROC


def compute_PROC(momentum_size, data):
    temp_data = copy.deepcopy(data)
    temp_data['close' + str(-1 * momentum_size)] = temp_data['close'].shift(momentum_size)
    cols = ['close', 'close' + str(-1 * momentum_size)]
    temp_data = temp_data[cols]
    PROC = (temp_data['close'] - temp_data['close' + str(-1 * momentum_size)]) / \
           temp_data['close' + str(-1 * momentum_size)]
    return PROC


def compute_obv(data):
    temp_data = copy.deepcopy(data)
    volume = temp_data['volume'].tolist()
    close = temp_data['close'].tolist()
    last_close = temp_data['last_close'].tolist()
    obvs = []
    for v, c, l in zip(volume, close, last_close):
        obv = v if not obvs else obvs[-1]
        if c > l:
            obv = obv + v
        if c < l:
            obv = obv - v
        obvs.append(obv)
    return obvs


def getDuplicateColumns(df):
    '''
    Get a list of duplicate columns.
    It will iterate over all the columns in dataframe and find the columns whose contents are duplicate.
    :param df: Dataframe object
    :return: List of columns whose contents are duplicates.
    '''
    duplicateColumnNames = set()
    # Iterate over all the columns in dataframe
    for x in range(df.shape[1]):
        # Select column at xth index.
        col = df.iloc[:, x]
        # Iterate over all the columns in DataFrame from (x+1)th index till end
        for y in range(x + 1, df.shape[1]):
            # Select column at yth index.
            otherCol = df.iloc[:, y]
            # Check if two columns at x 7 y index are equal
            if col.equals(otherCol):
                duplicateColumnNames.add(df.columns.values[y])

    return list(duplicateColumnNames)


def build_features(data):
    data = copy.deepcopy(data)
    data['last_close'] = data['close'].shift(1)
    data['close_ratio'] = data['close'] / data['last_close']
    data['high/close'] = data['high'] / data['close']
    data['low/close'] = data['low'] / data['close']
    data['open/close'] = data['open'] / data['close']
    data['is_agol_changed'] = cal_agol(data['close'].tolist(), data['open'].tolist())

    length = MAX_WINDOW + 1
    momentous = ([n for n in range(0, length) if bin(n).count('1') == 1])
    norm_features = ['high/close', 'low/close', 'open/close', 'volume', 'close_ratio']
    simple_features = ['open', 'close', 'high', 'low', 'volume', '#trades']

    col_name = 'OBV'
    data[col_name] = compute_obv(data)

    for momentum_size in momentous:
        col_name = 'RSI_' + str(momentum_size)
        data[col_name] = compute_RSI(momentum_size, data)
        col_name = '%K_' + str(momentum_size)
        data[col_name] = compute_stochasticOscillator(momentum_size, data)
        col_name = '%R_' + str(momentum_size)
        data[col_name] = compute_williamsR(momentum_size, data)
        col_name = 'PROC_' + str(momentum_size)
        data[col_name] = compute_PROC(momentum_size, data)

        for feature in norm_features:
            col_name = feature + '_mean_' + str(momentum_size)
            data[col_name] = data[feature].rolling(momentum_size).mean()
            col_name = feature + '_min_' + str(momentum_size)
            data[col_name] = data[feature].rolling(momentum_size).min()
            col_name = feature + '_max_' + str(momentum_size)
            data[col_name] = data[feature].rolling(momentum_size).max()

        for feature in simple_features:
            col_name = feature + '_ratio_' + str(momentum_size)
            data[col_name] = data[feature] / data[feature].shift(momentum_size)
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
    data = data.drop(columns=getDuplicateColumns(data))
    return data


# TODO: exp_smoothing, Holt-Winters