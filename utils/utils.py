__author__ = "Koren Gast"
from binance.client import Client
import pandas as pd
import copy
import numpy as np
from utils.api_keys import api_private, api_public
from scipy.stats import linregress


def get_klines(s_date, e_date, symbol, interval):
    client = Client(api_public, api_private)

    klines = client.get_historical_klines(symbol, interval, s_date, e_date)

    filtered_klines = []
    for k in klines:
        new_list = k[:6] + [k[-4]]
        new_list = [float(n) for n in new_list]
        filtered_klines.append(new_list)

    df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '#trades'],
                      data=filtered_klines)

    return df


def merge_klines(klines, merge_amount):
    merged_klines = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', '#trades', 'R^2'])

    def one_merge(df):
        row = dict()
        row['timestamp'] = df['timestamp'].iloc[0]
        row['open'] = df['open'].iloc[0]
        row['high'] = np.max(df['high'])
        row['low'] = np.min(df['low'])
        row['close'] = df['close'].iloc[-1]
        row['volume'] = np.sum(df['volume'])
        row['#trades'] = np.sum(df['#trades'])
        x = np.linspace(0, 1, df.shape[0])
        slope, intercept, r_value, p_value, std_err = linregress(x, df['close'])
        row['R^2'] = r_value ** 2
        return row

    for i in range(klines.shape[0] - merge_amount):
        merged_klines = merged_klines.append(one_merge(klines.iloc[i:i + merge_amount]), ignore_index=True)

    return merged_klines


def add_ys(df, cutoff, symbol, merging, is_features):
    new_df = copy.deepcopy(df)
    if not is_features:
        new_df[symbol+'_close_ratio'] = new_df[symbol+'_close'] / new_df[symbol+'_close'].shift(1)
    new_df['y'] = new_df[symbol+'_close_ratio'].shift(-1*merging)
    new_df['y_R^2'] = new_df[symbol+'_R^2'].shift(-1*merging)
    def condition(x):
        x = (x - 1) * 100
        bin = 0
        if x < -5 * cutoff:
            bin = -5
        elif -5 * cutoff < x < -2 * cutoff:
            bin = -2
        elif -2 * cutoff < x < -1*cutoff:
            bin = -1
        elif cutoff < x < 2 * cutoff:
            bin = 1
        elif 2 * cutoff < x < 5 * cutoff:
            bin = 2
        elif 5 * cutoff < x:
            bin = 5
        return bin

    new_df['y%'] = (new_df['y'] - 1) * 100
    new_df['y*r2'] = ((new_df['y%'] * new_df['y_R^2']) / 100) + 1
    new_df['y_bins'] = new_df['y*r2'].apply(condition)
    return new_df

def drop_unimportants(df):
    cols = list(df.columns)
    unimportnats = [x for x in cols if 'min' in x
                    or 'max' in x
                    or 'high_ratio' in x
                    or 'low_ratio' in x]
    return df.drop(unimportnats, axis=1)

def join_assets(assets, file_names, intervals, is_features=True):
    merged = pd.DataFrame()
    for a, f in zip(assets, file_names):
        print(a)
        if is_features:
            df = pd.read_csv('features/' + intervals + '/' + f)
        else:
            df = pd.read_csv('klines/' + intervals + '/' + f)
        df = drop_unimportants(df)
        df.columns = [a + '_' + str(col) if str(col) != 'timestamp' else str(col) for col in df.columns]

        if merged.shape == (0, 0):
            merged = df
        else:
            merged = merged.merge(df,
                                  left_on='timestamp',
                                  right_on='timestamp',)
    if is_features:
        merged = merged[DropCorrelated(merged, 0.95)]
    return merged


def prepare_data(features, CUTOFF, s2pred, merging, is_features=True):
    features_y = add_ys(features, CUTOFF, s2pred, merging, is_features)
    print('x1')
    l = features.shape[0]
    print('x2')
    X_train = np.array(features.iloc[:int(0.9 * l)].drop('timestamp', axis=1))
    print('x3')
    X_valid = np.array(features.iloc[int(0.9 * l):].drop('timestamp', axis=1))
    print('x4')
    y_train = np.array(features_y['y_bins'].iloc[:int(0.9 * l)])
    print('x5')
    y_valid = np.array(features_y['y_bins'].iloc[int(0.9 * l):])
    print('x6')
    df_valid = features.iloc[int(0.9 * l):]
    print('x7')
    df_valid_y = features_y.iloc[int(0.9 * l):]
    df_train_y = features_y.iloc[:int(0.9 * l)]
    print('x8')
    return X_train, X_valid, y_train, y_valid, df_train_y, df_valid, df_valid_y

def DropCorrelated(data, corr_threshold):
    corr_mat = data.corr(method='pearson')
    corr_mat.loc[:, :] = np.tril(corr_mat, k=-1)
    already_in = set()
    result = []
    for col in corr_mat:
        perfect_corr = corr_mat[col][corr_mat[col] > corr_threshold].index.tolist()
        if col not in already_in:
            perfect_corr.append(col)
            already_in.update(set(perfect_corr))
            result.append(col)
        # select_nested = [f[1:] for f in result]
        # select_flat = [i for j in select_nested for i in j]
    return result

def to_categorical(arr):
    uniques = np.unique(arr)
    n_values = uniques.shape[0]
    one_hot = np.eye(n_values)[arr.reshape(-1)]
    oh_dict_array = np.eye(n_values)[uniques]
    oh_dict = dict()
    for i in range(0,n_values):
        oh_dict[uniques[i]] = oh_dict_array[i,:]
    return one_hot, oh_dict




client_intervals = {
    '1M': Client.KLINE_INTERVAL_1MINUTE,
    '5M': Client.KLINE_INTERVAL_5MINUTE,
    '30M': Client.KLINE_INTERVAL_30MINUTE, }
