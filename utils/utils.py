__author__ = "Koren Gast"
from binance.client import Client
import pandas as pd
import copy
import numpy as np
from utils.api_keys import api_private, api_public
from scipy.stats import linregress

FEATURES_LENGTH = 10


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


def add_ys(df, cutoff, symbol, merging):
    new_df = copy.deepcopy(df)
    new_df['y'] = new_df[symbol+'_close_ratio'].shift(-1*merging)
    new_df['y_R^2'] = new_df[symbol+'_R^2'].shift(-1*merging)
    def condition(x):
        x = (x - 1) * 100
        bin = 0
        if x < -5 * cutoff:
            bin = -5
        elif -5 * cutoff < x < -2 * cutoff:
            bin = -2
        elif -2 * cutoff < x < cutoff:
            bin = -1
        elif cutoff < x < 2.5 * cutoff:
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


def merge_assets(assets, file_names, interval):
    dfs = dict()
    for a, f in zip(assets, file_names):
        dfs[a] = pd.read_csv('features/' + interval + '/' + f)
        dfs[a].columns = [a + '_' + str(col) if str(col) != 'timestamp' else str(col) for col in dfs[a].columns]
    merged = pd.DataFrame()
    for k in dfs.keys():
        if merged.shape == (0, 0):
            merged = dfs[k]
        else:
            merged = merged.merge(dfs[k],
                                  left_on='timestamp',
                                  right_on='timestamp',)
    return merged

def prepare_data(features, CUTOFF, s2pred, merging):
    features_y = add_ys(features, CUTOFF, s2pred, merging)
    l = features.shape[0]
    X_train = np.array(features.iloc[:int(0.8 * l)].drop('timestamp', axis=1))
    X_valid = np.array(features.iloc[int(0.8 * l):].drop('timestamp', axis=1))
    y_train = np.array(features_y['y_bins'].iloc[:int(0.8 * l)])
    y_valid = np.array(features_y['y_bins'].iloc[int(0.8 * l):])
    df_valid = features.iloc[int(0.8 * l):]
    df_valid_y = features_y.iloc[int(0.8 * l):]
    return X_train, X_valid, y_train, y_valid, df_valid, df_valid_y


client_intervals = {
    '1M': Client.KLINE_INTERVAL_1MINUTE,
    '5M': Client.KLINE_INTERVAL_5MINUTE,
    '30M': Client.KLINE_INTERVAL_30MINUTE, }
