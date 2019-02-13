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
        new_list = k[:6] + [k[-4]]
        new_list = [float(n) for n in new_list]
        filtered_klines.append(new_list)

    df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'Number of trades'],
                      data=filtered_klines)

    return df


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
