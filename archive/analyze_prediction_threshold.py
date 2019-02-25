__author__ = "Koren Gast"
import pandas as pd
from matplotlib import pyplot as plt
import glob
import numpy as np
from joblib import Parallel, delayed


def comul_long(threshold, results_df):
    val = 1
    for i in range(results_df.shape[0]):
        if results_df.iloc[i]['predictions'] > threshold:
            val = val * results_df.iloc[i]['y']
    return val


def comul_short(threshold, results_df):
    val = 1
    for i in range(results_df.shape[0]):
        if results_df.iloc[i]['predictions'] < threshold:
            val = val / results_df.iloc[i]['y']
    return val


def calc_th_df(posible_thresholds, results_df, is_long):
    vals = []
    for ps in posible_thresholds:
        if is_long:
            vals.append(comul_long(ps, results_df))
        else:
            vals.append(comul_short(ps, results_df))
    return vals


posible_thresholds = np.linspace(-0.05, 0.05, 101)
# posible_thresholds = np.linspace(-0.05, 0.05, 3)
path = 'predictions/5M_30M/'
models = ['RandomForest_50/', 'RandomForest_100/', 'RandomForest_300/',
          'AdaBoost_50/', 'AdaBoost_100/', 'AdaBoost_300/',
          'LSTM_50/']
# models = ['RandomForest_50/']
cols = ['y', 'y%', 'y_bins', 'predictions']
SYMBOLS = ['BNBUSDT', 'ETHUSDT', 'BTCUSDT', 'LTCUSDT', 'NEOUSDT']
# SYMBOLS = ['BNBUSDT']


# for model in models:
def analyze_model(model):
    allFiles = glob.glob(path + model + "/*.csv")
    allFiles = [f for f in allFiles if '_th' not in f]
    print(model)
    for symbol in SYMBOLS:
        print(symbol)
        for fn in allFiles:
            print(fn)
            if symbol in fn:
                res_df = pd.read_csv(fn, usecols=cols)
                res_df = res_df.iloc[:-7]
                long_ths = calc_th_df(posible_thresholds, res_df, is_long=True)
                long_csv = pd.read_csv(path + model + symbol + '_long_th.csv')
                if 'Unnamed: 0' in long_csv.columns:
                    long_csv = long_csv.set_index('Unnamed: 0')
                long_csv.loc[fn] = long_ths
                long_csv.to_csv(path + model + symbol + '_long_th.csv')
                short_ths = calc_th_df(posible_thresholds, res_df, is_long=False)
                short_csv = pd.read_csv(path + model + symbol + '_short_th.csv')
                if 'Unnamed: 0' in short_csv.columns:
                    short_csv = short_csv.set_index('Unnamed: 0')
                short_csv.loc[fn] = short_ths
                short_csv.to_csv(path + model + symbol + '_short_th.csv')


Parallel(n_jobs=10)(delayed(analyze_model)(e) for e in models)
