__author__ = "Koren Gast"
import glob
import pandas as pd
import numpy as np

THRESHOLDS = np.linspace(0, 0.2, 101)
path = 'predictions\\5M_30M\\qualitative\\RandomForest_100'
allFiles = glob.glob(path + "/*.csv")

cols = ['y', 'y%', 'y_bins', 'predictions']


def comul_long(threshold, results_df):
    val = 1
    for i in range(results_df.shape[0]):
        if results_df.iloc[i]['predictions'] > threshold:
            val = val * results_df.iloc[i]['y'] * (1 - 0.00075)
    return val


def comul_short(threshold, results_df):
    val = 1
    for i in range(results_df.shape[0]):
        if results_df.iloc[i]['predictions'] < -threshold:
            val = val / results_df.iloc[i]['y'] * (1 - 0.00075)
    return val


for th in THRESHOLDS:
    longs = []
    shorts = []
    transactions = []
    for fn in allFiles:
        res_df = pd.read_csv(fn, usecols=cols)
        res_df = res_df.iloc[:-7]
        long_prof = np.mean(res_df[res_df['predictions'] > th]['y']) - 0.00075
        short_prof = np.mean(res_df[res_df['predictions'] < -th]['y']) - 0.00075
        transactions.append(res_df[res_df['predictions'] > th].shape[0] + res_df[res_df['predictions'] < -th].shape[0])
        f = fn.replace(path, '')
        # print('File: {}, longs: {}, shorts: {}'.format(f, round(long_prof, 4),
        #                                                               round(short_prof, 4)))

        longs.append(long_prof)
        shorts.append(short_prof)
    # print('#############################################')
    longs = np.array(longs)
    shorts = np.array(shorts)
    longs = longs[~np.isnan(longs)]
    shorts = shorts[~np.isnan(shorts)]
    total = round((np.mean(longs) * np.mean(shorts)-1)*100, 4)
    print('Threshold: {}, avg long: {}, avg short: {}, #Transactions: {}, Total: {}%'.format(round(th, 4),
                                                                                            round(np.mean(longs), 4),
                                                                                            round(np.mean(shorts), 4),
                                                                                            round(np.mean(transactions),
                                                                                                  4),
                                                                                            total))
    # print('#############################################')
