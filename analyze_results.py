__author__ = "Koren Gast"
import glob
import pandas as pd
import numpy as np

THRESHOLDS = np.linspace(0.0, 0.2, 201)
# THRESHOLDS = [0.149]
path = 'predictions\\5M_30M\\qualitative\\RandomForest_100'
allFiles = glob.glob(path + "/*.csv")

cols = ['y', 'y%', 'y_bins', 'predictions']

# def comul_long(threshold, results_df):
#     val = 1
#     for i in range(results_df.shape[0]):
#         if results_df.iloc[i]['predictions'] > threshold:
#             val = val * results_df.iloc[i]['y'] * (1 - 0.00075)
#     return val
#
#
# def comul_short(threshold, results_df):
#     val = 1
#     for i in range(results_df.shape[0]):
#         if results_df.iloc[i]['predictions'] < -threshold:
#             val = val / results_df.iloc[i]['y'] * (1 - 0.00075)
#     return val


for fn in allFiles:
    res_df = pd.read_csv(fn, usecols=cols)
    res_df = res_df.iloc[:-7]
    longs = []
    transactions = []
    tuner_df = res_df[:int(res_df.shape[0] * 0.25)]
    for th in THRESHOLDS:
        long_prof = np.mean(tuner_df[tuner_df['predictions'] > th]['y']) - 0.00075
        longs.append(long_prof)
        trans = tuner_df[tuner_df['predictions'] > th].shape[0]
        transactions.append(trans)
    res_df = res_df[int(res_df.shape[0] * 0.25):]
    profits = [l ** t for l, t in zip(longs, transactions)]
    best_ind = np.argmax(profits)
    best_th = THRESHOLDS[best_ind]
    avg_long = np.mean(res_df[res_df['predictions'] > best_th]['y'])
    n_trans = res_df[res_df['predictions'] > best_th].shape[0]
    print('long: {}%, transactions: {}'.format(np.round((avg_long -1)*100, 4), n_trans))

"""

for th in THRESHOLDS:
    longs = []
    shorts = []
    transactions = []
    for fn in allFiles:
        res_df = pd.read_csv(fn, usecols=cols)
        res_df = res_df.iloc[:-7]
        long_prof = np.mean(res_df[res_df['predictions'] > th]['y']) - 0.00075
        # short_prof = np.mean(res_df[res_df['predictions'] < -th]['y']) - 0.00075
        # short_prof = 1
        tr = res_df[res_df['predictions'] > th].shape[0] #+ res_df[res_df['predictions'] < -th].shape[0]
        transactions.append(tr)
        f = fn.replace(path, '')
        # print('File: {}, longs: {}, shorts: {}, transactions: {}'.format(f, round(long_prof, 4),
        #                                                               round(short_prof, 4), tr))
        # print('File: {}, longs: {}, transactions: {}'.format(f, round(long_prof, 4), tr))

        longs.append(long_prof)
        # shorts.append(short_prof)
    # print('#############################################')
    longs = np.array(longs)
    transactions = np.array(transactions)
    # shorts = np.array(shorts)
    longs = longs[~np.isnan(longs)]
    transactions = transactions[~np.isnan(transactions)]
    total = sum([l*t for l,t in zip(longs, transactions)])/sum(transactions)
    total = (total-1)*100
    # shorts = shorts[~np.isnan(shorts)]
    # total = round((np.mean(longs) * np.mean(shorts)-1)*100, 4)
    # print('Threshold: {}, avg long: {}, avg short: {}, #Transactions: {}, Total: {}%'.format(round(th, 4),
    #                                                                                         round(np.mean(longs), 4),
    #                                                                                         round(np.mean(shorts), 4),
    #                                                                                         round(np.mean(transactions),
    #                                                                                               4),
    #                                                                                         total))
    print('Threshold: {}, #Transactions: {}, avg_long: {}%'.format(round(th, 4),
                                                                   round(np.mean(transactions),4),
                                                                   round(total, 4),))
    # print('#############################################')

"""
