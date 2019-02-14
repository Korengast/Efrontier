__author__ = "Koren Gast"
from utils.utils import get_klines, merge_klines, client_intervals
from utils.build_features import build_features
import pandas as pd

s_date = '01 Jan, 2019'
e_date = '31 Jan, 2019'
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
pull_interval = '1M'
data_interval = '30M'
merging = 30  # Should be equal to data_interval/pull_interval

kl_file_names = []
for symbol in symbols:
    kl_f_name = symbol + '_' + s_date + '_TO_' + e_date + '.csv'
    kl_file_names.append(kl_f_name)
    small_kl = get_klines(s_date, e_date, symbol, client_intervals[pull_interval])
    klines = merge_klines(
        small_kl,
        merge_amount=merging)
    klines.to_csv('klines/30M/' + kl_f_name, index=False)

features_file_names = []
for kl_f in kl_file_names:
    features_f_name = kl_f
    klines = pd.read_csv('klines/' + data_interval + '/' + kl_f)
    features_file_names.append(features_f_name)
    features = build_features(klines)
    features.to_csv('features/' + data_interval + '/' + features_f_name, index=False)

print('Database created')
