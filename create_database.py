__author__ = "Koren Gast"
from utils.utils import get_klines, merge_klines, client_intervals
from utils.build_features import build_features
import pandas as pd
import pathlib

s_date = '01 Jan, 2019'
e_date = '02 Jan, 2019'
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
pull_interval = '5M'
data_interval = '30M'
merging = 6  # Should be equal to data_interval/pull_interval
data_intervals = pull_interval + '_' + data_interval

kl_file_names = []
for symbol in symbols:
    kl_f_name = symbol + '_' + s_date + '_TO_' + e_date + '.csv'
    kl_file_names.append(kl_f_name)
    small_kl = get_klines(s_date, e_date, symbol, client_intervals[pull_interval])
    klines = merge_klines(
        small_kl,
        merge_amount=merging)
    pathlib.Path('klines/' + data_intervals).mkdir(exist_ok=True)
    klines.to_csv('klines/' + data_intervals + '/' + kl_f_name, index=False)

features_file_names = []

for kl_f in kl_file_names:
    features_f_name = kl_f
    klines = pd.read_csv('klines/' + data_intervals + '/' + kl_f)
    features_file_names.append(features_f_name)
    features = build_features(klines)
    pathlib.Path('features/' + data_intervals).mkdir(exist_ok=True)
    features.to_csv('features/' + data_intervals + '/' + features_f_name, index=False)

print('Database created')
