__author__ = "Koren Gast"
from utils.utils import get_klines, merge_klines, client_intervals
# from utils.build_features import build_features
from utils.build_qualitative_features import build_features
import pandas as pd
import pathlib
from joblib import Parallel, delayed

s_date = '01 Jan, 2018'
e_date = '31 Jan, 2018'
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
pull_interval = '5M'
data_interval = '30M'
merging = 6  # Should be equal to data_interval/pull_interval
data_intervals = pull_interval + '_' + data_interval

kl_file_names = []



#for symbol in symbols:
def load_symbol(symbol):
    print(symbol)
    kl_f_name = symbol + '_' + s_date + '_TO_' + e_date + '.csv'
    kl_file_names.append(kl_f_name)
    small_kl = get_klines(s_date, e_date, symbol, client_intervals[pull_interval])
    print(symbol + " klines")
    klines = merge_klines(
        small_kl,
        merge_amount=merging)
    print(symbol + " klines merged")
    pathlib.Path('klines/' + data_intervals).mkdir(exist_ok=True)
    klines.to_csv('klines/' + data_intervals + '/' + kl_f_name, index=False)
    return kl_f_name

#for kl_f in kl_file_names:
def build_feature(kl_f, merging):
    print(kl_f)
    features_f_name = kl_f
    klines = pd.read_csv('klines/' + data_intervals + '/' + kl_f)
    features = build_features(klines, merging)
    pathlib.Path('features/' + data_intervals).mkdir(exist_ok=True)
    features.to_csv('features/' + data_intervals + '/' + features_f_name, index=False)




results = Parallel(n_jobs=10)(delayed(load_symbol)(e) for e in symbols)
print(results)
Parallel(n_jobs=10)(delayed(build_feature)(e, merging) for e in results)
print('Database created')
