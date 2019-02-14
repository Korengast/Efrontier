__author__ = "Koren Gast"

from utils.utils import merge_assets, prepare_data

from models.random_forest import RandomForest

CUTOFF = 0.15  # in percents. The minimal value of ascending
N_ESTIMATORS = 100
s_date = '01 Jan, 2019'
e_date = '31 Jan, 2019'
symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
data_interval = '30M'
symbols_to_predict = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'LTCUSDT', 'NEOUSDT']
merging = 30  # Should be equal to data_interval/pull_interval

kl_file_names = []
for symbol in symbols:
    kl_f_name = symbol + '_' + s_date + '_TO_' + e_date + '.csv'
    kl_file_names.append(kl_f_name)
features_file_names = []
for kl_f in kl_file_names:
    features_f_name = kl_f
    features_file_names.append(features_f_name)

features = merge_assets(symbols, features_file_names, data_interval)

for s2pred in symbols_to_predict:
    X_train, X_valid, y_train, y_valid, df_valid, df_valid_y = prepare_data(features, CUTOFF, s2pred, merging)
    model = RandomForest(n_estimators=N_ESTIMATORS)
    model.fit(X=X_train, y=y_train)
    df_valid_y['predictions'] = model.predict(df_valid.drop('timestamp', axis=1))
    pred_file_name = s2pred + '.csv'
    df_valid_y.to_csv('predictions/'+pred_file_name, index=False)

print('Done')