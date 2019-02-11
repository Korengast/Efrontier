from binance.client import Client
from api.apis import api_private, api_public

__author__ = "Avi Levi"
import datetime
import json
import pandas as pd


# TO get a start date, end date, symbol, interval - The class will orgenize the data as table.



class PullHistoricalData(object):
    def __init__(self, start_time, end_time, symbol, interval):
        self.start_time = start_time
        self.end_date = end_time
        self.symbol = symbol
        self.interval = interval

        self.client = Client(api_public,
                             api_private)

        # self.symbols = self.client.get_all_tickers()

    def pull_data(self):
        klines = self.client.get_historical_klines(self.symbol, Client.KLINE_INTERVAL_3MINUTE, '01 Jan, 2017')

        filtered_klines = []
        for k in klines:
            new_list = k[1:6] + [k[-4]]
            new_list = [float(n) for n in new_list]
            filtered_klines.append(new_list)

        new_df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'Number of trades'],
                              data=filtered_klines)

        new_df.to_csv("my_try_15_01_jan_2017_data.csv", index=False)


start_time = datetime.datetime.utcnow() - datetime.timedelta(days=100)
end_time = datetime.datetime.utcnow()
a = PullHistoricalData(start_time.strftime("%m %d %Y"), end_time.strftime("%m %d %Y"), 'BTCUSDT',
                       Client.KLINE_INTERVAL_1HOUR)


# TO IT ONCE!!
a.pull_data()


def try_simple():
    my_data = pd.read_csv("my_try_15_01_jan_2017_data.csv")
    my_data['diff'] = my_data['close'] / my_data['open']
    my_data['pred_diff'] = my_data['diff']
    my_data['high'] = my_data['open'] / my_data['high']
    my_data['low'] = my_data['open'] / my_data['low']

    cols = ['high', 'low', 'volume', 'Number of trades', 'diff', 'pred_diff']
    my_data = my_data[cols]

    my_data['pred_diff'] = my_data['pred_diff'].apply(lambda x: 1 if x < 0.98 else 0)
    my_data['pred_diff'] = my_data['pred_diff'].astype(int)

    my_data['pred_diff'] = my_data['pred_diff'].shift(-1)
    my_data = my_data[:-1]
    my_train = my_data[0: int(my_data.shape[0] * 0.8)]
    my_test = my_data[int(my_data.shape[0] * 0.8):]

    from sklearn import linear_model

    # regr = linear_model.LinearRegression()
    regr = linear_model.LogisticRegression()


    column_to_predict = my_train['pred_diff']
    del my_train['pred_diff']

    regr.fit(my_train, column_to_predict)

    real_res = my_test['pred_diff']
    del my_test['pred_diff']

    model_res = regr.predict(my_test)

    from sklearn.metrics import roc_auc_score

    roc_auc_score(real_res, model_res)


def try_more_complex():

    my_data = pd.read_csv("my_try_15_01_jan_2017_data.csv")
    my_data['diff'] = my_data['close'] / my_data['open']
    my_data['high'] = my_data['open'] / my_data['high']
    my_data['low'] = my_data['open'] / my_data['low']

    cols = ['high', 'low', 'volume', 'Number of trades', 'diff']
    my_data = my_data[cols]

    from copy import deepcopy

    copy_data = deepcopy(my_data)
    accumulate_per_row = 5

    for i in range(accumulate_per_row):
        my_data['pos'] = my_data.index.tolist()

        copy_data.columns = [str(i+1) + '_' + col for col in copy_data.columns]
        copy_data['pos'] = copy_data.index.tolist()
        copy_data['pos'] -= i

        my_data = pd.merge(my_data,copy_data, how='inner', on=['pos'])

        del copy_data['pos']
        copy_data.columns = cols

    del my_data['pos']

    my_data['pred_diff'] = my_data[str(accumulate_per_row) + '_diff']

    my_data['pred_diff'] = my_data['pred_diff'].apply(lambda x: 1 if x < 0.98 else 0)
    my_data['pred_diff'] = my_data['pred_diff'].astype(int)

    my_data['pred_diff'] = my_data['pred_diff'].shift(-1)
    my_data = my_data[:-1]
    my_train = my_data[0: int(my_data.shape[0] * 0.8)]
    my_test = my_data[int(my_data.shape[0] * 0.8):]

    from sklearn import linear_model

    # regr = linear_model.LinearRegression()
    regr = linear_model.LogisticRegression()


    column_to_predict = my_train['pred_diff']
    del my_train['pred_diff']

    regr.fit(my_train, column_to_predict)

    real_res = my_test['pred_diff']
    del my_test['pred_diff']

    model_res = regr.predict(my_test)

    from sklearn.metrics import roc_auc_score

    roc_auc_score(real_res, model_res)


