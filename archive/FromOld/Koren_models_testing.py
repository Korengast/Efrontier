from binance.client import Client
import datetime
import json
import pandas as pd
from copy import deepcopy


my_data = pd.read_csv("my_try_15_01_jan_2017_data.csv")
cols = ['open']
my_data = my_data[cols]
copy_data = deepcopy(my_data)
accumulate_per_row = 100

for i in range(1,accumulate_per_row+1):  # The loop starts with i=1 and not i=0 which is the current period
    my_data['pos'] = my_data.index.tolist()
    copy_data.columns = [str(i) + '_' + col for col in copy_data.columns]
    copy_data['pos'] = copy_data.index.tolist()
    copy_data['pos'] -= i
    my_data = pd.merge(my_data, copy_data, how='inner', on=['pos'])
    del copy_data['pos']
    copy_data.columns = cols

del my_data['pos']
my_data = my_data[:-1]
my_train = my_data[0: int(my_data.shape[0] * 0.8)]
my_test = my_data[int(my_data.shape[0] * 0.8):]
from sklearn import linear_model

regr = linear_model.LinearRegression()
# regr = linear_model.LogisticRegression()


column_to_predict = my_train['open']
del my_train['open']


regr.fit(my_train, column_to_predict)

fee_rate=0.001

def get_simulated_profit (real, predicted):
    profit = 1 # start with 1$
    coin = 'USDT'
    buying_price = 0 # This will represent the USDT value I paid for the last BTC purchase
    for i in range(len(real) - 1):
        if coin=='USDT':
            if predicted[i + 1]>real[i]*(1 - 2 * fee_rate): # If I hold USDT, buy BTC only if...
                coin = 'BTC'
                buying_price = real[i]
                profit = profit*(1-fee_rate)
        else:
            if predicted[i + 1] < real[i] * (1 - 2 * fee_rate) and real[i]>buying_price* (1 - 2 * fee_rate):
                coin = 'USDT'
                profit = profit * (1 - fee_rate)
        if coin=='BTC': # For every period I hold BTC, my theoretic USD value grow\cease along with the BTC/USDT value
            profit = profit * real[i + 1] / real[i]
    return profit

real_test_open_values = my_test['open']
del my_test['open']


import random
# random prediction is the real value multiple by randomly choose between 0.8 to 1.2
rand_pred = []
for i in range( len(real_test_open_values)):
    rand_pred.append(real_test_open_values.iloc[i]*random.uniform(0.8,1.2))

pred_prof = get_simulated_profit(list(real_test_open_values), list(regr.predict(my_test))) # The simulated profit using the model
rand_prof = get_simulated_profit(list(real_test_open_values), rand_pred) # The simulated profit using the random predicted values
BTC_profit = real_test_open_values.iloc[len(real_test_open_values)-1]/real_test_open_values.iloc[0] # The profit for buying BTC at period 0 and do nothing

print 'Using the model, profit is: ' + str(round(pred_prof*100,2))+'%'
print 'Using random prediction, profit is: ' + str(round(rand_prof*100,2))+'%'
print 'Stay on BTC the entire period, profit is: ' + str(round(BTC_profit*100,2))+'%'