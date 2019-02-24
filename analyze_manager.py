__author__ = "Koren Gast"
import pandas as pd
from matplotlib import pyplot as plt
import glob
import numpy as np
from joblib import Parallel, delayed

y_preds = dict()
path = 'predictions\\5M_30M\\RandomForest_100\\Manager'
# allFiles = glob.glob(path + "/*.csv")
joined_df = pd.read_csv(path + '/Joined.csv')
cols = joined_df.columns
pred_cols = [c for c in cols if '_pred' in c]
y_cols = [c for c in cols if '_y' in c]

money = 1
transactions = 0
for i in range(len(joined_df)):
    row = joined_df.iloc[i]
    preds = row[pred_cols]
    if np.sum(preds > 0.075) > 0:
        symbol_y = preds.idxmax(axis=1).replace('_predictions', '_y')
        money = money*row[symbol_y]*(1-0.00075)
        transactions += 1
    if np.sum(preds < -0.075) > 0:
        symbol_y = preds.idxmin(axis=1).replace('_predictions', '_y')
        money = (money/row[symbol_y])*(1-0.00075)
        transactions += 1
print(money)
print(transactions)

