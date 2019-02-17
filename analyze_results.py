__author__ = "Koren Gast"
import pandas as pd

file_path = 'predictions/5M_30M/'
file_name = 'RandomForest_10_NEOUSDT.csv'
res_df = pd.read_csv(file_path + file_name)


pos_df = res_df[res_df['y_bins'] > 0]
