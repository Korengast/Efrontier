__author__ = "Koren Gast"
import pandas as pd

results_file_name = 'predictions/BTCUSDT.csv'
res_df = pd.read_csv(results_file_name)[['timestamp', 'BTCUSDT_close', 'y_bins', 'predictions']]

pos_df = res_df[res_df['y_bins'] > 0]
