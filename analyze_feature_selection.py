__author__ = "Koren Gast"
import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

allFiles = glob.glob("predictions\\5M_30M\\feature_selection\\*.csv")


def comul(y):
    c = [1]
    for i in y:
        c.append(c[-1] * i)
    return c[1:]


best_measure = -5
best_feature = ''

for f in allFiles:
    df = pd.read_csv(f)
    df = df.dropna()
    df = df.sort_values('predictions', ascending=False)
    df['comulative'] = comul(df['y'])
    df_to_zero = df[df['predictions'] > 0]
    print('{}: {}'.format(f.replace('predictions\\5M_30M\\feature_selection\\', ''), np.mean(df_to_zero['comulative'])))
    if np.mean(df_to_zero['comulative']) > best_measure:
        best_measure = np.mean(df_to_zero['comulative'])
        best_feature = f.replace('predictions\\5M_30M\\feature_selection\\', '')
    plt.plot(df['predictions'], df['comulative'])
    plt.title(f.replace('predictions\\5M_30M\\feature_selection\\', ''))
    # plt.show()
# plt.show()
print(best_measure)
print(best_feature)
