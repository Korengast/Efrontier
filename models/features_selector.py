__author__ = "Koren Gast"
from utils.utils import prepare_data
import numpy as np
from joblib import Parallel, delayed

class Selector(object):
    def __init__(self, model, features_df, CUTOFF, s2pred, merging):
        self.s2pred = s2pred
        self.model = model

        self.features_df = features_df
        self.base_cols = ['timestamp', self.s2pred + '_close_ratio', self.s2pred + '_R^2']
        self.added_cols = []

        self.cutoff = CUTOFF
        self.merging = merging

        self.temp_results = dict()

        self.best_measure = -np.inf

    def features_testing(self, c):
        total_cols = self.base_cols + self.added_cols
        if c not in total_cols:
            cols = total_cols + [c]
            x_df = self.features_df[cols]
            x_df = x_df.dropna()
            X_train, X_valid, y_train, y_valid, df_train_y, df_valid, df_valid_y = \
                prepare_data(x_df, self.cutoff, self.s2pred, self.merging, is_features=True)

            self.model.fit(X=X_train, y=y_train)
            df_valid_y['predictions'] = self.model.predict(df_valid.drop('timestamp', axis=1))
            self.temp_results[c] = df_valid_y
            # df_valid_y.to_csv('predictions/5M_30M/feature_selection/' + c + '.csv', index=False)

    def features_adding(self, c):
        df = self.temp_results[c]
        df = df.dropna()
        df = df.sort_values('predictions', ascending=False)

        def comul(y):
            c = [1]
            for i in y:
                c.append(c[-1] * i)
            return c[1:]

        df['comulative'] = comul(df['y'])
        df_to_zero = df[df['predictions'] > 0]
        print('{}: {}'.format(c, np.mean(df_to_zero['comulative'])))
        d = {'feature': c, 'measure':  np.mean(df_to_zero['comulative'])}
        return d
        # if np.mean(df_to_zero['comulative']) > self.best_measure:
        #     self.best_measure = np.mean(df_to_zero['comulative'])
        #     best_feature = c

    def execute(self):
        self.temp_results = dict()
        Parallel(n_jobs=3)(delayed(self.features_testing)(e) for e in self.features_df.columns)
        features_scores = Parallel(n_jobs=3)(delayed(self.features_adding)(e) for e in self.temp_results.keys())
        temp_best_measure = -np.inf
        temp_feature_to_add = ''
        for fs in features_scores:
            if fs['measure'] > temp_best_measure:
                temp_best_measure = fs['measure']
