__author__ = "Koren Gast"
from utils.utils import prepare_data
import numpy as np
from joblib import Parallel, delayed
from models.random_forest import RandomForest
from models.adaBoost import AdaBoost
from models.MLP import MLP


class Selector(object):
    def __init__(self, model, features_df, CUTOFF, s2pred, merging, n_est, class_weights):
        self.s2pred = s2pred
        self.n_est = n_est
        self.class_weights = class_weights

        self.features_df = features_df
        self.base_cols = ['timestamp', self.s2pred + '_close_ratio', self.s2pred + '_R^2']
        self.added_cols = []

        self.cutoff = CUTOFF
        self.merging = merging

        self.temp_results = []

        self.best_measure = -np.inf

    def features_testing(self, c):
        total_cols = self.base_cols + self.added_cols
        if c not in total_cols:
            cols = total_cols + [c]
            x_df = self.features_df[cols]
            x_df = x_df.dropna()
            X_train, X_valid, y_train, y_valid, df_train, df_train_y, df_valid, df_valid_y = \
                prepare_data(x_df, self.cutoff, self.s2pred, self.merging, is_features=True)
            # model = AdaBoost(self.n_est, self.class_weights)
            model = RandomForest(self.n_est, self.class_weights)
            # model = MLP()
            model.fit(X=X_train, y=y_train)
            df_valid_y['predictions'] = model.predict(df_valid.drop('timestamp', axis=1))
            return (df_valid_y, c)
            # print(df_valid_y.shape)
            # self.temp_results[c] = df_valid_y
            # print(self.temp_results.keys())
            # df_valid_y.to_csv('predictions/5M_30M/feature_selection/' + c + '.csv', index=False)

    def features_adding(self, t):
        df = t[0]
        c = t[1]
        print(df.shape)
        df = df.dropna()
        df = df.sort_values('predictions', ascending=False)

        # def comul(y):
        #     c = [1]
        #     for i in y:
        #         c.append(c[-1] * i)
        #     return c[1:]
        #
        # df['comulative'] = comul(df['y'])
        # df_to_zero = df[df['predictions'] > 0.05]
        df_to_zero = df[df['predictions'] > 0.0]
        print('{}: {}'.format(c, np.mean(df_to_zero['y'])))
        d = {'feature': c, 'measure': np.mean(df_to_zero['y'])}
        return d
        # if np.mean(df_to_zero['comulative']) > self.best_measure:
        #     self.best_measure = np.mean(df_to_zero['comulative'])
        #     best_feature = c

    def execute(self, given_list=None, select=True):
        if select:
            keep_picking = True
            while (keep_picking):
                if given_list is None:
                    cols_to_choose = self.features_df.columns
                else:
                    cols_to_choose = given_list
                self.temp_results = Parallel(n_jobs=3)(
                    delayed(self.features_testing)(e) for e in cols_to_choose)
                self.temp_results = [tr for tr in self.temp_results if tr is not None]
                features_scores = Parallel(n_jobs=3)(delayed(self.features_adding)(e) for e in self.temp_results)
                temp_best_measure = -np.inf
                temp_feature_to_add = ''
                for fs in features_scores:
                    if fs['measure'] > temp_best_measure:
                        temp_best_measure = fs['measure']
                        temp_feature_to_add = fs['feature']
                print('Temp best measure: {}'.format(temp_best_measure))
                if temp_best_measure >= self.best_measure:
                    self.best_measure = temp_best_measure
                    self.added_cols = self.added_cols + [temp_feature_to_add]
                    print('{} added'.format(temp_feature_to_add))
                else:
                    keep_picking = False
        else:
            self.added_cols = given_list

    def get_cols(self):
        return self.base_cols + self.added_cols
