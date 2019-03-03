__author__ = "Koren Gast"
from utils.utils import prepare_data
import numpy as np
from joblib import Parallel, delayed
from models.random_forest import RandomForest
from models.adaBoost import AdaBoost
from models.MLP import MLP
import pandas as pd
import copy
t = 0

class Selector(object):
    def __init__(self, model_name, features_df, CUTOFF, s2pred, merging, n_est, class_weights):
        self.s2pred = s2pred
        self.n_est = n_est
        self.class_weights = class_weights
        self.model_name = model_name

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
            x_df = copy.deepcopy(self.features_df)
            x_df = x_df[cols].dropna()
            print(c)
            l = x_df.shape[0]
            X_train = np.array(x_df.iloc[:int(l*0.75)].drop(['timestamp'], axis=1))
            y_train = np.array(self.features_df['y_bins'].iloc[:int(l*0.75)])

            # X_train, X_valid, y_train, y_valid, df_train, df_train_y, df_valid, df_valid_y = \
            #     prepare_data(x_df, self.cutoff, self.s2pred, self.merging, is_features=True)
            model = None
            if 'AdaBoost' in self.model_name:
                model = AdaBoost(self.n_est, self.class_weights)
            if 'RandomForest' in self.model_name:
                model = RandomForest(self.n_est, self.class_weights)
            # model = MLP()
            model.fit(X=X_train, y=y_train)
            self.features_df['predictions'] = model.predict(x_df.drop('timestamp', axis=1))
            return self.features_df, c

    def features_adding(self, t):
        df = t[0]
        c = t[1]
        print(df.shape)
        df = df.dropna()
        df = df.sort_values('predictions', ascending=False)
        df_to_zero = df[df['predictions'] > 0.0]
        print('{}: {}'.format(c, np.mean(df_to_zero['y'])))
        d = {'feature': c, 'measure': np.mean(df_to_zero['y'])}
        return d

    def best_k(self, k):
        y_cols = ['y', 'y_R^2', 'y%', 'y*r2', 'y_bins']
        model = None
        df = self.features_df.drop(['timestamp'] + y_cols + self.base_cols, axis=1)
        X_train = np.array(df)
        y_train = np.array(self.features_df['y_bins'])
        if 'AdaBoost' in self.model_name:
            model = AdaBoost(self.n_est, self.class_weights)
        if 'RandomForest' in self.model_name:
            model = RandomForest(self.n_est, self.class_weights)
        model.fit(X=X_train, y=y_train)
        f = df.columns
        imp = model.get_feture_importances()
        cols_to_choose = pd.DataFrame({'features': f, 'importance': imp}).sort_values('importance', ascending=False)[
                             'features'].iloc[:k]
        return cols_to_choose

    def execute(self, given_list=None, select=True):
        if select:
            keep_picking = True
            while (keep_picking):
                if given_list is None:
                    # cols_to_choose = list(self.features_df.columns)
                    # cols_to_choose.remove('y')
                    # cols_to_choose.remove('y_R^2')
                    # cols_to_choose.remove('y%')
                    # cols_to_choose.remove('y*r2')
                    # cols_to_choose.remove('y_bins')
                    cols_to_choose = self.best_k(k=10)

                else:
                    cols_to_choose = given_list
                # TODO: timing and reduce time
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
