__author__ = "Koren Gast"
import numpy as np

class Model(object):
    def __init__(self):
        self.model = None
        self.name = "Generic"

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        pass
        # preds = self.predict(X)
        # return accuracy_score(y, preds), recall_score(y, preds), precision_score(y, preds)

    def predict(self, df_no_y):
        def one_pred(row):
            row = np.array(row, ndmin=2)
            probs = self.model.predict_proba(row)[0]
            weights = [-10 / 16, -5 / 16, -1 / 16, 0, 1 / 16, 5 / 16, 10 / 16]
            weighted_prob = 0.5 + sum([w * p for w, p in zip(weights, probs)])
            return weighted_prob

        predictions =  df_no_y.apply(one_pred, axis=1)
        return predictions