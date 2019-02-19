__author__ = "Koren Gast"
from models.model import Model
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost(Model):
    def __init__(self, n_estimators=100, class_weight=None):
        super().__init__()
        self.model = AdaBoostClassifier(n_estimators=n_estimators)
        self.name = "Random forest"

    def get_feture_importances(self, s):
        return self.model.feature_importances_