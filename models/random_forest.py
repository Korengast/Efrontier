__author__ = "Koren Gast"
from models.model import Model
from sklearn.ensemble import RandomForestClassifier


class RandomForest(Model):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators)
        self.name = "Random forest"

    def get_feture_importances(self):
        return self.model.feature_importances_
