__author__ = "Koren Gast"
from models.model import Model
from sklearn.neural_network import MLPClassifier


class MLP(Model):
    def __init__(self, layers_sizes=[8, 4, 2]):
        super().__init__()
        self.model = MLPClassifier(hidden_layer_sizes=layers_sizes, max_iter=100000)
        self.name = "MLP"

    def get_feture_importances(self, s):
        return [None]*s