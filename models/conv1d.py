__author__ = "Koren Gast"
from models.model import Model as GenModel
from keras.layers import Dense, Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from utils.utils import to_categorical
import numpy as np


class LSTM_model(GenModel):
    def __init__(self, n_features, n_bins):
        super().__init__()
        self.model = self.build(n_features, n_bins)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.name = "LSTM"
        self.oh_dict = dict()

    def build(self, n_features, n_bins):
        inputs = Input(shape=(n_features,))
        X = Conv1D(100, 10)(inputs)
        X = Conv1D(100, 10)(X)
        X = MaxPooling1D(3)(X)
        X = Conv1D(100, 10, activation='relu')(X)
        X = Conv1D(100, 10, activation='relu')(X)
        X = GlobalAveragePooling1D()(X)
        outputs = Dense(n_bins, activation='softmax')(X)
        model = Model(inputs=inputs, outputs=outputs)
        print(model.summary())
        return model

    def fit(self, X, y, epochs=20, batch_size=30):
        X = self.scaler.fit_transform(X)
        X = X.reshape(X.shape[0], 1, X.shape[1])
        y, oh_dict = to_categorical(y)
        self.oh_dict = oh_dict
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, df_no_y):

        def oh_dict_to_arr(oh_dict):
            arr = np.zeros(oh_dict[0].shape)
            for k in oh_dict.keys():
                arr = arr + k*oh_dict[k]
            return arr

        c = oh_dict_to_arr(self.oh_dict)
        # c = np.array([-5, -2, -1, 0, 1, 2, 5])
        # c = np.array([-1, 0, 1])

        def one_pred(row):
            row = np.array(row, ndmin=2)
            row = self.scaler.transform(row)
            row = row.reshape(row.shape[0], 1, row.shape[1])
            probs = self.model.predict(row)[0]
            weighted_prob = sum([w * p for w, p in zip(weights, probs)])
            return weighted_prob

        tot_w = sum(x for x in c if x > 0)
        weights = c / tot_w
        predictions = df_no_y.apply(one_pred, axis=1)
        # X = self.scaler.transform(X)
        # X = X.reshape(X.shape[0], 1, X.shape[1])
        # raw_preds = self.model.predict(X)
        # preds = np.argmax(raw_preds, axis=1)
        # probs = np.max(raw_preds, axis=1)
        # return preds, probs
        return predictions

    def get_feture_importances(self, s):
        return [None]*s
