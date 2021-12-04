import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

cabin_index = 10


class AttributeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing to do in this case

    def transform(self, X):
        # https://stackoverflow.com/a/48320451/604365
        ship_section = X[:, cabin_index].astype("<U1")  # Not sure this works

        return np.c_[X, ship_section]
