import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

cabin_index = 3


class AttributeAdder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self  # Nothing to do in this case

    def transform(self, X):
        # https://stackoverflow.com/a/48320451/604365
        # print(X)
        # print(type(X))
        # ship_section = X[:, cabin_index].astype("<U1")  # Not sure this works
        X["ship_section"] = X["Cabin"].fillna("U").str[0]
        X.drop(axis=1, columns="Cabin", inplace=True)
        return X
