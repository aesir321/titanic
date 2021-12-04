from sklearn.base import TransformerMixin


class ManualFeatureSelector(TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.features]
