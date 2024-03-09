from numpy import ndarray
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class RangeScaler(BaseEstimator, TransformerMixin):
    def __init__(self, var_to_range: dict[str, tuple[float, float]]):
        super(RangeScaler).__init__()
        self.var_to_range = var_to_range
        # desired new min and max values
        self.min = 0
        self.max = 1

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        scaled_X = X.copy()
        # for var, range in self.var_to_range.items():
        for var in X.columns:
            original_min, original_max = self.var_to_range[var]
            scaled_X[var] = (
                (X[var] - original_min) / (original_max - original_min)
            ) * (self.max - self.min) + self.min
        return scaled_X.values

    def fit_transform(self, X: pd.DataFrame, y=None) -> ndarray:
        return self.transform(X)
