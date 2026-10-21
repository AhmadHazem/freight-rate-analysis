import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


def drop_columns(X, drop_cols):
    return X.drop(columns=drop_cols, errors='ignore')


def abs_weight(X):
    """Negative weights look like input errors [CONC2]: -47500 -> 47500, NaN stays NaN."""
    X = X.copy()
    X['weight'] = X['weight'].abs()
    return X


class GroupImputer(BaseEstimator, TransformerMixin):
    """Fill missing values in `cols`.
    group_by: list of columns to group by (None = no grouping, use the whole column)
    strategy: 'mean', 'median' or 'zero'
    """
    def __init__(self, cols, group_by=None, strategy='median'):
        self.cols = cols
        self.group_by = group_by
        self.strategy = strategy

    def fit(self, X, y=None):
        cols = list(self.cols)

        # Global fallback value per column (used when a group is unseen or has only NaN)
        if self.strategy == 'zero':
            self.global_ = pd.Series(0.0, index=cols)
        else:
            # getattr(X[cols], 'median'): picks the .mean() or .median() method by name
            self.global_ = getattr(X[cols], self.strategy)()

        # Per-group value, learned from train only (skipped for 'zero' or no grouping)
        self.stats_ = None
        if self.group_by and self.strategy != 'zero':
            self.stats_ = getattr(X.groupby(list(self.group_by))[cols], self.strategy)().reset_index()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols:
            filled = X[c]
            if self.stats_ is not None:
                # merge(how='left'): attaches each row's group value, keeps row order, NaN if the group is unseen
                group_vals = X[list(self.group_by)].merge(
                    self.stats_[list(self.group_by) + [c]], on=list(self.group_by), how='left')[c].values
                filled = filled.fillna(pd.Series(group_vals, index=X.index))
            X[c] = filled.fillna(self.global_[c])        # then fall back to the global value
        return X


class ExpectedRateModel:
    """Expected price from distance + equipment [THOUGHT8]:

        log(rate) = start + slope * log(distance) + equipment shift

    Fitted by least squares. The first equipment level (alphabetical, 'Dry Van') is the
    baseline with shift 0. A load is an outlier when actual / expected is below `low`
    or above `high`.
    """
    def __init__(self, low=0.5, high=2.0):
        self.low = low
        self.high = high

    def _design(self, df):
        X = pd.DataFrame({'log_dist': np.log(df['distance']), 'const': 1.0}, index=df.index)
        for e in self.levels_:                       # one 0/1 column per non-baseline equipment
            X['eq_' + e] = (df['equipment'] == e).astype(float)
        return X

    def fit(self, df):
        levels = sorted(df['equipment'].unique())
        self.baseline_, self.levels_ = levels[0], levels[1:]
        X = self._design(df)
        beta, *_ = np.linalg.lstsq(X.values, np.log(df['posted_rate']).values, rcond=None)
        self.coef_ = pd.Series(beta, index=X.columns)
        return self

    @property
    def slope(self):
        return self.coef_['log_dist']

    @property
    def start(self):
        return self.coef_['const']

    @property
    def shifts(self):
        return {self.baseline_: 0.0, **{e: self.coef_['eq_' + e] for e in self.levels_}}

    def expected(self, df):
        return pd.Series(np.exp(self._design(df).values @ self.coef_.values), index=df.index)

    def ratio(self, df):
        return df['posted_rate'] / self.expected(df)

    def flag(self, df):
        """Boolean array, True = outlier."""
        ratio = self.ratio(df)
        return ((ratio < self.low) | (ratio > self.high)).values
