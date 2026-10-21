import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneGroupOut

from ..data.preprocessing import ExpectedRateModel
from .pipeline import FreightPipelineBuilder

OUTLIER_COL = 'is_outlier'


class FreightRateModel:
    """Trains the freight pipeline on outlier-free rows and evaluates it one quarter at a time.

    Expects the cleaned training data from DataCleaner (target, `quarter` and `is_outlier` columns).
    In CV, outliers are re-flagged from each fold's train rows and removed from train rows only;
    test rows are never trimmed.
    """
    def __init__(self, builder=None, outlier_low=0.5, outlier_high=2.0,
                 target='posted_rate', group_col='quarter'):
        self.builder = builder or FreightPipelineBuilder()
        self.outlier_low = outlier_low
        self.outlier_high = outlier_high
        self.target = target
        self.group_col = group_col

    def _detector(self, train_df):
        return ExpectedRateModel(self.outlier_low, self.outlier_high).fit(train_df)

    def _split_xy(self, df):
        return df.drop(columns=[self.target, OUTLIER_COL], errors='ignore'), df[self.target]

    def cross_validate(self, df):
        """Leave-one-quarter-out CV [THOUGHT6]."""
        X, y = self._split_xy(df)
        groups = df[self.group_col]
        pipe = self.builder.build()

        results = []
        for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
            held_out = groups.iloc[test_idx].iloc[0]

            # Formula learned from THIS fold's train rows only (df, not X: it needs the target)
            detector = self._detector(df.iloc[train_idx])
            bad_tr, bad_te = detector.flag(df.iloc[train_idx]), detector.flag(df.iloc[test_idx])
            kept_idx = train_idx[~bad_tr]

            # clone(): fresh unfitted pipeline, so all statistics come from this fold's kept train rows
            fold_pipe = clone(pipe).fit(X.iloc[kept_idx], y.iloc[kept_idx])
            pred_tr = fold_pipe.predict(X.iloc[kept_idx])
            pred_te = fold_pipe.predict(X.iloc[test_idx])
            y_te = y.iloc[test_idx].values

            results.append((
                held_out, len(test_idx), bad_tr.mean() * 100,
                mean_absolute_error(y.iloc[kept_idx], pred_tr),
                mean_absolute_error(y_te, pred_te),                      # all test rows (honest number)
                mean_absolute_error(y_te[~bad_te], pred_te[~bad_te]),    # without outlier rows
            ))

        res = pd.DataFrame(results, columns=['quarter', 'rows', 'train_dropped_%',
                                             'train_MAE', 'test_MAE', 'test_MAE_clean'])
        res['gap'] = res['test_MAE'] - res['train_MAE']    # large gap = overfitting or a shifted quarter
        return res

    @staticmethod
    def summarize_cv(res):
        return {
            'mean test MAE': res['test_MAE'].mean(),
            'row-weighted test MAE': np.average(res['test_MAE'], weights=res['rows']),
            'mean test MAE (clean)': res['test_MAE_clean'].mean(),
            'mean train MAE': res['train_MAE'].mean(),
        }

    def fit(self, df):
        """Fit the final pipeline on all of `df` minus its outliers (`is_outlier`, or flagged here if absent)."""
        bad = df[OUTLIER_COL].astype(bool).values if OUTLIER_COL in df else self._detector(df).flag(df)
        X, y = self._split_xy(df[~bad])
        self.feature_columns_ = list(X.columns)
        self.n_dropped_ = int(bad.sum())
        self.pipeline_ = self.builder.build().fit(X, y)
        return self

    def predict(self, X):
        # reindex: columns absent from X (e.g. quote_signal in the December inputs) become NaN;
        # the imputers / XGBoost handle the missing values
        return self.pipeline_.predict(X.reindex(columns=self.feature_columns_))

    def save(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
