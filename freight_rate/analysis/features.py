import pandas as pd

from .base import Analysis, AnalysisResult


class FeatureOverview(Analysis):
    """General statistics, dataset sizes, missing values [INS7]."""
    title = '1.1 Feature overview'

    def run(self):
        tt, val = self.train_test_df, self.validation_df
        r = AnalysisResult(self.title)

        feature_cols = tt.columns[1:-1]      # everything between load_id and posted_rate
        r.notes.append(f'Train-test size: {tt.shape}, validation size: {val.shape}')
        r.notes.append('Feature columns: ' + ', '.join(feature_cols))

        missing = pd.DataFrame({'train-test': tt.isnull().sum(), 'validation': val.isnull().sum()})
        missing = missing[missing.fillna(0).sum(axis=1) > 0]
        r.notes.append('Missing values: ' + ', '.join(
            f"{c} ({int(row['train-test'])} train-test / {int(row['validation'])} validation)"
            for c, row in missing.iterrows()))

        r.tables['train-test describe'] = tt.describe()
        r.tables['validation describe'] = val.describe()
        r.tables['train-test head'] = tt.head()
        r.tables['missing values'] = missing
        return r
