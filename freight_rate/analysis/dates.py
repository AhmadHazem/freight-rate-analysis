import matplotlib.pyplot as plt
import pandas as pd

from .base import Analysis, AnalysisResult


class DateAnalysis(Analysis):
    """Trend and seasonality of posted_rate [INS11]."""
    title = '1.1.4 Date analysis'

    def run(self):
        r = AnalysisResult(self.title)
        df = self.train_test_df.copy()
        df['date'] = pd.to_datetime(df['date'])

        # A trend or seasonal pattern shows up as a moving line
        monthly = df.groupby(df['date'].dt.to_period('M'))['posted_rate'].mean()
        fig, ax = plt.subplots(figsize=(10, 4))
        monthly.plot(ax=ax, marker='o', title='Mean posted_rate per month')
        r.figures['monthly_mean_rate'] = fig

        weekday = df.groupby(df['date'].dt.dayofweek)['posted_rate'].mean().rename_axis('weekday (0=Mon)')
        month = df.groupby(df['date'].dt.month)['posted_rate'].mean().rename_axis('month')

        r.notes += [
            f'Monthly mean posted_rate ranges ${monthly.min():,.0f}-${monthly.max():,.0f} '
            f'({monthly.idxmin()} lowest, {monthly.idxmax()} highest)',
            '[INS11] Date affects posted_rate but is not used as a feature; it defines the CV folds [CONC4]',
        ]
        r.tables['mean posted_rate per month'] = monthly
        r.tables['mean posted_rate per weekday'] = weekday
        r.tables['mean posted_rate per month number'] = month
        return r
