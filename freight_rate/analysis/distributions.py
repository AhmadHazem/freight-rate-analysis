import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import Analysis, AnalysisResult

NUMERIC_FEATURES = ['distance', 'weight', 'market_index', 'quote_signal']


class DistributionAnalysis(Analysis):
    """Feature vs target scatters, weight cap, target skew and box plots [INS8-INS10, INS12]."""
    title = '2.1 Distributions and box plots'

    def plot_feature_scatters(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        corrs = {}
        for feature, ax in zip(NUMERIC_FEATURES, axes.flatten()):
            data = self.train_test_df[[feature, 'posted_rate']].dropna()
            # A random 5000 rows keeps the plot fast and readable
            data = data.sample(n=min(5000, len(data)), random_state=self.seed)
            corrs[feature] = data[feature].corr(data['posted_rate'])
            ax.scatter(data[feature], data['posted_rate'], s=8, alpha=0.3)
            ax.set_xlabel(feature)
            ax.set_ylabel('posted_rate')
            ax.set_title(f'{feature} vs posted_rate (corr = {corrs[feature]:.2f})')
        fig.tight_layout()
        return fig, pd.Series(corrs, name='corr with posted_rate')

    @staticmethod
    def weight_cap(df):
        cap = df['weight'].max()
        n_capped = (df['weight'] == cap).sum()
        return cap, n_capped, n_capped / len(df)

    def plot_target(self):
        rate = self.train_test_df['posted_rate']
        fig, axes = plt.subplots(1, 2, figsize=(18, 5))
        axes[0].hist(rate, bins=100)
        axes[0].set(title='posted_rate (dollars)', xlabel='posted_rate', ylabel='rows')
        axes[1].hist(np.log(rate), bins=100)
        axes[1].set(title='log(posted_rate)', xlabel='log(posted_rate)')
        fig.tight_layout()
        return fig, rate.skew(), np.log(rate).skew()

    def plot_boxplots(self):
        features = ['weight', 'market_index', 'distance', 'quote_signal', 'posted_rate']
        plot_df = self.train_test_df.copy()
        plot_df['weight'] = plot_df['weight'].abs()    # -47500 -> 47500 for the plot only

        fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 6), squeeze=False)
        for feature, ax in zip(features, axes[0]):
            ax.boxplot(plot_df[feature].dropna(), showmeans=True)
            ax.set_title(feature)
            ax.set_ylabel(feature)
            ax.set_xticks([])
        fig.tight_layout()
        return fig

    def run(self):
        r = AnalysisResult(self.title)

        r.figures['features_vs_rate'], corrs = self.plot_feature_scatters()
        r.figures['posted_rate_hist'], skew_raw, skew_log = self.plot_target()
        r.figures['boxplots'] = self.plot_boxplots()

        n_negative = (self.train_test_df['weight'] < 0).sum()
        r.notes.append(f"[INS10] distance is the strongest feature (corr {corrs['distance']:.2f})")
        for name, df in [('train-test', self.train_test_df), ('validation', self.validation_df)]:
            cap, n_capped, share = self.weight_cap(df)
            r.notes.append(f'[INS8] {name}: weight capped at {cap:,.0f} on {n_capped} rows ({share:.1%})')
        r.notes += [
            f'[INS9] {n_negative} negative weights in train-test -> abs() [CONC2]',
            f'[INS12] posted_rate skew: raw {skew_raw:.2f}, log {skew_log:.2f} -> model log(rate) [THOUGHT7]',
        ]
        r.tables['correlation with posted_rate'] = corrs
        r.tables['train-test top weight values'] = self.train_test_df['weight'].value_counts().head()
        r.tables['validation top weight values'] = self.validation_df['weight'].value_counts().head()
        return r
