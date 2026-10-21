import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data.preprocessing import ExpectedRateModel
from .base import Analysis, AnalysisResult


class OutlierAnalysis(Analysis):
    """Expected-price formula log(price) ~ log(distance) + equipment, and the loads it flags [INS13, THOUGHT8].

    Analysis only: no rows are removed here. In training, outliers are dropped from the train rows only.
    """
    title = '2.2 Outlier analysis (expected-price formula)'

    def __init__(self, train_test_df, validation_df=None, seed=None, low=0.5, high=2.0):
        super().__init__(train_test_df, validation_df, seed)
        self.low, self.high = low, high

    def prepare(self):
        df = self.train_test_df.copy()
        df['quarter'] = pd.to_datetime(df['date']).dt.to_period('Q').astype(str)
        df['ln_distance'] = np.log(df['distance'])
        df['ln_price'] = np.log(df['posted_rate'])

        self.model = ExpectedRateModel(self.low, self.high).fit(df)
        df['expected'] = self.model.expected(df)
        df['ratio'] = df['posted_rate'] / df['expected']
        df['too_cheap'] = df['ratio'] < self.low
        df['too_expensive'] = df['ratio'] > self.high
        df['outlier'] = df['too_cheap'] | df['too_expensive']
        return df

    def plot_log_log(self, sample):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(sample['distance'], sample['posted_rate'], s=4, alpha=0.3)
        axes[0].set(title='Before: distance vs price (curved band)', xlabel='distance', ylabel='posted_rate')
        axes[1].scatter(sample['ln_distance'], sample['ln_price'], s=4, alpha=0.3)
        axes[1].set(title='After: ln(distance) vs ln(price) (straight band)', xlabel='ln(distance)', ylabel='ln(price)')
        fig.tight_layout()
        return fig

    def plot_fit_lines(self, df, sample):
        # A line through just two Dry Van loads vs the least-squares line through all loads
        van = df[df['equipment'] == 'Dry Van']
        a, b = van.iloc[0], van.iloc[1]
        two_slope = (b['ln_price'] - a['ln_price']) / (b['ln_distance'] - a['ln_distance'])
        two_start = a['ln_price'] - two_slope * a['ln_distance']
        slope, start = self.model.slope, self.model.start

        x = np.linspace(df['ln_distance'].min(), df['ln_distance'].max(), 50)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(sample['ln_distance'], sample['ln_price'], s=3, alpha=0.2, color='gray')
        axes[0].plot(x, two_start + two_slope * x, color='red', linewidth=2, label=f'two loads: slope {two_slope:.3f}')
        axes[0].plot(x, start + slope * x, color='blue', linewidth=2, label=f'all loads: slope {slope:.3f}')
        axes[0].scatter([a['ln_distance'], b['ln_distance']], [a['ln_price'], b['ln_price']], color='red', s=60, zorder=5)
        axes[0].set(title='Line from two loads vs all loads (Dry Van)', xlabel='ln(distance)', ylabel='ln(price)')
        axes[0].legend()
        axes[1].scatter(sample['ln_distance'], sample['ln_price'], s=3, alpha=0.2, color='gray')
        for e, s in self.model.shifts.items():
            axes[1].plot(x, start + slope * x + s, linewidth=2, label=f'{e} (+{s:.3f})')
        axes[1].set(title='Same slope, equipment shifts the line', xlabel='ln(distance)', ylabel='ln(price)')
        axes[1].legend()
        fig.tight_layout()
        return fig, (a, b, two_slope, two_start)

    def equipment_check(self, df):
        """Equipment effect measured directly on the data (800-1200 mi, no outliers) vs the formula."""
        median = df[~df['outlier'] & df['distance'].between(800, 1200)].groupby('equipment')['posted_rate'].median()
        check = pd.DataFrame({
            'median price 800-1200 mi': median,
            'data vs Dry Van %': ((median / median['Dry Van'] - 1) * 100).round(1),
            'formula vs Dry Van %': pd.Series({e: (np.exp(s) - 1) * 100 for e, s in self.model.shifts.items()}).round(1),
        })
        fig, ax = plt.subplots(figsize=(7, 4))
        check[['data vs Dry Van %', 'formula vs Dry Van %']].plot.bar(ax=ax, rot=0)
        ax.set(title='Equipment effect: data vs formula', ylabel='% more expensive than Dry Van')
        return check, fig

    def expected_table(self):
        slope, start = self.model.slope, self.model.start
        return pd.DataFrame(
            {e: [np.exp(start + slope * np.log(m) + s) for m in (300, 1000, 2000)] for e, s in self.model.shifts.items()},
            index=['300 mi', '1,000 mi', '2,000 mi']).round(0)

    def plot_flags(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].hist(np.log10(df['ratio']), bins=np.linspace(-0.8, 0.8, 81))
        axes[0].set_yscale('log')
        axes[0].axvline(np.log10(self.low), color='red', linestyle='--')
        axes[0].axvline(np.log10(self.high), color='red', linestyle='--')
        axes[0].set(title=f'Ratio = actual / expected (log10), cuts at {self.low} and {self.high}',
                    xlabel='log10(ratio)', ylabel='rows (log)')
        ok = df[~df['outlier']].sample(4000, random_state=self.seed)
        bad = df[df['outlier']]
        axes[1].scatter(ok['distance'], ok['posted_rate'], s=3, alpha=0.3, label='normal')
        axes[1].scatter(bad['distance'], bad['posted_rate'], s=10, color='tab:orange', label='outlier')
        axes[1].set(title='Flagged loads', xlabel='distance', ylabel='posted_rate')
        axes[1].legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_flag_rates(by_quarter, by_equipment):
        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        axes[0].bar(by_quarter.index, by_quarter.values)
        axes[0].set(title='% flagged per quarter', ylabel='%')
        axes[1].bar(by_equipment.index, by_equipment.values)
        axes[1].set(title='% flagged per equipment', ylabel='%')
        fig.tight_layout()
        return fig

    def run(self):
        r = AnalysisResult(self.title)
        df = self.prepare()
        sample = df.sample(5000, random_state=self.seed)

        r.figures['log_log_transform'] = self.plot_log_log(sample)
        r.figures['fit_lines'], (a, b, two_slope, two_start) = self.plot_fit_lines(df, sample)
        check, r.figures['equipment_check'] = self.equipment_check(df)
        r.figures['ratio_and_flagged_loads'] = self.plot_flags(df)
        by_quarter = df.groupby('quarter')['outlier'].mean() * 100
        by_equipment = df.groupby('equipment')['outlier'].mean() * 100
        r.figures['flag_rates'] = self.plot_flag_rates(by_quarter, by_equipment)

        table = self.expected_table()
        reefer = table.loc['1,000 mi', 'Reefer']
        slope, shifts = self.model.slope, self.model.shifts

        r.notes += [
            'ln(price) = start + slope * ln(distance) + equipment shift',
            f"Two loads ({a['distance']:.1f} mi ${a['posted_rate']:.2f}, {b['distance']:.1f} mi ${b['posted_rate']:.2f}): "
            f'slope {two_slope:.3f}, start {two_start:.3f}',
            f'Slope {slope:.3f}: doubling distance multiplies price by {2 ** slope:.2f}. '
            f'Start {self.model.start:.3f} only sets the line height.',
            f"Equipment: Flatbed +{shifts['Flatbed']:.3f}, Reefer +{shifts['Reefer']:.3f}, Dry Van is the baseline (0).",
            f'Example: 1,000-mile Reefer posted at $1,100 -> ratio 1,100 / {reefer:,.0f} = {1100 / reefer:.2f}'
            + (f', below {self.low}, outlier' if 1100 / reefer < self.low else ''),
            f"Outliers: {df['outlier'].sum():,} of {len(df):,} rows = {df['outlier'].mean():.2%} "
            f"({df['too_cheap'].sum()} too cheap = {df['too_cheap'].mean():.2%}, "
            f"{df['too_expensive'].sum()} too expensive = {df['too_expensive'].mean():.2%})",
            f"Loads with ratio between 0.8 and 1.25: {df['ratio'].between(0.8, 1.25).mean():.2%}",
            'By quarter:   ' + ', '.join(f'{k} {v:.2f}%' for k, v in by_quarter.items()),
            'By equipment: ' + ', '.join(f'{k} {v:.2f}%' for k, v in by_equipment.items()),
            'Analysis only: no rows are removed here. In training, outliers are dropped from the train rows only.',
        ]
        r.tables['fitted coefficients'] = self.model.coef_.round(3).rename('fitted')
        r.tables['equipment effect: data vs formula'] = check
        r.tables['expected price by distance and equipment'] = table
        return r
