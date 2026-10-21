import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import Analysis, AnalysisResult

EQUIPMENT = ['Dry Van', 'Reefer', 'Flatbed']
COLORS = {'Dry Van': 'tab:blue', 'Reefer': 'tab:green', 'Flatbed': 'tab:red'}
DISTANCE_BINS = [0, 250, 500, 1000, 1500, 2000, 2500, 3500]


class EquipmentAnalysis(Analysis):
    """Equipment effect on price and rate per mile, compared within distance buckets [INS14]."""
    title = '1.1.5 Equipment analysis'

    def run(self):
        r = AnalysisResult(self.title)
        d = self.train_test_df.copy()
        d['rpm'] = d['posted_rate'] / d['distance']
        d['dist_bin'] = pd.cut(d['distance'], DISTANCE_BINS)    # compare like with like

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        axes[0, 0].boxplot([d.loc[d.equipment == e, 'posted_rate'] for e in EQUIPMENT],
                           tick_labels=EQUIPMENT, showfliers=False)
        axes[0, 0].set_title('posted_rate by equipment (outlier dots hidden)')

        axes[0, 1].boxplot([d.loc[d.equipment == e, 'rpm'] for e in EQUIPMENT],
                           tick_labels=EQUIPMENT, showfliers=False)
        axes[0, 1].set_title('Rate per mile by equipment')

        rpm_bin = d.groupby(['dist_bin', 'equipment'], observed=True)['rpm'].median().unstack()[EQUIPMENT]
        for e in EQUIPMENT:
            axes[1, 0].plot(range(len(rpm_bin)), rpm_bin[e], marker='o', color=COLORS[e], label=e)
        axes[1, 0].set_xticks(range(len(rpm_bin)), [str(b) for b in rpm_bin.index], rotation=30)
        axes[1, 0].set_title('Median rate per mile by distance bucket')
        axes[1, 0].set_xlabel('distance (mi)')
        axes[1, 0].legend()

        # Premium over Dry Van at the same distance
        prem = rpm_bin[['Reefer', 'Flatbed']].div(rpm_bin['Dry Van'], axis=0).sub(1).mul(100)
        x = np.arange(len(prem))
        axes[1, 1].bar(x - .2, prem['Reefer'], .4, color=COLORS['Reefer'], label='Reefer')
        axes[1, 1].bar(x + .2, prem['Flatbed'], .4, color=COLORS['Flatbed'], label='Flatbed')
        axes[1, 1].set_xticks(x, [str(b) for b in prem.index], rotation=30)
        axes[1, 1].axhline(0, color='k', lw=.5)
        axes[1, 1].set_title('% premium over Dry Van (same distance bucket)')
        axes[1, 1].legend()
        fig.tight_layout()
        r.figures['equipment_overview'] = fig

        summary = d.groupby('equipment')[['posted_rate', 'rpm', 'distance']].median().loc[EQUIPMENT].round(2)
        summary['n_loads'] = d['equipment'].value_counts()[EQUIPMENT]
        summary['rpm_premium_vs_dry_van_%'] = ((summary['rpm'] / summary.loc['Dry Van', 'rpm'] - 1) * 100).round(1)

        r.notes.append('[INS14] Median rate-per-mile premium over Dry Van: ' + ', '.join(
            f"{e} {summary.loc[e, 'rpm_premium_vs_dry_van_%']:+.1f}%" for e in ['Reefer', 'Flatbed']))
        r.tables['equipment summary (medians)'] = summary
        r.tables['premium over Dry Van by distance bucket (%)'] = prem.round(1).T
        return r
