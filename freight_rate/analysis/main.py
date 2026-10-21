"""Run every analysis: print the summaries, save the figures and write a Markdown report.

    python -m freight_rate.analysis.main [--output-dir reports/analysis] [--only DateAnalysis ...] [--show]
"""
import argparse
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from ..data.data_loader import PROJECT_ROOT, DataLoader
from . import ALL_ANALYSES


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def print_result(result):
    bar = '=' * 80
    print(f'\n{bar}\n{result.title}\n{bar}')
    for note in result.notes:
        print(f'- {note}')
    for name, table in result.tables.items():
        print(f'\n[{name}]\n{table.to_string()}')


def save_figures(result, figures_dir):
    """Save each figure as <analysis>__<figure>.png and return {figure name: path}."""
    paths = {}
    for name, fig in result.figures.items():
        path = figures_dir / f'{slug(result.title)}__{name}.png'
        fig.savefig(path, dpi=110, bbox_inches='tight')
        paths[name] = path
    return paths


def write_report(sections, report_path):
    lines = ['# Freight-rate analysis report', '']
    for result, fig_paths in sections:
        lines += [f'## {result.title}', '']
        lines += [f'- {note}' for note in result.notes] + ['']
        for name, table in result.tables.items():
            lines += [f'**{name}**', '', '```text', table.to_string(), '```', '']
        for name, path in fig_paths.items():
            lines += [f'![{name}]({path.relative_to(report_path.parent).as_posix()})', '']
    report_path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    names = [cls.__name__ for cls in ALL_ANALYSES]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=PROJECT_ROOT, type=Path)
    parser.add_argument('--output-dir', default=PROJECT_ROOT / 'reports' / 'analysis', type=Path)
    parser.add_argument('--only', nargs='+', choices=names, help='run only these analyses')
    parser.add_argument('--seed', type=int, default=0, help='random_state for sampling (-1 = random each run)')
    parser.add_argument('--show', action='store_true', help='also open the figures in windows')
    args = parser.parse_args(argv)

    if not args.show:
        matplotlib.use('Agg')       # save-only, no windows

    loader = DataLoader(args.data_dir)
    train_test_df, validation_df = loader.load_train_test(), loader.load_validation()
    seed = None if args.seed < 0 else args.seed

    figures_dir = args.output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    with pd.option_context('display.width', 200, 'display.max_columns', 50):
        for cls in ALL_ANALYSES:
            if args.only and cls.__name__ not in args.only:
                continue
            result = cls(train_test_df, validation_df, seed=seed).run()
            print_result(result)
            sections.append((result, save_figures(result, figures_dir)))
            if not args.show:
                plt.close('all')
        report_path = args.output_dir / 'report.md'
        write_report(sections, report_path)

    print(f'\nFigures saved to {figures_dir}\nReport written to {report_path}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
