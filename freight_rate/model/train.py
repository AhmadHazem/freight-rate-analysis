"""Train on the cleaned CSV, cross-validate and save the model.

    python -m freight_rate.model.train [--data data/processed/train_clean.csv]
                                                 [--model-out models/freight_rate_model.joblib] [--skip-cv]

Run `python -m freight_rate.data.prepare` first to create the cleaned CSV.
"""
import argparse
from pathlib import Path

from ..data.data_loader import PROCESSED_DIR, PROJECT_ROOT, DataLoader
from .modeling import FreightRateModel

DEFAULT_MODEL_PATH = PROJECT_ROOT / 'models' / 'freight_rate_model.joblib'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=PROCESSED_DIR / 'train_clean.csv', type=Path)
    parser.add_argument('--model-out', default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument('--skip-cv', action='store_true', help='skip leave-one-quarter-out CV')
    args = parser.parse_args(argv)

    if not args.data.is_file():
        raise SystemExit(f'Cleaned data not found: {args.data}\n'
                         'Run: python -m freight_rate.data.prepare')
    df = DataLoader.read(args.data)
    model = FreightRateModel()

    if not args.skip_cv:
        print('Leave-one-quarter-out cross-validation')
        res = model.cross_validate(df)
        print(res.round(2).to_string(index=False))
        for name, value in model.summarize_cv(res).items():
            print(f'{name:<22}: {value:.2f}')
        cv_path = args.model_out.with_name('cv_results.csv')
        cv_path.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(cv_path, index=False)
        print(f'Wrote {cv_path}\n')

    model.fit(df)
    model.save(args.model_out)
    print(f'Trained on {len(df) - model.n_dropped_:,} rows ({model.n_dropped_:,} outliers dropped)')
    print(f'Saved model to {args.model_out}')


if __name__ == '__main__':
    main()
