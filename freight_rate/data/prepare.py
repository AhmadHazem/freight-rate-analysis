"""Read a raw CSV and write the cleaned CSV for training or inference.

    python -m freight_rate.data.prepare                                    # train-test.csv -> data/processed/train_clean.csv
    python -m freight_rate.data.prepare --mode infer --input validation.csv  # -> data/processed/validation_clean.csv

train : row-wise cleaning + `is_outlier` flags (needs posted_rate)
infer : the same row-wise cleaning, target / placeholder columns removed, no outlier flags
"""
import argparse
from pathlib import Path

from .cleaner import DataCleaner
from .data_loader import PROCESSED_DIR, PROJECT_ROOT, DataLoader


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mode', choices=['train', 'infer'], default='train')
    parser.add_argument('--input', type=Path, help='default (train mode): train-test.csv')
    parser.add_argument('--output', type=Path, help='default: data/processed/train_clean.csv or <input>_clean.csv')
    parser.add_argument('--outlier-low', default=0.5, type=float, help='actual / expected below this = outlier')
    parser.add_argument('--outlier-high', default=2.0, type=float, help='actual / expected above this = outlier')
    args = parser.parse_args(argv)

    if args.input is None:
        if args.mode == 'infer':
            parser.error('--input is required in infer mode')
        args.input = PROJECT_ROOT / DataLoader.TRAIN_TEST
    if args.output is None:
        name = 'train_clean.csv' if args.mode == 'train' else f'{args.input.stem}_clean.csv'
        args.output = PROCESSED_DIR / name

    raw = DataLoader.read(args.input)
    cleaner = DataCleaner(args.outlier_low, args.outlier_high)
    clean = cleaner.prepare_training(raw) if args.mode == 'train' else cleaner.prepare_inference(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.output, index=False)
    print(f'Read {len(raw):,} rows from {args.input} ({args.mode} mode)')
    if args.mode == 'train':
        print(f"Flagged {clean['is_outlier'].sum():,} outliers ({clean['is_outlier'].mean():.2%})")
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
