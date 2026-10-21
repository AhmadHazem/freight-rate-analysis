"""Load a saved model and predict `predicted_rate` for a CSV.

    python -m freight_rate.data.prepare --mode infer --input validation.csv
    python -m freight_rate.model.infer --input data/processed/validation_clean.csv \
        --template validation-predictions-template.csv --output outputs/validation_predictions.csv

The input can be the cleaned CSV from `data.prepare --mode infer` or the raw CSV: it goes through
DataCleaner.prepare_inference either way (idempotent), then through the saved pipeline, which
applies the imputation / scaling / encoding learned on the training data.

With --template, predictions are written into the template by load_id (template order kept).
Without it, the output is the input file with its predicted_rate column filled (columns added by the
cleaner, such as `quarter`, are left out).
"""
import argparse
from pathlib import Path

import pandas as pd

from ..data.cleaner import ADDED_COLUMNS, DataCleaner
from ..data.data_loader import DataLoader
from .modeling import FreightRateModel
from .train import DEFAULT_MODEL_PATH


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--model', default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument('--template', type=Path, help='CSV with load_id, predicted_rate to fill')
    args = parser.parse_args(argv)

    model = FreightRateModel.load(args.model)
    raw = DataLoader.read(args.input)
    # Same row-wise cleaning as the training data (a no-op if the input is already cleaned)
    features = DataCleaner().prepare_inference(raw)
    preds = model.predict(features).round(2)

    if args.template:
        out = DataLoader.read(args.template)
        out['predicted_rate'] = out['load_id'].map(pd.Series(preds, index=raw['load_id']))
    else:
        # Original columns only (score.py checks the exact December layout), predicted_rate last
        out = raw.drop(columns=ADDED_COLUMNS + ['predicted_rate'], errors='ignore')
        out['predicted_rate'] = preds

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f'Predicted {len(preds):,} rows with {args.model}')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
