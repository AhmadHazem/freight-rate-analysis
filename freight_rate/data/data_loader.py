from pathlib import Path

import pandas as pd

# Project root: freight_rate/data/data_loader.py -> two levels up from the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Generated (cleaned) CSVs live outside the code package
PROCESSED_DIR = PROJECT_ROOT / 'data' / 'processed'


class DataLoader:
    """Reads the project CSV files from `data_dir`."""

    TRAIN_TEST = 'train-test.csv'
    VALIDATION = 'validation.csv'
    DECEMBER = 'december-chart-inputs.csv'
    TEMPLATE = 'validation-predictions-template.csv'

    def __init__(self, data_dir=PROJECT_ROOT):
        self.data_dir = Path(data_dir)

    def _read(self, name):
        return pd.read_csv(self.data_dir / name)

    @staticmethod
    def read(path):
        """Read any CSV by path (e.g. a cleaned file or a file to predict)."""
        return pd.read_csv(path)

    @staticmethod
    def add_time_columns(df):
        """Parse `date` and add `quarter` (e.g. '2025Q1'), the fold label used in cross-validation."""
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['quarter'] = df['date'].dt.to_period('Q').astype(str)
        return df

    def load_train_test(self, add_time_columns=False):
        df = self._read(self.TRAIN_TEST)
        return self.add_time_columns(df) if add_time_columns else df

    def load_validation(self, add_time_columns=False):
        df = self._read(self.VALIDATION)
        return self.add_time_columns(df) if add_time_columns else df

    def load_december(self):
        return self._read(self.DECEMBER)

    def load_template(self):
        return self._read(self.TEMPLATE)
