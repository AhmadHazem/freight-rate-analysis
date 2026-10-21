from .data_loader import DataLoader
from .preprocessing import ExpectedRateModel, abs_weight, drop_columns

# [CONC1] coordinates are not needed; pickup/delivery stay as imputation group keys
COORDINATE_COLUMNS = ['pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']
# Columns the model cannot do without; the rest (weight, market_index, quote_signal) may be missing / NaN
REQUIRED_COLUMNS = ['pickup', 'delivery', 'distance', 'equipment']
# Target and the empty prediction placeholder never go in as features
TARGET_COLUMNS = ['posted_rate', 'predicted_rate']
# Columns this cleaner adds; prediction outputs drop them to keep the original file layout
ADDED_COLUMNS = ['quarter', 'day', 'is_outlier']


class DataCleaner:
    """Row-wise cleaning shared by training and inference, plus outlier flagging for training data.

    Anything learned from data (imputation values) stays in the model pipeline, so it is fitted
    on training rows only and saved with the model.
    """
    def __init__(self, outlier_low=0.5, outlier_high=2.0):
        self.outlier_low = outlier_low
        self.outlier_high = outlier_high

    def clean(self, df):
        df = drop_columns(df, COORDINATE_COLUMNS)
        if 'weight' in df:
            df = abs_weight(df)                       # [CONC2]
        if 'date' in df:
            df = DataLoader.add_time_columns(df)      # date -> datetime, + quarter (CV fold label)
            # Day of month only, year and month dropped: 2025-12-01 -> 1, ordinal 1-31.
            # `date` itself stays in the file (not a model feature) so outputs keep their original layout.
            df['day'] = df['date'].dt.day
        return df

    def flag_outliers(self, df):
        """Add `is_outlier` from the expected-price formula fitted on all of `df` [THOUGHT8].

        Rows are flagged, not removed: cross-validation re-flags each fold from its own train
        rows, and keeps outliers in the held-out quarter so the test MAE stays honest.
        """
        df = df.copy()
        df['is_outlier'] = ExpectedRateModel(self.outlier_low, self.outlier_high).fit(df).flag(df)
        return df

    @staticmethod
    def check_columns(df, required):
        missing = [c for c in required if c not in df]
        if missing:
            raise ValueError(f'Missing required columns: {missing}')

    def prepare_training(self, df):
        self.check_columns(df, REQUIRED_COLUMNS + ['posted_rate'])
        return self.flag_outliers(self.clean(df))

    def prepare_inference(self, df):
        """Same row-wise cleaning as training, without the target and without outlier flags
        (outliers are judged on posted_rate, which inference data does not have).
        Safe to run on an already-cleaned file: every step is idempotent."""
        self.check_columns(df, REQUIRED_COLUMNS)
        return self.clean(drop_columns(df, TARGET_COLUMNS))
