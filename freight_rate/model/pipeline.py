import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from ..data.preprocessing import GroupImputer, drop_columns

# [CONC1] cities' coordinates are not needed; pickup/delivery are kept until imputation (group keys).
# DataCleaner already drops the coordinates; listing them here keeps the pipeline safe on raw input.
DROP_COLUMNS = ['load_id', 'pickup_lat', 'pickup_lon', 'delivery_lat', 'delivery_lon']

XGB_PARAMS = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=-1)


class FreightPipelineBuilder:
    """Builds the full sklearn pipeline: drop -> impute -> scale/encode -> XGBoost on log(rate)."""

    def __init__(self, drop_cols=None, xgb_params=None):
        self.drop_cols = DROP_COLUMNS if drop_cols is None else drop_cols
        self.xgb_params = {**XGB_PARAMS, **(xgb_params or {})}

    def build_imputers(self):
        # [THOUGHT5] impute from the same route first, global value as fallback
        return [
            ('imp_weight', GroupImputer(cols=['weight'], group_by=['pickup', 'delivery'], strategy='median')),
            ('imp_market', GroupImputer(cols=['market_index'], group_by=['pickup', 'delivery'], strategy='mean')),
        ]

    def build_preprocessor(self):
        weight_branch = Pipeline([
            ('abs', FunctionTransformer(np.abs)),      # [CONC2] -47500 -> 47500
            ('scale', StandardScaler()),
        ])
        return ColumnTransformer([
            ('weight', weight_branch, ['weight']),
            ('num', StandardScaler(), ['distance', 'market_index', 'quote_signal']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['equipment']),
            # Ordinal: the integer day keeps its order (1 < 2 < ... < 31), so the trees can split on it
            ('day', 'passthrough', ['day']),
        ], remainder='drop')   # pickup, delivery, date, quarter are dropped here

    def build_model(self):
        # [THOUGHT7] trains on log(y), predict() returns dollars via exp()
        return TransformedTargetRegressor(
            regressor=XGBRegressor(**self.xgb_params),
            func=np.log,
            inverse_func=np.exp,
        )

    def build(self):
        return Pipeline([
            ('drop', FunctionTransformer(drop_columns, kw_args={'drop_cols': self.drop_cols})),
            *self.build_imputers(),
            ('prep', self.build_preprocessor()),
            ('model', self.build_model()),
        ])
