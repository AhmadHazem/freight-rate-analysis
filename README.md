# Freight Rate Prediction

Predict the `posted_rate` (price in dollars) of a freight load from its route, distance, equipment, weight and
market signals. The assessment brief is in `freight-rate-ml-assessment-1-.pdf`.

- **Train / test data:** `train-test.csv`, 48,000 loads from 2025-01-01 to 2025-10-31, with `posted_rate`
- **Validation data:** `validation.csv`, 12,000 loads from 2025-11-01 to 2025-12-31, without `posted_rate`
- **December chart:** `december-chart-inputs.csv`, one fixed load (Lexington → Fort Wayne, 360 mi, Dry Van,
  32,000 lb) for every day of December 2025

The deliverables are `validation_predictions.csv` (one `predicted_rate` per validation `load_id`) and the December
file with its `predicted_rate` column filled. `score.py` checks both files and draws `candidate_december.png`.

---

## 1. Environment

The project was developed and tested in a conda environment called `freight_analysis`:

| Package      | Tested version |
|--------------|----------------|
| Python       | 3.12.15        |
| numpy        | 2.5.4          |
| pandas       | 3.0.6          |
| matplotlib   | 3.11.2 (3.9+ needed) |
| scikit-learn | 1.9.1          |
| xgboost      | 3.4.1          |
| joblib       | 1.6.0          |
| ipykernel    | 7.3.0 (only for `main.ipynb`) |

### Setup

```bash
conda create -n freight_analysis python=3.12
conda activate freight_analysis

python -m pip install -r requirements.txt                   # numpy, pandas, matplotlib (also used by score.py)
python -m pip install scikit-learn xgboost joblib ipykernel  # model + notebook kernel
```

To use the notebook, select the **freight_analysis** kernel in Jupyter / VS Code.

Run every command below **from the project root** (the folder containing this README). The code finds the data
files relative to the project root, so it works from any shell once the environment is active.

---

## 2. Quick start: the full run

```bash
# 1. Exploratory analysis: summaries, figures and a Markdown report
python -m freight_rate.analysis.main

# 2. Training data: raw -> cleaned
python -m freight_rate.data.prepare

# 3. Cross-validate, train and save the model
python -m freight_rate.model.train

# 4. Prediction data: raw -> cleaned (the same cleaning as the training data)
python -m freight_rate.data.prepare --mode infer --input validation.csv
python -m freight_rate.data.prepare --mode infer --input december-chart-inputs.csv

# 5. Predict
python -m freight_rate.model.infer --input data/processed/validation_clean.csv \
    --template validation-predictions-template.csv --output validation_predictions.csv
python -m freight_rate.model.infer --input data/processed/december-chart-inputs_clean.csv \
    --output december-chart-predictions.csv

# 6. Score
python score.py --predictions validation_predictions.csv --december-predictions december-chart-predictions.csv
```

Step 3 takes the longest because it trains 5 XGBoost models (4 CV folds + the final model). It finishes in a few
minutes on a laptop; add `--skip-cv` to train only the final model.

### What each step writes

| Step | Output |
|------|--------|
| analysis | `reports/analysis/report.md`, `reports/analysis/figures/*.png` |
| prepare (train) | `data/processed/train_clean.csv` |
| train | `models/freight_rate_model.joblib`, `models/cv_results.csv` |
| prepare (infer) | `data/processed/validation_clean.csv`, `data/processed/december-chart-inputs_clean.csv` |
| infer | `validation_predictions.csv`, `december-chart-predictions.csv` |
| score | `scorer_results/candidate_december.png` |

---

## 3. Project structure

```
freight-rate-analysis/
├── main.ipynb                 # original exploration notebook (source of everything below)
├── insight_logger.md          # insights [INS*], thoughts [THOUGHT*] and conclusions [CONC*]
├── score.py                   # provided scorer
├── train-test.csv, validation.csv, december-chart-inputs.csv, validation-predictions-template.csv
└── freight_rate/
    ├── analysis/              # every analysis from the notebook, as classes
    │   ├── base.py            #   Analysis base class + AnalysisResult (notes, tables, figures)
    │   ├── features.py        #   FeatureOverview
    │   ├── cities.py          #   CityRouteAnalysis
    │   ├── dates.py           #   DateAnalysis
    │   ├── equipment.py       #   EquipmentAnalysis
    │   ├── distributions.py   #   DistributionAnalysis
    │   ├── outliers.py        #   OutlierAnalysis
    │   └── main.py            #   ENTRY: runs them all, prints, saves figures, writes report.md
    ├── data/                  # reading and cleaning data
    │   ├── data_loader.py     #   DataLoader: CSV paths, reading, date/quarter columns
    │   ├── preprocessing.py   #   drop_columns, abs_weight, GroupImputer, ExpectedRateModel
    │   ├── cleaner.py         #   DataCleaner: prepare_training / prepare_inference
    │   └── prepare.py         #   ENTRY: raw CSV -> cleaned CSV (--mode train | infer)
    └── model/                 # model building, training and prediction
        ├── pipeline.py        #   FreightPipelineBuilder: sklearn pipeline + XGBoost
        ├── modeling.py        #   FreightRateModel: cross_validate, fit, predict, save, load
        ├── train.py           #   ENTRY: CV -> fit -> save model
        └── infer.py           #   ENTRY: load model -> predict a CSV
```

The three packages are independent and connect only through imports:

```
analysis ──uses──► data ◄──uses── model
```

- `data` imports neither of the others. You can run it on its own.
- `model` imports `data` for the cleaner and the outlier formula. It trains on the CSV that `data.prepare` writes.
- `analysis` imports `data` for the loader and the outlier formula, so the analysis and the model use the same
  definition of an outlier.

---

## 4. Analysis module (`freight_rate.analysis`)

Each class reproduces one section of `main.ipynb` and returns an `AnalysisResult` containing findings (tagged with
IDs from `insight_logger.md`), tables and matplotlib figures.

| Class | Findings |
|-------|----------|
| `FeatureOverview` | Dataset sizes and `describe()` output. Missing values: weight (300 / 165), market_index (374 / 249) |
| `CityRouteAnalysis` | 64 cities in train-test, 72 in validation (8 unseen: Chicago, San Diego, …). 736 routes appear only in validation. Coordinates are consistent across files. On a single route the distance and price still vary, so city names add little |
| `DateAnalysis` | Monthly mean price runs from $2,256 to $2,497. The date matters, so it is used for the CV split rather than as a feature |
| `EquipmentAnalysis` | Per mile, Reefer costs about 13% and Flatbed about 8% more than Dry Van, compared within the same distance bucket |
| `DistributionAnalysis` | distance correlates 0.90 with the price. Weight is capped at 47,500 (2.5% of rows) and 292 weights are negative. The price is right-skewed (skew 1.90, and -0.49 after taking the log) |
| `OutlierAnalysis` | Fits `ln(price) = start + slope·ln(distance) + equipment shift` (slope 0.871). Loads whose price is below 0.5× or above 2× the expected value are flagged: 677 rows (1.41%) |

```bash
python -m freight_rate.analysis.main                           # run everything, save to reports/analysis/
python -m freight_rate.analysis.main --show                    # also open the figures in windows
python -m freight_rate.analysis.main --only OutlierAnalysis DateAnalysis
python -m freight_rate.analysis.main --seed -1                 # new random samples on each run
python -m freight_rate.analysis.main --output-dir some/folder
```

---

## 5. Data module (`freight_rate.data`)

`prepare.py` cleans a raw CSV and writes the result to `data/processed/`. The cleaning only changes values row by
row: no statistics are learned from the data, so training and prediction data receive exactly the same treatment.

| Step | Why |
|------|-----|
| Drop `pickup_lat/lon`, `delivery_lat/lon` | [CONC1] Coordinates add nothing beyond distance |
| `abs(weight)` | [CONC2] Negative weights look like input errors (-47,500 → 47,500) |
| Parse `date`, add `quarter` | `quarter` is the cross-validation fold label |
| Add `day` = day of month (`2025-12-01` → `1`) | Year and month dropped; `day` is an ordinal model feature (1–31). `date` stays in the file for the output layout but is not a feature |
| **train mode only:** add `is_outlier` | Loads priced below 0.5× or above 2× their expected price ([THOUGHT8]). Rows are flagged, not deleted (see below) |
| **infer mode only:** drop `posted_rate` / `predicted_rate` | The target must never enter as a feature |

```bash
python -m freight_rate.data.prepare                                   # train-test.csv -> data/processed/train_clean.csv
python -m freight_rate.data.prepare --outlier-low 0.6 --outlier-high 1.67
python -m freight_rate.data.prepare --mode infer --input validation.csv
python -m freight_rate.data.prepare --mode infer --input any.csv --output somewhere/any_clean.csv
```

**What 0.5 and 2.0 mean:** `ratio = actual price / expected price`. A ratio of 1.0 means the load is priced as
expected. Below 0.5 means it is less than half the expected price; above 2.0 means it is more than double. 98.6%
of loads fall between 0.8 and 1.25, so only clearly wrong prices are flagged.

**Why flag instead of delete:** cross-validation must test each model on its held-out quarter *including*
outliers. Otherwise the reported error would look better than real performance.

---

## 6. Model module (`freight_rate.model`)

### Pipeline (`pipeline.py`)

```
drop load_id / coordinates
 → impute weight with the route (pickup, delivery) median, falling back to the global median   [THOUGHT5]
 → impute market_index with the route mean, falling back to the global mean
 → weight: abs + StandardScaler | distance, market_index, quote_signal: StandardScaler | equipment: one-hot
   | day: ordinal integer 1-31, passed through as is
 → XGBRegressor (500 trees, lr 0.05, depth 6) trained on log(posted_rate), predictions exp()'d back to $   [THOUGHT7]
```

`pickup`, `delivery`, `date` and `quarter` are not model features [CONC1, CONC4]; from the date only `day` is used. `pickup` and `delivery` are only
used as keys for the route imputation. Everything learned (imputation values, scalers, encoder, trees) lives inside
the pipeline, so it is fitted on training rows only and saved with the model.

### Training (`train.py`)

1. **Leave-one-quarter-out CV** [THOUGHT6]: four folds, each holding out one whole quarter of 2025. In each fold,
   outliers are re-flagged using a formula fitted on that fold's training rows only, and removed from the training
   rows only.
2. **Final fit** on all cleaned rows except those with `is_outlier` set.
3. **Save** the fitted model to `models/freight_rate_model.joblib` and the CV table to `models/cv_results.csv`.

```bash
python -m freight_rate.model.train
python -m freight_rate.model.train --skip-cv
python -m freight_rate.model.train --data path/to/clean.csv --model-out path/to/model.joblib
```

**CV results** (MAE in $):

| Held-out quarter | Rows | Train MAE | Test MAE | Test MAE without outliers |
|------------------|------|-----------|----------|---------------------------|
| 2025Q1 | 14,291 | 40.80 | 120.43 | 75.97 |
| 2025Q2 | 14,515 | 42.31 | 104.20 | 54.50 |
| 2025Q3 | 14,341 | 36.97 | 119.69 | 68.25 |
| 2025Q4 | 4,853  | 41.48 | 134.06 | 73.97 |
| **Mean** |      | **40.39** | **119.59** | **68.17** |

### Inference (`infer.py`)

`infer.py` loads the saved model, runs `DataCleaner.prepare_inference` on the input and predicts. The input can be
the cleaned CSV from `prepare --mode infer` or the raw CSV: cleaning a file that is already clean changes nothing,
and both give identical predictions.

- With `--template`, the predictions are written into the template by `load_id`, keeping the template's row order.
- Without it, the output is the input file with `predicted_rate` filled in.
- Columns missing from the input (for example `market_index` and `quote_signal` in the December file) are treated
  as missing values. `market_index` is filled with the route mean; XGBoost handles a missing `quote_signal` natively.

```bash
python -m freight_rate.model.infer --input data/processed/validation_clean.csv \
    --template validation-predictions-template.csv --output validation_predictions.csv
python -m freight_rate.model.infer --input validation.csv --template validation-predictions-template.csv \
    --output validation_predictions.csv --model path/to/model.joblib
```

---

## 7. Known limitations

- **The December chart is flat.** All 31 December rows have identical features, the file has no `market_index` or
  `quote_signal`, and `date` is not a model feature. As a result, every day gets the same prediction (about $978).
  To show day-to-day movement, the model would need a time or market-trend signal.
- **Data shift.** Validation `market_index` averages 0.93, against 1.08 in train-test. Validation also contains 8
  cities and 736 routes never seen in training. For those routes, imputation falls back to global values.
- **Train/test gap.** Train MAE is about 40 and test MAE about 120. The 1.4% of test rows that are outliers account for
  about 45% of the total test error, because the model is not trained to reproduce them.

---

## 8. Submission checklist

- GitHub repository with code, dependencies and run instructions (this README)
- `validation_predictions.csv`
- PDF or DOCX report with the validation results, the data split approach and `candidate_december.png`
- 2-3 minute Loom link
