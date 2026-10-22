# Car-price regression with five manual notebooks

Five notebooks cover exploration, cleaning, feature engineering, training and held-out evaluation. Run them in Jupyter or VS Code. Settings and model code are included in the notebooks.

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter notebook regression/car_price_prediction/notebooks
```

Choose a Python kernel with the project dependencies, or the repository's `.venv/bin/python` interpreter in VS Code. The notebooks find their data when started from the repository root, the car-price project folder or `notebooks/`.

## Notebook stages

1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb): price/mileage/engine/year distributions, categorical relationships, missingness, outliers and centered VIF.
2. [02_data_cleaning.ipynb](notebooks/02_data_cleaning.ipynb): label validation, exact duplicates, suspicious measurements and training-fitted preprocessing.
3. [03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb): snapshot age, log mileage, mileage per year and missing-engine indicators.
4. [04_training.ipynb](notebooks/04_training.ipynb): model selection, CV, optional tuning, overfitting diagnostics and the training report.
5. [05_testing_evaluation.ipynb](notebooks/05_testing_evaluation.ipynb): frozen-candidate evaluation, model comparisons, residuals and price-band/percentage-error diagnostics.

Run all five in order for the full analysis. For training/testing alone, **Run All in 04, then Run All in 05**. Training includes its own cleaning and feature definitions, so 01–03 are optional. Testing works in a fresh kernel and loads the latest saved candidates and settings.

The observations describe the saved run. Each model is scored from its own predictions in price units, with percentage errors and price-band summaries for context.

## Choose models and settings

Edit the settings cell near the top of **04_training.ipynb**:

```python
SELECTED_MODELS = ["DummyMedian", "LinearRegression", "RandomForest", "DecisionTree"]
```

`None` compares every installed family. Available names are `DummyMedian`, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `DecisionTree`, `RandomForest`, `GradientBoostedTreesModel`, `HistogramGradientBoostedTreesModel`, and optional `XGBoost`. An unknown or explicitly selected unavailable model raises an error. Tuning adds variants only from the selected families.

The same cell controls the seed, split fractions, CV folds, tuning, tuning shortlist size, overfitting/underperformance thresholds and `reference_year`. For a shorter run, set `cv_folds` to `3` and `tune` to `False`. To change the candidates, rerun 04 and then 05. Diagnostic settings in 01–03 should match 04 when comparing observations.

## Data policy

The source CSV is preserved under `car_prices_dataset/train/car_price.csv`.

- Generate stable `RowId` values from the original one-based CSV row number. IDs are excluded from model inputs.
- Exclude missing, nonfinite or nonpositive prices from supervised analysis. Predictors with missing values remain usable.
- Keep the first occurrence of each exact labeled source record before splitting, so identical records cannot appear in different splits.
- Retain valid price/mileage extremes and older cars. No global percentile trimming is applied.
- Omit the high-cardinality `Model` predictor to limit the number of categories in the baseline; keep it in error tables for inspection.
- Mark engine volumes outside `(0, 6.5]` as unknown under the dataset-specific validity rule. Also mark invalid mileage/year measurements as unknown, then impute inside each CV fold.
- Use a fixed snapshot year of **2016**, the latest model year in this source, for age features. No sale dates or mileage-unit conversion are assumed.

The current source has 4,345 rows. Excluding 172 missing prices and 50 exact duplicate labeled records leaves 4,123 usable rows. Default splits are **2,638 training, 660 validation and 825 ground truth**. Source data remains unchanged.

Learned imputation, scaling and encoding are fitted inside training CV folds using scikit-learn estimator composition. Models fit log1p prices and invert predictions with expm1. Every final candidate is fitted only on training rows; testing evaluates those exact fits.

If you edit a cleaning or feature rule, keep its definitions consistent across 02–05 and rerun training/testing. Testing checks source data, saved split/model/score fingerprints and matching cleaning/feature definitions before evaluation.

## Generated files

```text
car_price_prediction/
├── readme.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_training.ipynb
│   └── 05_testing_evaluation.ipynb
├── car_prices_dataset/
│   ├── train/
│   │   ├── car_price.csv               # original source
│   │   ├── training_data.csv           # written by 04
│   │   └── validation_data.csv         # written by 04
│   └── ground_truth/
│       ├── ground_truth_features.csv   # written by 04
│       ├── ground_truth.csv            # RowId and known Price, written by 04
│       ├── ground_truth_predictions.csv # written by 05
│       └── ground_truth_comparison.csv  # written by 05
├── artifacts/
│   ├── candidates.joblib               # all training-fitted candidates
│   ├── run_metadata.json               # settings, data audit, tuning and fingerprints
│   ├── model_comparison.csv            # training/CV/validation metrics
│   ├── training_testing_comparison.csv # combined metrics and flags
│   ├── prediction_errors.csv           # per-car errors for every candidate
│   └── price_band_comparison.csv        # below-5k, 5k–50k and 50k+ diagnostics
└── reports/
    ├── training_performance.html
    ├── testing_evaluation.html
    └── model_comparison.html
```

Reports embed Plotly and open offline. Successful training overwrites current training outputs and removes outdated testing tables/reports. Run 05 to regenerate them for the new candidates.

## Read the comparison

Price-unit RMSE is the main metric. MAE, RMSLE, R², mean absolute percentage error and the percentage of predictions within 50% of actual price provide context. The 50% tolerance is a descriptive error threshold.

- `overfitting_flag`: validation RMSE / training RMSE exceeds `overfit_ratio` (default 1.5).
- `best_model_flag`: lowest ground-truth RMSE, including numerical ties.
- `underperforming_flag`: ground-truth RMSE is at least `underperformance_ratio` times the best observed RMSE (default 1.25).

The comparison report shows training, validation and testing RMSE/R² together. Testing also includes price-band scores and the largest relative errors, plus notebook analysis of cars below 50k. The lowest held-out RMSE is highlighted alongside the other metrics. Repeated tuning after inspecting ground truth reduces its independence.
