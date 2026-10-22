# House-price regression

Open the notebooks in Jupyter or VS Code and run their cells manually. Code and editable run settings are included in the notebooks.

From the repository root, activate your Python environment and install dependencies if needed:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter notebook regression/house_price_prediction/notebooks
```

In VS Code, choose the repository's `.venv/bin/python` interpreter. In Jupyter, choose a Python kernel with the installed dependencies. The notebooks locate the dataset when started from the repository root, this project folder, or `notebooks/`.

## Notebooks

1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb): inspect training-only price distributions, feature relationships, missingness, redundancy and outliers.
2. [02_data_cleaning.ipynb](notebooks/02_data_cleaning.ipynb): inspect structural absence rules, label validation, imputation, scaling and encoding.
3. [03_feature_engineering.ipynb](notebooks/03_feature_engineering.ipynb): inspect space, bathroom, age, presence and ordinal-quality features.
4. [04_training.ipynb](notebooks/04_training.ipynb): choose models, train, cross-validate, optionally tune, and produce the training report.
5. [05_testing_evaluation.ipynb](notebooks/05_testing_evaluation.ipynb): load the last training run, score held-out known prices, and produce testing and model-comparison reports.

Run all five in order for the full analysis. For a model comparison alone, **Run All in 04, then Run All in 05**. Training includes data splitting, cleaning and feature engineering, so 01–03 do not have to run first. Testing works in a fresh kernel and does not retrain models.

## Choose models in training

Edit the settings cell near the top of **04_training.ipynb** before running it:

```python
SELECTED_MODELS = ["DummyMedian", "Ridge", "GradientBoostedTreesModel"]
```

`None` compares every installed candidate. Available names are `DummyMedian`, `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `RandomForest`, `GradientBoostedTreesModel`, `HistogramGradientBoostedTreesModel`, and optional `XGBoost`. Selecting an unavailable or unknown name raises an error. XGBoost is optional unless explicitly selected.

The same cell controls `random_state`, `ground_truth_fraction`, `validation_fraction`, `cv_folds`, `tune`, `tune_top_n`, `overfit_ratio` and `underperformance_ratio`. For a faster run, set `cv_folds` to `3` and `tune` to `False`. Tuning adds variants only from selected model families, shortlisted by validation performance and tuned using training CV.

Testing automatically uses the saved training settings and all candidates from that run. To change models or split settings, edit 04 and rerun 04 then 05. Settings in 01–03 control their independent diagnostic views; match their split settings to 04 if you change them.

## Generated files

```text
house_price_prediction/
├── readme.md
├── notebooks/                         # the five notebooks above
├── house_prices_dataset/
│   ├── data_description.txt
│   ├── train/
│   │   ├── train.csv                  # untouched source
│   │   ├── training_data.csv          # written by 04
│   │   └── validation_data.csv        # written by 04
│   └── ground_truth/
│       ├── ground_truth_features.csv  # written by 04
│       ├── ground_truth.csv           # Id and known SalePrice, written by 04
│       ├── ground_truth_predictions.csv # written by 05
│       └── ground_truth_comparison.csv  # written by 05
├── artifacts/
│   ├── candidates.joblib              # training-only fitted candidates
│   ├── run_metadata.json              # settings, tuning and file fingerprints
│   ├── model_comparison.csv           # train/CV/validation scores, written by 04
│   └── training_testing_comparison.csv # combined scores and flags, written by 05
└── reports/                           # interactive HTML, viewable offline
    ├── training_performance.html      # written by 04
    ├── testing_evaluation.html        # written by 05
    └── model_comparison.html          # written by 05
```

Successful training overwrites its current outputs and removes outdated testing reports and tables. Run testing to regenerate them for the new candidates. Testing checks the saved source data, splits, models, scores and feature definitions for consistency and asks you to rerun training if they changed.

Each notebook includes the definitions it needs. If you edit a cleaning or feature rule, keep the corresponding definitions in 02–05 consistent, then rerun training and testing. No notebook imports code from another notebook.

## Interpret the comparison

With defaults, 20% of source rows is held out as labeled ground truth; 20% of the remainder is validation. This produces 934 training, 234 validation and 292 ground-truth rows. Target-dependent exploration uses training only. Stateless features use row-wise formulas. Learned imputation, scaling and encoding are fitted inside each CV fold using scikit-learn estimator composition. Every saved final candidate is fitted only on training rows, and testing evaluates those exact fits.

Dollar RMSE is the main comparison measure. MAE, RMSLE and R² provide context. Lower errors are better; higher R² is better.

- **Overfitting risk:** validation RMSE / training RMSE exceeds `overfit_ratio` (default 1.5). This is a heuristic; inspect CV spread and errors too.
- **Best observed:** lowest ground-truth RMSE, including numerical ties.
- **Underperforming:** ground-truth RMSE is at least `underperformance_ratio` times the best observed RMSE (default 1.25).
- **Comparison report:** grouped training, validation and testing RMSE and R², with every candidate's metrics and flags.

The reports compare the selected candidates across error measures and flag generalization gaps. Repeated tuning after inspecting ground truth reduces its independence. No global outlier deletion or automatic winner refit is applied.
