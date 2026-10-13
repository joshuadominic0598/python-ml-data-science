# Customer-purchase classification: shared preparation, individual models

This folder has two common preparation notebooks and seven independent classifier notebooks. Run them manually in Jupyter or VS Code. Every classifier produces its own fitted-model artifacts and one offline HTML report. There is no runner, aggregate model comparison or model ranking.

From the repository root:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter notebook classification/notebooks
```

Use the repository's `.venv/bin/python` interpreter in VS Code, or a Jupyter Python kernel with the dependencies installed. The notebooks find the source data from the repository root, `classification/`, or `notebooks/`.

## Run order

1. [01_data_exploration.ipynb](notebooks/01_data_exploration.ipynb): common schema/quality checks, class balance, feature distributions, purchase-group differences and age/salary relationships. Target-dependent exploration uses the training part of the default stratified split.
2. [02_data_cleaning.ipynb](notebooks/02_data_cleaning.ipynb): common deterministic cleaning, source-row IDs, duplicate/invalid-record audit, and the shared cleaned CSV.
3. Run any classifier below with **Run All**. Each works in a fresh kernel after 02 and includes feature/preprocessing choices, training-only CV, final train/test evaluation, model interpretation and report generation.

| Notebook | Model-specific preparation and explanation |
|---|---|
| [03_logistic_regression_classifier.ipynb](notebooks/03_logistic_regression_classifier.ipynb) | Median imputation, optional polynomial features, scaling, regularization and coefficients |
| [03_knn_classifier.ipynb](notebooks/03_knn_classifier.ipynb) | Scaling, neighbor count, distance metric and distance weighting |
| [03_naive_bayes_classifier.ipynb](notebooks/03_naive_bayes_classifier.ipynb) | Gaussian class distributions, conditional independence, scaling and variance smoothing |
| [03_decision_tree_classifier.ipynb](notebooks/03_decision_tree_classifier.ipynb) | Unscaled inputs, entropy splits, depth/leaf controls and a tree-rule plot |
| [03_random_forest_classifier.ipynb](notebooks/03_random_forest_classifier.ipynb) | Unscaled inputs, bootstrap trees, feature subsampling and leaf/depth controls |
| [03_support_vector_classifier.ipynb](notebooks/03_support_vector_classifier.ipynb) | Scaling, the RBF kernel, C/gamma interaction and continuous decision margins |
| [03_xgboost_classifier.ipynb](notebooks/03_xgboost_classifier.ipynb) | Unscaled inputs, binary boosting, depth/learning-rate/tree-count controls |

Classifier notebooks do not depend on each other. For fitting alone, run **02, then the desired 03 notebook**; 01 is optional. XGBoost is needed only for its own notebook.

## Data and shared cleaning

The original CSV is preserved, byte for byte, at `customer_purchases_dataset/customer_details.csv`. `Age` and `EstimatedSalary` are predictors; `Purchased` is a binary label with **1 = purchased**, the positive class.

Cleaning parses numeric columns, excludes unknown/non-binary labels, marks nonpositive ages and negative salaries missing, excludes records with both predictors missing, and removes repeated equivalent labeled records. Remaining predictor gaps are preserved for training-fitted imputation. Valid extremes are retained. There is no global learned scaling or imputation.

The source has no customer identifier. `RowId` is the original one-based CSV row number and is excluded from fitting. The duplicate policy conservatively removes repeated equivalent labeled records; it cannot establish whether identical rows represent the same person. Different outcomes for the same age/salary profile are retained.

The current source contains **400 rows**, including **33 repeated labeled records**. Cleaning retains **367 rows**, with 229 non-purchases and 138 purchases. Default classifier splits contain **275 training and 92 testing rows**.

Common outputs are the cleaned CSV, `artifacts/common/cleaning_metadata.json`, `data_cleaning_audit.csv`, and `excluded_rows.csv`. Model notebooks verify source/cleaned-data fingerprints before fitting. If the source or cleaning rules change, rerun 02 and the classifier notebooks whose reports you want to refresh.

## Configure one classifier

Each 03 notebook has editable settings near the top:

```python
config = {
    "random_state": 0,
    "test_size": 0.25,
    "cv_folds": 5,
    "tune": True,
    "scoring": "recall",
}
```

Change the estimator or `PARAM_GRID` in that notebook to adjust its hyperparameters. Set `tune=False` to keep the explicitly defined estimator settings; training-only CV still reports their score. Supported tuning objectives are recall, F1, precision, ROC AUC, average precision, balanced accuracy and accuracy. Logistic Regression also exposes `POLYNOMIAL_DEGREE`, defaulting to 1.

Each model uses a stratified train/test split. All learned imputation, scaling and feature expansion are fitted inside the training CV folds. Tuning selects parameters within that classifier family and refits on training rows only; it never uses the test set for parameter search or early stopping. Reruns overwrite only that classifier's artifact folder and report.

## Files and reports

```text
classification/
├── readme.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_logistic_regression_classifier.ipynb
│   ├── 03_knn_classifier.ipynb
│   ├── 03_naive_bayes_classifier.ipynb
│   ├── 03_decision_tree_classifier.ipynb
│   ├── 03_random_forest_classifier.ipynb
│   ├── 03_support_vector_classifier.ipynb
│   └── 03_xgboost_classifier.ipynb
├── customer_purchases_dataset/
│   ├── customer_details.csv            # unchanged original source
│   └── cleaned_data.csv                # shared deterministic preparation
├── artifacts/
│   ├── common/                         # cleaning metadata, audit and exclusions
│   └── <classifier_name>/
│       ├── model.joblib                # classifier with fitted preprocessing
│       ├── run_metadata.json           # settings, parameters, data hashes and split IDs
│       ├── training_data.csv
│       ├── testing_data.csv
│       ├── performance.csv             # this model's train/test metrics
│       ├── test_predictions.csv        # classes, continuous scores and error types
│       ├── classification_report.csv
│       ├── confusion_matrix.csv
│       ├── cv_results.csv              # this model's training-CV search or fold scores
│       └── model_details.csv
└── reports/
    └── <classifier_name>.html          # one report for each model run
```

Reports embed Plotly for offline viewing. Each contains the model explanation, actual fitted hyperparameters, CV results, train/test metrics, per-class scores, misclassified customers, confusion matrix, ROC/precision–recall curves and an age/salary decision-region plot. A bounded 120×120 grid replaces the original extremely large decision-boundary mesh. Decision Tree also displays its learned rules in its notebook without requiring Graphviz.

## Read one model's performance

The original concern about missed purchasers is preserved: **recall** measures the share of purchasers captured. **Precision** measures how many predicted purchasers actually purchased, and **F1** summarizes their balance. Accuracy and balanced accuracy provide additional context. False negatives and false positives are explicitly identified by source RowId.

ROC AUC and average precision use continuous purchase probabilities, or SVC decision margins, rather than hard class labels. SVC margins are not probabilities, so its report omits probability log loss. Other classifiers report probability log loss alongside the classification metrics.

Inspect train/test gaps, CV variability and the small held-out sample before interpreting results. Repeated changes after inspecting test performance reduce its independence. Reports describe each classifier separately; no cross-model comparison or winner is generated.
