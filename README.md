# Predicting Term-Deposit Subscription from Bank Telemarketing Data

CT4101 Machine Learning portfolio project, Robert Fitzgerald Bellone (10102634).

**Repository:** https://github.com/RobertFitzB/CT4101-ml-project

**Question:** Can a banking client's profile and contact history predict whether they will subscribe to a term deposit, using only information available before a call is made? (binary classification)

**Dataset:** Bank Marketing (bank-full), UCI Machine Learning Repository, CC BY 4.0.
Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. https://doi.org/10.24432/C5K306
See [data/README.md](data/README.md) for details. The data is downloaded by a script and not stored in the repository.

## Repository structure

```
notebooks/   numbered notebooks, run in order
  01_data_audit_and_baseline.ipynb   data audit, preprocessing, evaluation design, baselines (Milestone 2)
  02_model_comparison.ipynb          data balancing, model comparison, error analysis, fairness check (Milestone 3)
src/
  download_data.py   downloads bank-full.csv from UCI into data/raw/
  data.py            loading and feature preparation (drops duration and day)
  modelling.py       pipeline with optional resampling, threshold selection, per-group error rates
data/        dataset documentation (raw data is git-ignored)
reports/     figures and the final report
```

## Setup

Requires **Python 3.12** (tested with 3.12.6 on Windows 11).

```
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

## Running the work

From the repository root:

```
python src/download_data.py
jupyter notebook notebooks/01_data_audit_and_baseline.ipynb
```

Then run all cells from top to bottom. To run the whole notebook without opening it:

```
jupyter nbconvert --to notebook --execute --inplace notebooks/01_data_audit_and_baseline.ipynb
```

The model comparison notebook is run the same way, after the data has been downloaded. It takes roughly 15 minutes on a 12-core laptop, as every model is tuned under every balancing technique:

```
jupyter nbconvert --to notebook --execute --inplace notebooks/02_model_comparison.ipynb
```

Figures are saved to `reports/figures/` and the results table to `reports/model_comparison.csv`.

## Evaluation design (summary)

- `duration` is dropped because it is only known after the call ends.
- `day` (day of the month) is dropped. It has no plausible link to whether a client subscribes, and combined with `month` it identifies specific calendar dates that will not recur. The evidence is in section 6 of the notebook.
- A stratified 80/20 train/test split is made with `random_state=42`. The test set is kept aside for the final comparison only.
- Stratified 5-fold cross-validation is used on the training set, with all preprocessing inside a scikit-learn `Pipeline` to avoid leakage.
- The main metric is PR-AUC, with ROC-AUC and F1 also reported.
- Class weighting, random oversampling, random undersampling and SMOTE-NC are compared for every model. Resampling happens inside an `imblearn` `Pipeline`, so it only affects the training portion of each fold.
- The decision threshold is tuned inside cross-validation. The test set has not been used so far and is reserved for the final comparison at Milestone 4.

## Assistance declaration

See [ASSISTANCE.md](ASSISTANCE.md).
