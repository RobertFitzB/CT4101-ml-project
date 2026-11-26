"""Pipeline and evaluation helpers for the model comparison notebook."""
import numpy as np
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(model, num_features, cat_features, sampler=None):
    """Scale, optionally resample, one-hot encode, then fit the model.

    The features are the same as in notebook 01. Resampling sits between scaling
    and encoding because SMOTE-NC needs scaled numbers (otherwise balance dominates
    its distances) and whole categories (it cannot work on one-hot columns).

    The imblearn Pipeline only runs the sampler during fit, so validation folds
    are never resampled and keep the real class ratio.
    """
    scale = ColumnTransformer(
        [("num", StandardScaler(), num_features)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    encode = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features)],
        remainder="passthrough",
    )
    return Pipeline([
        ("scale", scale),
        ("sampler", sampler if sampler is not None else "passthrough"),
        ("encode", encode),
        ("model", model),
    ])


def best_f1_threshold(y_true, proba):
    """Return the probability threshold that gives the highest F1."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last precision/recall pair has no threshold attached to it
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return thresholds[np.argmax(f1)]


def cross_fitted_decisions(y_true, oof_proba, folds):
    """Turn out-of-fold probabilities into yes/no decisions without self-selection.

    The threshold applied to each fold is chosen from the other four folds only,
    so no row helps to pick the threshold that is then used to judge it.

    Returns the decisions, the threshold used per fold and the F1 per fold.
    """
    y_true = np.asarray(y_true)
    decisions = np.zeros(len(y_true), dtype=int)
    thresholds, fold_f1 = [], []
    for _, val_idx in folds:
        other_idx = np.setdiff1d(np.arange(len(y_true)), val_idx)
        threshold = best_f1_threshold(y_true[other_idx], oof_proba[other_idx])
        decisions[val_idx] = oof_proba[val_idx] >= threshold
        thresholds.append(threshold)
        fold_f1.append(f1_score(y_true[val_idx], decisions[val_idx]))
    return decisions, np.array(thresholds), np.array(fold_f1)


def bootstrap_scores(y_true, values, metric, n_boot=1000, random_state=42):
    """Recompute a metric on bootstrap resamples of the rows, for confidence intervals.

    values can be probabilities (for PR-AUC) or yes/no decisions (for F1 or recall).
    The same random_state always draws the same resamples, so two models scored
    with it are compared on identical resamples, which makes their difference paired.
    """
    y_true, values = np.asarray(y_true), np.asarray(values)
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        scores.append(metric(y_true[idx], values[idx]))
    return np.array(scores)


def slice_metrics(y_true, decisions, groups):
    """Error rates for each group, used for the error analysis and fairness check."""
    # Categorical keeps an ordered grouping (such as age bands) in its natural order
    frame = pd.DataFrame({"y": np.asarray(y_true), "pred": decisions, "group": pd.Categorical(groups)})
    rows = []
    for group, part in frame.groupby("group", observed=True):
        tp = ((part["pred"] == 1) & (part["y"] == 1)).sum()
        fp = ((part["pred"] == 1) & (part["y"] == 0)).sum()
        fn = ((part["pred"] == 0) & (part["y"] == 1)).sum()
        tn = ((part["pred"] == 0) & (part["y"] == 0)).sum()
        rows.append({
            "group": group,
            "clients": len(part),
            "subscribers": tp + fn,
            "actual rate": (tp + fn) / len(part),
            "flagged rate": (tp + fp) / len(part),
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "recall": tp / (tp + fn) if tp + fn else np.nan,
            "false positive rate": fp / (fp + tn) if fp + tn else np.nan,
        })
    return pd.DataFrame(rows).set_index("group")
