"""Evaluation helpers: discrimination, calibration, and threshold metrics with
bootstrap confidence intervals. The positive class is always *death*."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)


def threshold_for_recall(y_true, y_score, target_recall: float = 0.80) -> float:
    """Highest threshold that still catches `target_recall` of deaths.

    Chosen on out-of-fold *training* predictions, then frozen for the test set.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    ok = np.where(recall[:-1] >= target_recall)[0]
    return float(thresholds[ok[-1]])


def point_metrics(y_true, y_score, threshold: float) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else np.nan
    precision = tp / (tp + fp) if tp + fp else np.nan
    return {
        "ROC AUC": roc_auc_score(y_true, y_score),
        "PR AUC": average_precision_score(y_true, y_score),
        "Brier": brier_score_loss(y_true, y_score),
        "Recall (deaths)": recall,
        "Precision (deaths)": precision,
        "F1 (deaths)": 2 * precision * recall / (precision + recall) if precision + recall else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
    }


def bootstrap_metrics(
    y_true, y_score, threshold: float, n_boot: int = 1000, seed: int = 42
) -> pd.DataFrame:
    """Point estimate and 95% percentile bootstrap CI for each metric."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    point = point_metrics(y_true, y_score, threshold)
    draws = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        draws.append(point_metrics(y_true[idx], y_score[idx], threshold))
    draws = pd.DataFrame(draws)
    return pd.DataFrame(
        {
            "estimate": pd.Series(point),
            "ci_low": draws.quantile(0.025),
            "ci_high": draws.quantile(0.975),
        }
    )


def paired_auc_difference(
    y_true, score_a, score_b, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float, float]:
    """ROC AUC(a) - AUC(b) on the same bootstrap resamples, with 95% CI."""
    y_true, score_a, score_b = map(np.asarray, (y_true, score_a, score_b))
    rng = np.random.default_rng(seed)
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        diffs.append(roc_auc_score(y_true[idx], score_a[idx]) - roc_auc_score(y_true[idx], score_b[idx]))
    est = roc_auc_score(y_true, score_a) - roc_auc_score(y_true, score_b)
    return est, float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
