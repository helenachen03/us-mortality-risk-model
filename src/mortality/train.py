"""End-to-end experiment: tune, evaluate on a held-out test set, save results.

Run from the repo root:
    python -m mortality.train            (with src/ on PYTHONPATH, or after `pip install -e .`)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from .data import ALL_FEATURES, FEATURE_LABELS, RACE_LABELS, load_modelling_table
from .evaluate import bootstrap_metrics, paired_auc_difference, threshold_for_recall
from .models import SEED, build_models

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures"
RES_DIR = ROOT / "results"

# Categorical slots, fixed order; colour follows the model, never its rank.
COLORS = {
    "Age only (logistic)": "#8a8984",
    "Demographics (logistic)": "#eda100",
    "All features (logistic)": "#1baf7a",
    "Random forest": "#eb6834",
    "XGBoost": "#2a78d6",
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "lines.linewidth": 2,
    }
)


@dataclass
class Experiment:
    df: pd.DataFrame
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    fitted: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    oof_scores: dict = field(default_factory=dict)
    test_scores: dict = field(default_factory=dict)
    thresholds: dict = field(default_factory=dict)


def split(df: pd.DataFrame, test_size: float = 0.2) -> Experiment:
    X = df[ALL_FEATURES]
    y = df["died"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )
    return Experiment(df, X_tr, X_te, y_tr, y_te)


def fit_all(exp: Experiment, n_iter: int = 30, verbose: bool = True) -> Experiment:
    """Tune on training folds only (scored by PR AUC, which focuses on the
    minority death class), then refit on all training data."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, spec in build_models().items():
        feats = spec["features"]
        X_tr = exp.X_train[feats]
        if spec["param_dist"]:
            search = RandomizedSearchCV(
                spec["pipeline"],
                spec["param_dist"],
                n_iter=n_iter,
                scoring="average_precision",
                cv=cv,
                n_jobs=-1,
                random_state=SEED,
            )
            search.fit(X_tr, exp.y_train)
            model = search.best_estimator_
            exp.best_params[name] = {
                k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                for k, v in search.best_params_.items()
            }
        else:
            model = clone(spec["pipeline"]).fit(X_tr, exp.y_train)
            exp.best_params[name] = {}

        # Out-of-fold training predictions are used only to pick a threshold.
        oof = cross_val_predict(
            clone(model), X_tr, exp.y_train, cv=cv, method="predict_proba", n_jobs=-1
        )[:, 1]
        exp.oof_scores[name] = oof
        exp.thresholds[name] = threshold_for_recall(exp.y_train, oof, 0.80)
        exp.test_scores[name] = model.predict_proba(exp.X_test[feats])[:, 1]
        exp.fitted[name] = model
        if verbose:
            print(f"fitted {name:<26} {exp.best_params[name]}")
    return exp


def results_table(exp: Experiment, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for name, score in exp.test_scores.items():
        b = bootstrap_metrics(exp.y_test, score, exp.thresholds[name], n_boot=n_boot)
        row = {"model": name}
        for metric, r in b.iterrows():
            row[metric] = r["estimate"]
            row[f"{metric} CI"] = f"{r['ci_low']:.3f}–{r['ci_high']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def baseline_gains(exp: Experiment, reference: str = "Age only (logistic)") -> pd.DataFrame:
    rows = []
    for name, score in exp.test_scores.items():
        if name == reference:
            continue
        est, lo, hi = paired_auc_difference(exp.y_test, score, exp.test_scores[reference])
        rows.append({"model": name, "ΔROC AUC vs age only": est, "95% CI": f"{lo:+.3f} to {hi:+.3f}"})
    return pd.DataFrame(rows).set_index("model")


# ---------------------------------------------------------------- figures
def _legend(ax, **kw):
    ax.legend(loc=kw.pop("loc", "lower right"), fontsize=9, labelcolor=INK_2, **kw)


def plot_roc_pr(exp: Experiment, path: Path | None = None):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, s in exp.test_scores.items():
        fpr, tpr, _ = roc_curve(exp.y_test, s)
        a1.plot(fpr, tpr, color=COLORS[name], label=name)
        p, r, _ = precision_recall_curve(exp.y_test, s)
        a2.plot(r, p, color=COLORS[name], label=name)
    a1.plot([0, 1], [0, 1], ls="--", lw=1, color=GRID)
    a1.set(xlabel="False positive rate", ylabel="True positive rate (deaths caught)",
           title="ROC curve, held-out test set")
    prevalence = exp.y_test.mean()
    a2.axhline(prevalence, ls="--", lw=1, color=GRID)
    a2.text(0.99, prevalence + 0.015, f"base rate {prevalence:.0%}", ha="right", color=INK_2, fontsize=8)
    a2.set(xlabel="Recall (deaths caught)", ylabel="Precision", ylim=(0, 1.02),
           title="Precision–recall curve, held-out test set")
    _legend(a1)
    _legend(a2, loc="upper right")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def plot_calibration(exp: Experiment, names=None, path: Path | None = None):
    names = names or list(exp.test_scores)
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=GRID, label="Perfect calibration")
    for name in names:
        frac, mean_pred = calibration_curve(exp.y_test, exp.test_scores[name], n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, marker="o", ms=5, color=COLORS[name], label=name)
    ax.set(xlabel="Predicted 10-year risk", ylabel="Observed 10-year mortality",
           title="Calibration (deciles of predicted risk)", xlim=(0, 1), ylim=(0, 1))
    _legend(ax, loc="upper left")
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def plot_survival_by_risk(exp: Experiment, name: str = "XGBoost", path: Path | None = None):
    """Kaplan–Meier curves over the *full* follow-up, grouped by predicted-risk
    quintile on the test set. Uses follow-up time the classifier never saw."""
    from lifelines import KaplanMeierFitter

    test = exp.df.loc[exp.X_test.index, ["followup_months", "mortstat"]].copy()
    test["risk"] = exp.test_scores[name]
    test["group"] = pd.qcut(test["risk"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    ramp = ["#c5dbf5", "#8db8ec", "#5596e0", "#2a78d6", "#14498a"]  # one hue, light→dark
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for color, (g, d) in zip(ramp, test.groupby("group", observed=True)):
        kmf = KaplanMeierFitter().fit(d["followup_months"] / 12, d["mortstat"], label=str(g))
        kmf.plot_survival_function(ax=ax, ci_show=False, color=color, lw=2)
        ax.text(kmf.timeline[-1] + 0.2, kmf.survival_function_.iloc[-1, 0], str(g),
                va="center", fontsize=8, color=INK_2)
    ax.axvline(10, ls="--", lw=1, color=GRID)
    ax.set(xlabel="Years since NHANES interview", ylabel="Survival probability",
           title=f"Survival by predicted-risk quintile ({name})", ylim=(0, 1.02))
    _legend(ax, loc="lower left", title="Risk quintile", title_fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path)
    return fig


def shap_summary(exp: Experiment, name: str = "XGBoost", path: Path | None = None, n: int = 2000):
    import shap

    model = exp.fitted[name]
    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    X = exp.X_test.sample(min(n, len(exp.X_test)), random_state=SEED)
    Xt = pd.DataFrame(prep.transform(X), columns=prep.get_feature_names_out(), index=X.index)
    labels = {c: FEATURE_LABELS.get(c, c) for c in Xt.columns}
    labels.update({c: f"Race: {RACE_LABELS.get(int(float(c.split('_')[-1])), c)}"
                   for c in Xt.columns if c.startswith("race_eth_")})
    labels.update({c: f"Missing: {FEATURE_LABELS.get(c.replace('missingindicator_', ''), c)}"
                   for c in Xt.columns if c.startswith("missingindicator_")})
    Xt = Xt.rename(columns=labels)
    sv = shap.TreeExplainer(clf).shap_values(Xt)
    plt.figure()
    shap.summary_plot(sv, Xt, max_display=12, show=False, plot_size=(8, 5.2))
    fig = plt.gcf()
    fig.axes[0].set_title(f"What drives predicted risk ({name}, SHAP)", loc="left")
    fig.axes[0].set_xlabel("SHAP value (impact on log-odds of death)")
    if path:
        fig.savefig(path)
    return fig, pd.Series(np.abs(sv).mean(0), index=Xt.columns).sort_values(ascending=False)


def main(n_iter: int = 30, n_boot: int = 1000):
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    df = load_modelling_table()
    print(f"{len(df):,} participants, {df['died'].mean():.1%} died within 10 years")
    exp = fit_all(split(df), n_iter=n_iter)

    table = results_table(exp, n_boot=n_boot)
    gains = baseline_gains(exp)
    table.to_csv(RES_DIR / "test_metrics.csv")
    gains.to_csv(RES_DIR / "gain_over_age_baseline.csv")
    (RES_DIR / "best_params.json").write_text(json.dumps(exp.best_params, indent=2))

    plot_roc_pr(exp, FIG_DIR / "roc_pr_curves.png")
    plot_calibration(exp, ["Age only (logistic)", "All features (logistic)", "Random forest", "XGBoost"],
                     FIG_DIR / "calibration.png")
    plot_survival_by_risk(exp, path=FIG_DIR / "survival_by_risk_quintile.png")
    _, importance = shap_summary(exp, path=FIG_DIR / "shap_summary.png")
    importance.to_csv(RES_DIR / "shap_importance.csv", header=["mean_abs_shap"])

    joblib.dump(exp.fitted["XGBoost"], RES_DIR / "xgboost_pipeline.joblib")

    cols = ["ROC AUC", "PR AUC", "Brier", "Recall (deaths)", "Precision (deaths)", "Specificity"]
    print(table[cols].round(3).to_string())
    print(gains.round(3).to_string())
    return exp, table, gains


if __name__ == "__main__":
    main()
