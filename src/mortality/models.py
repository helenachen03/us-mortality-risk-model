"""Model definitions. Every model is a full sklearn Pipeline, so imputation and
scaling are fit inside each cross-validation fold (no train/test leakage)."""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import ALL_FEATURES, CATEGORICAL_FEATURES, DEMOGRAPHIC_FEATURES

SEED = 42


def _preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    cat = [f for f in features if f in CATEGORICAL_FEATURES]
    num = [f for f in features if f not in CATEGORICAL_FEATURES]
    num_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    transformers = [("num", Pipeline(num_steps), num)]
    if cat:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def _logreg(features: list[str]) -> Pipeline:
    return Pipeline(
        [
            ("prep", _preprocessor(features, scale=True)),
            ("clf", LogisticRegression(max_iter=2000)),
        ]
    )


def logistic_pipeline(features: list[str] = ALL_FEATURES) -> Pipeline:
    """Scaled, imputed logistic regression (the model used by the Streamlit demo)."""
    return _logreg(features)


def build_models() -> dict[str, dict]:
    """Return {name: {"features", "pipeline", "param_dist"}}.

    The first two are deliberately simple baselines: the question is not just
    "which model wins" but "how much do clinical measurements add beyond age?"
    """
    from xgboost import XGBClassifier  # imported lazily: needs OpenMP on macOS

    return {
        "Age only (logistic)": {
            "features": ["age"],
            "pipeline": _logreg(["age"]),
            "param_dist": None,
        },
        "Demographics (logistic)": {
            "features": DEMOGRAPHIC_FEATURES,
            "pipeline": _logreg(DEMOGRAPHIC_FEATURES),
            "param_dist": None,
        },
        "All features (logistic)": {
            "features": ALL_FEATURES,
            "pipeline": _logreg(ALL_FEATURES),
            "param_dist": {"clf__C": loguniform(1e-3, 10)},
        },
        "Random forest": {
            "features": ALL_FEATURES,
            "pipeline": Pipeline(
                [
                    ("prep", _preprocessor(ALL_FEATURES, scale=False)),
                    ("clf", RandomForestClassifier(n_estimators=400, n_jobs=1, random_state=SEED)),
                ]
            ),
            "param_dist": {
                "clf__max_depth": [None, 6, 10, 16],
                "clf__min_samples_leaf": randint(1, 40),
                "clf__max_features": ["sqrt", 0.3, 0.5],
            },
        },
        "XGBoost": {
            "features": ALL_FEATURES,
            "pipeline": Pipeline(
                [
                    ("prep", _preprocessor(ALL_FEATURES, scale=False)),
                    (
                        "clf",
                        XGBClassifier(
                            n_estimators=400,
                            eval_metric="logloss",
                            n_jobs=1,  # parallelism comes from the CV search instead
                            random_state=SEED,
                        ),
                    ),
                ]
            ),
            "param_dist": {
                "clf__learning_rate": loguniform(0.01, 0.2),
                "clf__max_depth": randint(2, 7),
                "clf__min_child_weight": randint(1, 20),
                "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.5, 0.5),
                "clf__reg_lambda": loguniform(0.1, 10),
            },
        },
    }
