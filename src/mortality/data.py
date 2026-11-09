"""Load the linked NHANES + National Death Index file and build the modelling table.

Each row is one NHANES adult participant (1999-2018 cycles) linked to the
2019 public-use NDI mortality file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# Outcome: death from any cause within this many months of the NHANES interview.
HORIZON_MONTHS = 120

# NHANES "refused" / "don't know" codes.
NONRESPONSE_CODES = {"DMDEDUC2": [7, 9], "SMQ020": [7, 9]}

# Human-readable names for the modelling features.
FEATURE_LABELS = {
    "age": "Age (years)",
    "female": "Female",
    "race_eth": "Race/ethnicity",
    "education": "Education level",
    "ever_smoker": "Smoked 100+ cigarettes",
    "bmi": "BMI",
    "waist_cm": "Waist circumference (cm)",
    "height_cm": "Height (cm)",
    "leg_len_cm": "Upper leg length (cm)",
    "arm_circ_cm": "Arm circumference (cm)",
    "sbp": "Systolic BP (mean of readings)",
    "dbp": "Diastolic BP (mean of readings)",
    "triglycerides": "Triglycerides (mg/dL)",
    "ldl": "LDL cholesterol (mg/dL)",
    "total_chol": "Total cholesterol (mg/dL)",
    "hdl": "HDL cholesterol (mg/dL)",
    "glucose": "Fasting glucose (mg/dL)",
    "hba1c": "HbA1c (%)",
}

RACE_LABELS = {
    1: "Mexican American",
    2: "Other Hispanic",
    3: "Non-Hispanic White",
    4: "Non-Hispanic Black",
    5: "Other / multiracial",
}

DEMOGRAPHIC_FEATURES = ["age", "female", "race_eth", "education", "ever_smoker"]
CLINICAL_FEATURES = [
    "bmi", "waist_cm", "height_cm", "leg_len_cm", "arm_circ_cm",
    "sbp", "dbp", "triglycerides", "ldl", "total_chol", "hdl", "glucose", "hba1c",
]
CATEGORICAL_FEATURES = ["race_eth"]
ALL_FEATURES = DEMOGRAPHIC_FEATURES + CLINICAL_FEATURES

# mg/dL per mmol/L
_HDL_FACTOR = 38.67
_GLUCOSE_FACTOR = 18.016


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA_DIR / "nhanes_ndi_df.csv")


def clean_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Harmonise units, recode non-response, and rename to readable columns.

    - Drops duplicate SI-unit lab columns (they carry the same information).
    - HDL and glucose were reported under different variable names across
      NHANES cycles; these are coalesced into one mg/dL column each.
    - Diastolic BP readings of ~0 are an NHANES artefact and are set to missing.
    - Individual BP readings are replaced by their mean.
    """
    df = raw.copy()
    for col, codes in NONRESPONSE_CODES.items():
        df.loc[df[col].isin(codes), col] = np.nan

    for col in ["BPXDI1", "BPXDI2", "BPXDI3"]:
        df.loc[df[col] < 1, col] = np.nan
    dbp = df[["BPXDI1", "BPXDI2", "BPXDI3"]].mean(axis=1)
    sbp = df[["BPXSY1", "BPXSY2", "BPXSY3"]].mean(axis=1)

    hdl = df["LBDHDD"].fillna(df["LBDHDDSI"] * _HDL_FACTOR)
    glucose = df["LBXGLU"].fillna(df["LBDGLUSI"] * _GLUCOSE_FACTOR)

    out = pd.DataFrame(
        {
            "seqn": df["SEQN"],
            "cycle_start": df["year"],
            "mortstat": df["MORTSTAT"],
            "followup_months": df["PERMTH_INT"],
            "age": df["RIDAGEYR"],
            "female": (df["RIAGENDR"] == 2).astype(int),
            "race_eth": df["RIDRETH1"],
            "education": df["DMDEDUC2"],
            "ever_smoker": df["SMQ020"].map({1: 1, 2: 0}),
            "bmi": df["BMXBMI"],
            "waist_cm": df["BMXWAIST"],
            "height_cm": df["BMXHT"],
            "leg_len_cm": df["BMXLEG"],
            "arm_circ_cm": df["BMXARMC"],
            "sbp": sbp,
            "dbp": dbp,
            "triglycerides": df["LBXTR"],
            "ldl": df["LBDLDL"],
            "total_chol": df["LBXTC"],
            "hdl": hdl,
            "glucose": glucose,
            "hba1c": df["LBXGH"],
        }
    )
    return out


def add_outcome(df: pd.DataFrame, horizon: int = HORIZON_MONTHS) -> pd.DataFrame:
    """Label death within `horizon` months; drop participants censored before it.

    died = 1  if confirmed death with follow-up < horizon
    died = 0  if follow-up reached the horizon alive
    dropped   if alive but follow-up ended before the horizon (outcome unknown)

    Only survey cycles in which *every* surviving participant was followed for
    the full horizon are kept. In later cycles, follow-up ends before the
    horizon, so only deaths can be labelled and the label would be biased
    (e.g. every kept 2011-2017 participant would be a death).
    """
    alive = df["mortstat"] == 0
    min_alive_followup = df.loc[alive].groupby("cycle_start")["followup_months"].min()
    complete_cycles = min_alive_followup[min_alive_followup >= horizon].index
    df = df[df["cycle_start"].isin(complete_cycles)]

    died = (df["mortstat"] == 1) & (df["followup_months"] < horizon)
    out = df.copy()
    out["died"] = died.astype(int)
    return out


def load_modelling_table(horizon: int = HORIZON_MONTHS) -> pd.DataFrame:
    return add_outcome(clean_features(load_raw()), horizon)
