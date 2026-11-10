"""Interactive 10-year mortality risk demo.

Run from the repo root:
    streamlit run app/streamlit_app.py

Uses the regularised logistic regression from the study. It matches XGBoost on
held-out ROC AUC (0.889 vs 0.892, difference not significant) and its
per-feature contributions are exact, so every prediction can be explained.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mortality import data  # noqa: E402
from mortality.models import logistic_pipeline  # noqa: E402
from mortality.train import split  # noqa: E402

RAISES, LOWERS = "#eb6834", "#2a78d6"

EDUCATION = {
    1: "Less than 9th grade",
    2: "9th–11th grade",
    3: "High school / GED",
    4: "Some college or associate degree",
    5: "College graduate or above",
}

st.set_page_config(page_title="10-Year Mortality Risk", page_icon="🩺", layout="wide")


# --------------------------------------------------------------------- model
@st.cache_resource(show_spinner="Training model on NHANES data…")
def load_model():
    df = data.load_modelling_table()
    exp = split(df)
    params = json.loads((ROOT / "results" / "best_params.json").read_text())["All features (logistic)"]
    model = logistic_pipeline().set_params(**{f"clf__{k}": v for k, v in params.items()})
    model.fit(exp.X_train, exp.y_train)

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    Xt_train = prep.transform(exp.X_train)
    names = prep.get_feature_names_out()
    reference = pd.DataFrame(
        {
            "age": exp.X_train["age"],
            "female": exp.X_train["female"],
            "risk": model.predict_proba(exp.X_train)[:, 1],
        }
    )
    return model, np.asarray(Xt_train).mean(axis=0), names, clf.coef_[0], reference


def describe(row: pd.DataFrame) -> dict[str, str]:
    """Bar label for each model input, showing the user's own value."""
    r = row.iloc[0]

    def num(v, fmt, unit=""):
        return "not given" if pd.isna(v) else f"{v:{fmt}}{unit}"

    return {
        "age": f"Age: {r.age:.0f}",
        "female": "Sex: female" if r.female else "Sex: male",
        "race_eth": f"Race/ethnicity: {data.RACE_LABELS[int(r.race_eth)]}",
        "education": f"Education: {EDUCATION[int(r.education)]}",
        "ever_smoker": "Ever smoked: yes" if r.ever_smoker else "Ever smoked: no",
        "bmi": f"BMI: {r.bmi:.1f}",
        "waist_cm": f"Waist: {r.waist_cm:.0f} cm",
        "height_cm": f"Height: {r.height_cm:.0f} cm",
        "leg_len_cm": f"Upper-leg length: {r.leg_len_cm:.0f} cm",
        "arm_circ_cm": f"Arm circumference: {r.arm_circ_cm:.1f} cm",
        "sbp": f"Systolic BP: {r.sbp:.0f}",
        "dbp": f"Diastolic BP: {r.dbp:.0f}",
        "triglycerides": f"Triglycerides: {num(r.triglycerides, '.0f')}",
        "ldl": f"LDL: {num(r.ldl, '.0f')}",
        "total_chol": f"Total cholesterol: {num(r.total_chol, '.0f')}",
        "hdl": f"HDL: {num(r.hdl, '.0f')}",
        "glucose": f"Fasting glucose: {num(r.glucose, '.0f')}",
        "hba1c": f"HbA1c: {num(r.hba1c, '.1f', '%')}",
    }


def base_feature(col: str) -> str:
    col = col.replace("missingindicator_", "")
    return "race_eth" if col.startswith("race_eth_") else col


def explain(model, train_mean, names, coef, row: pd.DataFrame) -> pd.DataFrame:
    """Exact log-odds contribution of each input vs. the average participant."""
    xt = np.asarray(model.named_steps["prep"].transform(row))[0]
    contrib = coef * (xt - train_mean)
    labels = describe(row)
    out = (
        pd.DataFrame({"feature": [labels[base_feature(n)] for n in names], "contribution": contrib})
        .groupby("feature", as_index=False)["contribution"].sum()
    )
    out["direction"] = np.where(out["contribution"] >= 0, "Raises risk", "Lowers risk")
    out["odds"] = np.exp(out["contribution"])
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index)


# --------------------------------------------------------------------- inputs
def optional_number(label, lo, hi, default, step, help=None, key=None):
    known = st.checkbox(f"I know my {label.split(' (')[0].lower()}", value=True, key=f"{key}_known")
    if not known:
        st.caption(
            "Left blank. Note: in the survey, people with missing lab results had higher "
            "mortality, so a blank value nudges the estimate up."
        )
        return np.nan
    return st.number_input(label, lo, hi, default, step, help=help, key=key)


def sidebar_inputs() -> pd.DataFrame:
    sb = st.sidebar
    sb.header("About you")
    age = sb.slider("Age", 18, 85, 55)
    sex = sb.radio("Sex", ["Male", "Female"], horizontal=True)
    race = sb.selectbox("Race/ethnicity", list(data.RACE_LABELS.values()), index=2)
    edu = sb.selectbox("Education", list(EDUCATION.values()), index=2)
    smoker = sb.radio("Smoked at least 100 cigarettes in your life?", ["No", "Yes"], horizontal=True)

    sb.header("Body measurements")
    height = sb.number_input("Height (cm)", 130.0, 210.0, 170.0, 0.5)
    weight = sb.number_input("Weight (kg)", 30.0, 250.0, 78.0, 0.5)
    bmi = weight / (height / 100) ** 2
    sb.caption(f"BMI: **{bmi:.1f}**")
    waist = sb.number_input("Waist circumference (cm)", 55.0, 180.0, 97.0, 0.5)
    arm = sb.number_input("Upper-arm circumference (cm)", 15.0, 60.0, 32.5, 0.5,
                          help="Measured midway between shoulder and elbow.")
    leg = sb.number_input("Upper-leg length (cm)", 23.0, 56.0, 39.0, 0.5,
                          help="Hip crease to top of the knee, seated.")

    sb.header("Blood pressure")
    sbp = sb.number_input("Systolic (top number, mmHg)", 70, 260, 122)
    dbp = sb.number_input("Diastolic (bottom number, mmHg)", 30, 140, 70)

    with sb.expander("Blood tests (optional)", expanded=False):
        tc = optional_number("Total cholesterol (mg/dL)", 70.0, 700.0, 191.0, 1.0, key="tc")
        ldl = optional_number("LDL cholesterol (mg/dL)", 10.0, 600.0, 112.0, 1.0, key="ldl")
        hdl = optional_number("HDL cholesterol (mg/dL)", 6.0, 220.0, 52.0, 1.0, key="hdl")
        tg = optional_number("Triglycerides (mg/dL)", 10.0, 4000.0, 106.0, 1.0, key="tg")
        glu = optional_number("Fasting glucose (mg/dL)", 20.0, 600.0, 99.0, 1.0, key="glu")
        a1c = optional_number("HbA1c (%)", 3.5, 18.0, 5.5, 0.1, key="a1c")

    race_code = {v: k for k, v in data.RACE_LABELS.items()}[race]
    edu_code = {v: k for k, v in EDUCATION.items()}[edu]
    row = {
        "age": age, "female": int(sex == "Female"), "race_eth": race_code,
        "education": edu_code, "ever_smoker": int(smoker == "Yes"),
        "bmi": bmi, "waist_cm": waist, "height_cm": height, "leg_len_cm": leg,
        "arm_circ_cm": arm, "sbp": sbp, "dbp": dbp, "triglycerides": tg,
        "ldl": ldl, "total_chol": tc, "hdl": hdl, "glucose": glu, "hba1c": a1c,
    }
    return pd.DataFrame([row])[data.ALL_FEATURES].astype(float)


# --------------------------------------------------------------------- page
model, train_mean, names, coef, reference = load_model()
row = sidebar_inputs()
risk = float(model.predict_proba(row)[:, 1][0])

age, female = row.at[0, "age"], row.at[0, "female"]
peers = reference[(reference["age"].between(age - 5, age + 5)) & (reference["female"] == female)]
peer_risk = float(peers["risk"].mean())
peer_pct = float((peers["risk"] < risk).mean() * 100)

st.title("10-year mortality risk from a routine health exam")
st.markdown(
    "An interactive demo of the model from "
    "[this study](https://github.com/helenachen03/us-mortality-risk-model): "
    "a logistic regression trained on 9,900 NHANES participants (1999–2008) "
    "linked to the National Death Index. Adjust the inputs in the sidebar."
)

c1, c2, c3 = st.columns(3)
c1.metric("Predicted 10-year risk of death", f"{risk:.1%}")
c2.metric(
    f"Average for {'women' if female else 'men'} aged {max(18, int(age) - 5)}–{min(85, int(age) + 5)}",
    f"{peer_risk:.1%}",
    help=f"Mean predicted risk among {len(peers):,} similar participants in the training data.",
)
c3.metric(
    "Compared with same age & sex",
    f"higher than {peer_pct:.0f}%",
    help="Share of same-age, same-sex participants with a lower predicted risk.",
)

st.subheader("What's driving this estimate")
st.caption(
    "Each bar is that input's exact contribution to the model's log-odds of death, "
    "relative to the average study participant. Hover over a bar to see it as an odds multiplier."
)
contrib = explain(model, train_mean, names, coef, row)
swatch = '<span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:{}"></span>'
st.markdown(
    f"{swatch.format(RAISES)} Raises risk &nbsp;&nbsp; {swatch.format(LOWERS)} Lowers risk",
    unsafe_allow_html=True,
)
top = contrib.head(10).copy()
chart = (
    alt.Chart(top)
    .mark_bar(cornerRadiusEnd=4, height=18)
    .encode(
        x=alt.X("contribution:Q", title="Effect on log-odds of death"),
        y=alt.Y("feature:N", sort=list(top["feature"]), title=None,
                axis=alt.Axis(labelLimit=320, labelFontSize=12)),
        color=alt.Color(
            "direction:N",
            scale=alt.Scale(domain=["Raises risk", "Lowers risk"], range=[RAISES, LOWERS]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("feature:N", title="Input"),
            alt.Tooltip("direction:N", title="Effect"),
            alt.Tooltip("odds:Q", title="Odds multiplier", format=".2f"),
        ],
    )
    .properties(height=alt.Step(28))
)
rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8984").encode(x="x:Q")
st.altair_chart(chart + rule, width="stretch")

with st.expander("How to read this, and its limits"):
    st.markdown(
        """
- **Age dominates.** In the study, age alone reached ROC AUC 0.865, and all the other
  inputs together raised it to 0.889. Changing a lab value moves the estimate far less
  than changing age.
- **Associations, not causes.** For example, low cholesterol and low BMI *raise* predicted
  risk, because serious illness and frailty lower both. Changing these numbers yourself
  would not change your risk the way the model suggests.
- **Data from 1999–2008, without survey weights.** The estimate reflects that sample,
  not today's U.S. population.
- **Blind spot for younger adults.** The model catches only about 1 in 10 deaths under
  age 55, which are often from accidents or other sudden causes.
"""
    )

st.warning(
    "This is a portfolio demonstration of a statistical model, **not medical advice** "
    "and not a clinical tool. Talk to a healthcare professional about your health.",
    icon="⚠️",
)
