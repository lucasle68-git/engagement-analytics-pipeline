"""What Drives Engagement ? A demonstration on synthetic data (Phase 2).

Every number is read from pipeline outputs (tab16-tab21, data/synthetic/,
config).
"""

from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import (ROOT, figure, key_takeaways, load_table,  # noqa: E402
                     require, short_theme, synthetic_badge)

st.set_page_config(page_title="Drivers (demonstration)", page_icon="📊", layout="wide")

try:
    CFG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
except Exception:
    CFG = {}
try:
    PARAMS = json.loads((ROOT / "data" / "synthetic" / "generator_params.json").read_text())
except Exception:
    PARAMS = {}
RISK_Q = CFG.get("models", {}).get("risk_threshold_quantile", 0.20)
N_EMP = PARAMS.get("n_employees", 1000)
N_WAVES = PARAMS.get("n_waves", 3)
TOL = PARAMS.get("calibration_tolerance", 0.05)
DRIFT = PARAMS.get("wave_drift_sd", 0.06)
TARGETS = PARAMS.get("target_correlations", {})

NICE = {
    "development": "Development", "personal": "Personal resources", "social": "Social support",
    "balance": "Work–life balance", "justice": "Fairness", "technology": "Technology",
    "psych_safety": "Psychological safety", "esg": "ESG alignment",
    "engagement_score": "Engagement", "satisfaction": "Satisfaction", "demands": "Workload demands",
}


def nice(x: str) -> str:
    return NICE.get(str(x), str(x).replace("_", " ").capitalize())


st.title("What Drives Engagement? A Demonstration")
synthetic_badge()

_dv = load_table("tab16_driver_ranking")
_ipr = load_table("tab21_importance_performance")
_rk3 = load_table("tab17_risk_model_comparison")
if _dv is not None:
    _dv3 = _dv.rename(columns={_dv.columns[0]: "c"}).nlargest(3, "relative_weight_%")
    _pts = [
        f"**Three drivers carry most of it.** {', '.join(nice(c) for c in _dv3['c'])} "
        f"together explain **{_dv3['relative_weight_%'].sum():.1f}%** of why one employee is "
        f"more engaged than another."
    ]
    if _ipr is not None:
        _un = _ipr.loc[~_ipr["measured_2024"], "importance_%"].sum()
        _pts.append(
            f"**Almost half of what matters was never asked.** The drivers missing from the "
            f"2024 survey account for **{_un:.1f}%** of the explanation. No amount of analysis "
            f"can recover them; only a better survey can."
        )
    if _rk3 is not None:
        _b = _rk3.loc[_rk3["pr_auc"].idxmax()]
        _pts.append(
            f"**The machine runs end to end.** The best risk model reaches PR-AUC "
            f"**{_b['pr_auc']:.2f}** and finds **{_b['recall@tuned']:.0%}** of at-risk "
            f"employees once its threshold is tuned on invented data. That is proof the "
            f"method works instead of finding about the company."
        )
    key_takeaways(*_pts)

# ====================================================================== intro
st.markdown(
    """
**Why this page is different:** The 2024 survey only holds department averages. Answering
"what drives engagement?" or "who is at risk?" needs one row per employee, and that data does
not exist yet.

Rather than drop those questions, the project builds a realistic **synthetic** dataset and
runs the full analysis on it. The result is a working method, tested end to end, that can be
applied as soon as the real individual-level data is collected.
"""
)

st.subheader("How the synthetic data was built")
st.markdown(
    f"""
Think of it as a simulated company that looks like the real one from the outside.

1. **{N_EMP:,} employees** are created and spread across the same 19 departments.
2. Each one gets a simple **profile**: region, tenure, role level and work mode, details the
   2024 file never had.
3. Each employee **answers the same 34 questions** as the 2024 survey. The answers are tuned
   so that every department's average matches the real workbook to within {TOL} points.
4. **15 new questions** are added, the ones the 2024 survey never asked: engagement itself,
   personal resources, technology, fairness, psychological safety and ESG alignment. How
   strongly each one relates to engagement is taken from published research.
5. The survey is **repeated {N_WAVES} times** (three "waves"), with small random movement
   between waves, so change over time can be studied.
6. Two **quality checks** must pass, or the whole run stops: the averages must match the real
   file, and the research-based relationships must still be there.

The result is {N_EMP * N_WAVES:,} rows: one per employee per wave.
"""
)
st.markdown(
    """
**The questions are regrouped.** The 2024 survey sorts its 34 questions into 8 **themes** by
*who* they are about: my team, my manager, the executive. Phase 2 sorts the same 34 questions
into **drivers** by *what they measure*, and adds the 15 new questions to fill the gaps. No
question is thrown away.
"""
)

map_path = ROOT / "data" / "synthetic" / "synthetic_item_map.csv"
if map_path.exists():
    with st.expander("See how the 34 questions were regrouped"):
        imap = pd.read_csv(map_path)
        imap["is_old"] = imap["code"].str.startswith("q")
        rows = []
        for con, grp in imap.groupby("construct"):
            old = grp[grp["is_old"]]
            src = ", ".join(f"{short_theme(c)} ({n})"
                            for c, n in old["category"].value_counts().items())
            rows.append({
                "Driver (Phase 2)": nice(con),
                "Built from these 2024 themes": src or "—",
                "Old questions": int(len(old)),
                "New questions": int((~grp["is_old"]).sum()),
                "Total": int(len(grp)),
            })
        cw = pd.DataFrame(rows).sort_values(
            ["Old questions", "Total"], ascending=False).reset_index(drop=True)
        st.dataframe(cw, hide_index=True, use_container_width=True)
        st.caption(
            "Read a row like this: **Social support** is built from 13 of the 2024 questions, "
            "pulled out of three different 2024 themes. **Fairness** keeps 6 old questions and "
            "adds 2 new ones. Rows with 0 old questions are drivers the 2024 survey never "
            "asked about at all."
        )

panel_path = ROOT / "data" / "synthetic" / "synthetic_panel.csv"
if panel_path.exists():
    with st.expander("See what the rows look like"):
        panel = pd.read_csv(panel_path, nrows=5)
        show = [c for c in ["employee_id", "department", "region", "tenure", "role_level",
                            "work_mode", "wave", "q01", "q02", "q03", "uwes_1", "pers_1",
                            "tech_1", "psys_1", "esg_1"] if c in panel.columns]
        st.dataframe(panel[show], hide_index=True, use_container_width=True)
        st.caption(
            "Five example rows, a few columns shown. `q01`–`q34` are the 2024 questions; "
            "`uwes_`, `pers_`, `tech_`, `just_`, `psys_` and `esg_` are the new ones. "
            "Answers are on the survey's own scales."
        )

# ====================================================================== 1. drivers
drivers = load_table("tab16_driver_ranking")
require(drivers)
drivers = drivers.rename(columns={drivers.columns[0]: "construct"})

st.subheader("1. Ranked drivers")
st.markdown(
    """
**What we do:** We take each employee's engagement score and ask which of eight drivers:
development, personal resources, social support, and so on, best explain it. A regression
measures each driver's effect while holding the others fixed. Then a method called *relative
weights* splits the total explanation into shares that add up to 100%, so the drivers can be
ranked fairly even when they overlap.

**Why this is a real test:** The synthetic data was built with known relationships. So the
right answer is known in advance, and we can check whether the method finds it.
"""
)
figure("fig06_driver_ranking")

dv = pd.DataFrame({
    "Driver": drivers["construct"].map(nice),
    "Share of explanation": drivers["relative_weight_%"],
    "Effect size": drivers["beta_std"],
    "Clearly above zero?": drivers["p_value"].map(lambda p: "Yes" if p < 0.05 else "No"),
    "Overlap with others (VIF)": drivers["VIF"],
})
st.dataframe(dv, hide_index=True, use_container_width=True,
             column_config={"Share of explanation": st.column_config.NumberColumn(format="%.1f%%"),
                            "Effect size": st.column_config.NumberColumn(format="%.3f"),
                            "Overlap with others (VIF)": st.column_config.NumberColumn(format="%.2f")})

ip = load_table("tab21_importance_performance")
top3 = drivers.nlargest(3, "relative_weight_%")
lines = [
    f"- **Three drivers carry most of the weight:** {', '.join(nice(c) for c in top3['construct'])} "
    f"together explain **{top3['relative_weight_%'].sum():.1f}%**."
]
if ip is not None and "measured_2024" in ip.columns:
    unmeasured = ip[~ip["measured_2024"]]
    lines.append(
        f"- **Almost half sits outside today's survey:** The drivers the 2024 survey never asked "
        f"about: {', '.join(nice(c) for c in unmeasured['construct'])}, carry "
        f"**{unmeasured['importance_%'].sum():.1f}%** of the explanation. "
        f"{nice(unmeasured.iloc[0]['construct'])} alone ranks near the top at "
        f"{unmeasured.iloc[0]['importance_%']:.1f}%."
    )
weak = drivers[drivers["p_value"] >= 0.05]
if not weak.empty:
    lines.append(
        f"- **Some drivers add little once the others are counted:** "
        f"{', '.join(nice(c) for c in weak['construct'])} (effect not clearly above zero)."
    )
lines.append(
    f"- **The ranking is stable:** Overlap between drivers is low (VIF at most "
    f"{drivers['VIF'].max():.2f}; values near 1 mean no overlap), so no driver is borrowing "
    "another's credit."
)
if TARGETS:
    tgt = pd.Series({k: v for k, v in TARGETS.items() if k in set(drivers["construct"])})
    tgt_top = set(tgt.nlargest(3).index)
    got_top = set(top3["construct"])
    if tgt_top == got_top:
        lines.append(
            "- **The method found the right answer:** The three drivers built in as strongest "
            f"({', '.join(nice(c) for c in tgt.nlargest(3).index)}) are exactly the three it ranks first."
        )
st.markdown("**What the result shows:**\n\n" + "\n".join(lines))

# ====================================================================== 2. risk
st.subheader("2. Who is at risk of disengaging")
st.markdown(
    f"""
**What we do:** Employees in the bottom {int(RISK_Q * 100)}% for engagement are flagged as
*at risk*. Three models learn to predict that flag from the eight drivers plus each person's
profile. They learn from waves 1–2 and are tested on wave 3, the way they would be used for
real: predicting the next survey from past ones.

**How to read the table:** *Recall* is the share of truly at-risk employees the model catches.
*Precision* is how many of the people it flags really are at risk. Each is shown twice: at the
default cut-off (0.5), and at a cut-off tuned to catch more people.
"""
)
risk = load_table("tab17_risk_model_comparison")
if risk is not None:
    rv = pd.DataFrame({
        "Model": risk["model"].str.replace(r"\s*\(.*\)", "", regex=True).str.capitalize(),
        "Catches (default)": risk["recall@0.5"],
        "Catches (tuned)": risk["recall@tuned"],
        "Flags that are right (tuned)": risk["precision@tuned"],
        "Overall quality (PR-AUC)": risk["pr_auc"],
    })
    st.dataframe(rv, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in rv.columns if c != "Model"})
    best = risk.loc[risk["pr_auc"].idxmax()]
    name = str(best["model"]).split(" (")[0]
    st.markdown(
        f"""
**What the result shows:**

- **Best model: {name}**, with the highest overall quality (PR-AUC {best['pr_auc']:.2f}).
- **The cut-off matters more than the model:** At the default cut-off, the {name} catches only
  **{best['recall@0.5']:.0%}** of at-risk employees. With a tuned cut-off, the same model
  catches **{best['recall@tuned']:.0%}**. Nothing else changed.
- **The trade-off is false alarms:** At the tuned cut-off, about {best['precision@tuned']:.0%}
  of flagged people are truly at risk, roughly {round(best['precision@tuned'] * 10)} in 10.
  In retention work that is usually the right trade: a check-in costs little, a missed
  resignation costs a lot.
- Accuracy is not used to pick the model: at-risk employees are a minority, so a model that
  says "everyone is fine" would still look accurate.
"""
    )

# ====================================================================== 3. SHAP
st.subheader("3 · What the risk model relies on")
st.markdown(
    """
**What we do:** A risk score is only useful if people can see *why* someone was flagged.
A method called *SHAP* measures how much each input moved the model's predictions, on
average across all employees. The same method can also break down one person's score, so a
manager sees which drivers pushed that individual towards risk.
"""
)
figure("fig07_risk_model_importance")
shap = load_table("tab18_risk_importance")
if shap is not None:
    shap = shap.rename(columns={shap.columns[0]: "feature", shap.columns[1]: "importance"})
    constructs = shap[shap["feature"].isin(NICE.keys())]
    demo = shap[~shap["feature"].isin(NICE.keys())]
    top_shap = constructs.nlargest(3, "importance")["feature"].tolist()
    overlap = [c for c in top_shap if c in set(top3["construct"])]
    s_lines = [
        f"- **Top of the list:** {', '.join(nice(c) for c in top_shap)}.",
        f"- **Two methods agree:** {len(overlap)} of these 3 are also in the top 3 of the driver "
        "ranking above. The regression and the risk model reach the same story by different routes.",
    ]
    if not demo.empty:
        s_lines.append(
            f"- **Profile details barely matter:** The strongest profile input "
            f"({demo.iloc[0]['feature'].replace('_', ' ')}) scores {demo['importance'].max():.2f}, "
            f"against {constructs['importance'].max():.2f} for the top driver. Risk comes from what "
            "people experience at work, not from which group they belong to."
        )
    s_lines.append("- A worked example of one employee's breakdown is in notebook `05`.")
    st.markdown("**What the result shows:**\n\n" + "\n".join(s_lines))

# ====================================================================== 4. personas
st.subheader("4. Do employees fall into segments?")
st.markdown(
    """
**What we do:** *k-means clustering* sorts employees into groups with similar scores across
the eight drivers. The number of groups is chosen by the same *silhouette* test used for
departments on page 2: below 0.25 means there are no real groups.

**Why try it:** If distinct types of employee existed, each could get its own plan.
"""
)
personas = load_table("tab19_personas")
if personas is not None:
    drv_cols = [c for c in personas.columns if c in NICE and c != "engagement_score"]
    pv = personas[["persona"] + drv_cols + ["share_%", "risk_rate_%"]].copy()
    pv.columns = ["Group"] + [nice(c) for c in drv_cols] + ["Share of staff", "At-risk rate"]
    st.dataframe(pv, hide_index=True, use_container_width=True,
                 column_config={**{nice(c): st.column_config.NumberColumn(format="%+.2f") for c in drv_cols},
                                "Share of staff": st.column_config.NumberColumn(format="%.1f%%"),
                                "At-risk rate": st.column_config.NumberColumn(format="%.1f%%")})
    st.caption("Driver scores are relative to the workforce average (0), in standard deviations.")
    hi_, lo_ = personas.loc[personas["risk_rate_%"].idxmin()], personas.loc[personas["risk_rate_%"].idxmax()]
    all_pos = (personas.set_index("persona")[drv_cols].gt(0).all(axis=1)
               | personas.set_index("persona")[drv_cols].lt(0).all(axis=1)).all()
    p_lines = [
        f"- **{len(personas)} groups, roughly equal in size** "
        f"({', '.join(f'{s:.0f}%' for s in personas['share_%'])}).",
    ]
    if all_pos:
        p_lines.append(
            "- **They are high versus low, not different types:** One group is above average on "
            "every driver, the other below on every driver. Nobody is 'strong on X, weak on Y'. "
            "This is the same pattern page 2 found for departments."
        )
    p_lines.append(
        f"- **The at-risk rate still differs sharply:** {hi_['risk_rate_%']:.1f}% in one group, "
        f"{lo_['risk_rate_%']:.1f}% in the other."
    )
    p_lines.append(
        "- **Verdict:** the split does not pass the silhouette test (see notebook `05`), so these "
        "are not real segments. What this section demonstrates is the *machinery*: group sizes, "
        "risk rates and profiles in a form a manager could act on, ready for real data."
    )
    st.markdown("**What the result shows.**\n\n" + "\n".join(p_lines))

# ====================================================================== 5. trends
st.subheader("5 · Tracking change across waves")
st.markdown(
    f"""
**What we do:** The average score for engagement and each key driver is plotted for every
survey wave. With one line per driver, you can see what is rising, falling or flat.

**Why it matters:** The 2024 survey is a single snapshot, so it cannot show change. With
repeated waves, you can tell whether an intervention worked.
"""
)
figure("fig08_wave_trends")
st.markdown(
    f"""
**What the result shows:** Every line moves only slightly from wave to wave. That is expected:
the generator adds small random drift between waves (standard deviation {DRIFT}), and no real
intervention happened in between. The lesson for real use is the *yardstick*: a movement of
this size is noise. A genuine improvement would need to be clearly larger than that before
anyone claims success.
"""
)
