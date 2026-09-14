"""Current State Explorer — "Where do we stand?" (survey data, Phase 1)."""

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import figure, load_table, require, survey_badge  # noqa: E402

st.set_page_config(page_title="Current State", page_icon="📊", layout="wide")

st.title("Current State Explorer")
survey_badge()

ranking = load_table("tab03_theme_ranking")
gaps = load_table("tab04_gap_matrix")
require(ranking, gaps)

st.subheader("Theme ranking")
st.bar_chart(ranking.set_index(ranking.columns[0])["mean"])
st.markdown(
    "**What this means.** Longer bars are stronger themes. Read the *order* rather than "
    "the individual values: themes about the immediate working environment usually sit "
    "above themes about distant organisational layers."
)

st.subheader("A department against the company average")
gm = gaps.set_index(gaps.columns[0])
dept = st.selectbox("Choose a department", gm.index)
row = gm.loc[dept].sort_values()
st.bar_chart(row)
st.markdown(
    f"**What this means.** Bars below zero are themes where department **{dept}** sits "
    "under the company average, above zero where it sits over. The shape matters more "
    "than any single bar: one deep negative with the rest near zero is a local problem "
    "with a named owner, while consistently below-average bars across most themes are a "
    "departmental climate problem. The two need different responses, and a theme average "
    "alone cannot tell them apart."
)

st.subheader("The full department × theme grid")
figure("fig02_department_gap_heatmap")
st.markdown(
    "**What this means.** Read across a row to compare departments on one theme, and "
    "down a column to see whether a department is uniformly weak or weak in one place. "
    "Departments that score well are useful as internal benchmarks — colleagues to ask "
    "*how*, rather than a league table."
)

top = load_table("tab05_top_gaps")
if top is not None:
    st.subheader("Largest individual gaps")
    st.dataframe(top, use_container_width=True)
    st.markdown(
        "**Why item level, not theme level.** Large negative gaps often turn up inside "
        "the company's *best-scoring* theme, sometimes in departments whose theme-level "
        "gap looks unremarkable. Averaging hides exactly the cells an intervention "
        "would target."
    )
