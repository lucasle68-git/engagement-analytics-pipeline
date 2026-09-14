"""Streamlit stakeholder report — Home / executive summary.

Design principles (see coding report §8):
- Plain language first; technical detail behind expanders.
- A persistent provenance banner on every page — governance, not decoration.
- The app only READS pipeline outputs; it never recomputes analysis, so a number
  here cannot drift from the notebook that produced it.

Run from the project root:  streamlit run app/Home.py
"""

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    figure, load_table, provenance, require, short_theme, survey_badge,
)

st.set_page_config(page_title="Employee Engagement Report", page_icon="📊", layout="wide")

st.title("Employee Engagement — stakeholder report")
st.caption(
    "An interactive read of the analysis for non-technical readers · "
    "MGT5496P Business Analytics Consultancy"
)
survey_badge()

ranking = load_table("tab03_theme_ranking")
require(ranking)

st.subheader("Where engagement stands — in one view")
c1, c2, c3 = st.columns(3)
best, worst = ranking.iloc[0], ranking.iloc[-1]
c1.metric("Strongest theme", short_theme(best.iloc[0]), f"{best['mean']:.2f} / 5")
c2.metric("Weakest theme", short_theme(worst.iloc[0]), f"{worst['mean']:.2f} / 5")

gaps = load_table("tab04_gap_matrix")
if gaps is not None:
    below = int((gaps.set_index(gaps.columns[0]).mean(axis=1) < 0).sum())
    c3.metric("Departments below company average", below)

figure("fig01_theme_ranking")

st.markdown(
    """
**What this means.** Scores sit on a single 1–5 scale — two questions were asked on
1–10 and have been converted so everything is comparable. The ranking is worth reading
for its *shape* rather than its level: themes covering the immediate working
environment tend to sit above those covering the organisationally distant layers.

The more useful observation is how narrow the whole range is. When eight themes
compress into a band this tight, a company-level theme ranking is a poor place to aim
an intervention — which is why the next page moves down to department and item level.
"""
)

st.subheader("How to read this app")
st.markdown(
    """
| Page | Question it answers | Data |
|---|---|---|
| **1 · Current State** | Where do we stand, and which departments diverge? | Survey |
| **2 · Hotspots & Risk** | Which problems are company-wide, and which are local? | Survey |
| **3 · Drivers** | What actually drives engagement? | Synthetic demonstration |
| **4 · 2026 Survey Plan** | So what do we change? | Both |

Pages 1–2 describe what the survey *can* support. Page 3 demonstrates the analysis the
survey **cannot** support today, on generated data, to show what better data would
unlock. Page 4 turns the gap between them into a survey specification.
"""
)

with st.expander("How were these numbers calculated?"):
    st.markdown(
        "Department results in the source workbook are stored as *differences* from the "
        "company average, so absolute scores are reconstructed first (department delta + "
        "company mean). All questions are then placed on a common 1–5 scale, and every "
        "theme score is recomputed from its own questions — the workbook's own theme "
        "roll-ups are not reused, because two of them average a 1–10 question together "
        "with 1–5 questions and are not interpretable on either scale. "
        "Full detail: notebook `01_data_loading_quality.ipynb`."
    )

if provenance() == "demo":
    with st.expander("Why does this say DEMO?"):
        st.markdown(
            "The client's survey is confidential and is not distributed with this "
            "project. To keep the analysis runnable and inspectable by anyone, the "
            "repository ships a fabricated workbook with the *same structure* as the "
            "real one — same questions, same scales, same encoding — and entirely "
            "invented numbers.\n\n"
            "The banner is not written by hand on each page: it is derived from "
            "`data.provenance` in `config/config.yaml`, the same switch that controls "
            "the badge stamped onto every saved figure. Point the project at the real "
            "workbook and both change together."
        )
