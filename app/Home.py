"""Streamlit stakeholder report — Home / executive summary.

Design principles:
- Plain language first; technical detail behind expanders.
- A provenance banner on every page — governance, not decoration.
- The app only READS pipeline outputs; it never recomputes the analysis, so a
  number here cannot drift from the notebook that produced it.

Run from the project root:  make app   (or: streamlit run app/Home.py)
"""

from pathlib import Path
import sys

import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared import (  # noqa: E402
    ROOT, figure, key_takeaways, load_table, provenance, require, short_theme,
    survey_badge,
)

st.set_page_config(page_title="Employee Engagement Report", page_icon="📊", layout="wide")

st.title("Employee Engagement: Stakeholder report")
st.caption(
    "An interactive read of the analysis for non-technical readers · "
    "MGT5496P Business Analytics Consultancy"
)
survey_badge()

# ---------------------------------------------------------------- key takeaways
_rk = load_table("tab03_theme_ranking")
_gm = load_table("tab04_gap_matrix")
_cp = load_table("tab10_composite_index")
if _rk is not None:
    _pts = [
        f"**Engagement is positive, but flat.** All eight themes sit between "
        f"**{_rk['mean'].min():.2f}** and **{_rk['mean'].max():.2f}** out of 5. Nothing is in "
        f"crisis, and nothing stands out as the obvious place to act."
    ]
    if _gm is not None and _cp is not None:
        _per = _gm.set_index(_gm.columns[0]).mean(axis=1)
        _dep = _cp[_cp.iloc[:, 0] != "Company"]["composite_index"]
        _pts.append(
            f"**The real differences are between departments, not themes.** "
            f"**{int((_per < 0).sum())} of {len(_per)}** departments score below the company "
            f"average, and department scores run from **{_dep.min():.2f}** to "
            f"**{_dep.max():.2f}**, a wider spread than the themes above. Page 1 names the "
            f"departments and shows where each one differs."
        )
    _pts.append(
        "**This survey can describe, not explain.** It holds department averages only, so it "
        "shows *where* engagement stands but never *why*. Page 3 demonstrates the answer on "
        "better data; page 4 lists what the 2026 survey must collect to get it."
    )
    key_takeaways(*_pts)


ranking = load_table("tab03_theme_ranking")
require(ranking)

# ---------------------------------------------------------------- headline tiles
st.subheader("Engagement at a glance")

best, worst = ranking.iloc[0], ranking.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Strongest theme · {short_theme(best.iloc[0])}", f"{best['mean']:.2f} / 5")
c2.metric(f"Weakest theme · {short_theme(worst.iloc[0])}", f"{worst['mean']:.2f} / 5")

gaps = load_table("tab04_gap_matrix")
if gaps is not None:
    per_dept = gaps.set_index(gaps.columns[0]).mean(axis=1)
    c3.metric("Departments below company average",
              f"{int((per_dept < 0).sum())} of {len(per_dept)}")

composite = load_table("tab10_composite_index")
if composite is not None:
    company = composite.loc[composite.iloc[:, 0] == "Company", "composite_index"]
    if not company.empty:
        c4.metric("Company composite score", f"{float(company.iloc[0]):.2f} / 5")

figure("fig01_theme_ranking")

# ---------------------------------------------------------------- what it means
try:
    cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    ten_point = cfg.get("data", {}).get("ten_point_items", [])
except Exception:
    ten_point = []
ten_point_list = "\n".join(f"- *{q}*" for q in ten_point) or "- *(see config.yaml)*"

lo, hi = ranking["mean"].min(), ranking["mean"].max()

st.markdown(
    f"""
**How to read the scores.** Every score is on one 1–5 scale. Two questions were
asked on a 1–10 scale instead:

{ten_point_list}

They were converted with a simple straight-line rescale, `1 + (score − 1) × 4/9`,
so 1 stays 1 and 10 becomes 5. That makes every question comparable.

**What stands out.** Themes about people's day-to-day work: their team, their role,
their manager, score higher than themes about the more distant layers of the
organisation, such as heads of department and the executive team.

But the whole range is narrow: all eight themes sit between **{lo:.2f}** and
**{hi:.2f}**. A spread this small is a weak basis for choosing where to act. That is why
the next pages move down to department level, and then to individual questions.
"""
)

# ---------------------------------------------------------------- navigation
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

**Themes and drivers.** Pages 1–2 use the 2024 survey's 8 **themes**, which group the 34
questions by *who* they are about. Pages 3–4 regroup the same questions into **drivers**,
by *what they measure*. Page 3 shows the full mapping.
"""
)

# ---------------------------------------------------------------- expanders
with st.expander("How were these numbers calculated?"):
    st.markdown(
        """
The survey file needs five steps before any number on this page can be trusted.

1. **Load and check.** The file is read and its layout is checked: the right columns,
   the right theme headings. If anything has changed, the run stops with a clear error
   instead of producing a wrong answer.
2. **Reshape.** The spreadsheet is turned into one row per department per question,
   which makes every later step simple and testable.
3. **Rebuild the real scores.** The file stores each department as a *difference* from
   the company average, not as a score. So each score is rebuilt:
   company average + department difference.
4. **Put everything on one scale.** The two 1–10 questions are converted to 1–5, as
   described above.
5. **Recompute every theme.** Each theme score is recalculated from its own questions.
   The file's own theme totals are not used, because two of them mix 1–10 and 1–5
   questions and so do not mean anything on either scale.

Full detail: notebook `01_data_loading_quality`.
"""
    )

if provenance() == "demo":
    with st.expander("Why does this say DEMO?"):
        st.markdown(
            "The client's survey is confidential and is not shared with this project. So "
            "that anyone can still run and check the analysis, the repository includes a "
            "stand-in workbook with **the same structure** as the real one: the same "
            "questions, the same scales, the same layout. Only the numbers are invented.\n\n"
            "The banner is not written by hand on each page: it is derived from "
            "`data.provenance` in `config/config.yaml`, the same switch that controls "
            "the badge stamped onto every saved figure. Point the project at the real "
            "workbook and both change together."
        )
