"""The 2026 Survey Plan — "So what do we change?" (Phase 3 synthesis).

Reads tab20-tab23 and the figures the pipeline produced. Nothing is recomputed.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import (figure, key_takeaways, load_table, require,  # noqa: E402
                     survey_badge, synthetic_badge)

st.set_page_config(page_title="2026 Survey Plan", page_icon="📊", layout="wide")

NICE = {
    "development": "Development", "personal": "Personal resources", "social": "Social support",
    "balance": "Work–life balance", "justice": "Fairness", "technology": "Technology",
    "psych_safety": "Psychological safety", "esg": "ESG alignment",
}


def nice(x: str) -> str:
    return NICE.get(str(x), str(x).replace("_", " ").capitalize())


# Plain-language versions of tab20 / tab22, keyed by the raw table's client question /
# element. A row whose key is missing here falls back to the raw text, so a change in
# synthesis.py can never silently drop a row from the page.
CAP_COLS = ["Client question", "Today’s survey", "What the demonstration showed",
            "What 2026 must collect"]
CAP_PLAIN = {
    "What drives engagement? (O2)": (
        "What drives engagement?",
        "Can’t answer: only department averages, no direct engagement score",
        "Ranked the drivers and confirmed the ranking with more than one method",
        "Each person’s answers + a direct engagement measure"),
    "Who is at risk? (O3)": (
        "Who is at risk?",
        "Only by department: flags warning-sign questions, not people",
        "A risk score per person, tested on a later wave, with reasons shown",
        "Each person’s answers + role, region and tenure"),
    "Are the scales sound?": (
        "Are the scales reliable?",
        "Can’t be tested on averages. One flaw found: mixed 1–5 and 1–10 scales",
        "Drivers measured with only 2–3 questions were the least reliable",
        "Each person’s answer to every question"),
    "Are there segments? (O1/O3)": (
        "Are there employee segments?",
        "No: departments don’t split into distinct types",
        "A ready-made grouping tool: size, risk rate and profile per group",
        "Profile data + department sizes"),
    "Is engagement moving? (O1)": (
        "Is engagement changing?",
        "Can’t tell: one survey, one point in time",
        "Scores tracked across three survey waves",
        "The same survey, repeated over time"),
}

RD_COLS = ["Decision", "What", "Why"]
RD_PLAIN = {
    "Manager / team / leadership relationship blocks": (
        "KEEP", "Manager, team and leadership questions",
        "Good coverage of support at work, and the company’s current strengths"),
    "eNPS / recommendation and retention items": (
        "KEEP", "Recommendation (eNPS) and intent-to-stay questions",
        "A stable yardstick to compare year on year"),
    "UWES-9 engagement block (vigour / dedication / absorption)": (
        "ADD", "A standard engagement scale (9 questions)",
        "There is no direct engagement score today; every driver and risk model needs one"),
    "Personal-resources scale (>=4 items)": (
        "ADD", "Personal resources: self-belief, optimism, resilience (4+ questions)",
        "Strongest link to engagement in published research and a top driver in the "
        "demonstration, yet not asked in 2024"),
    "Technology & work-mode items + work-mode field": (
        "ADD", "Technology and work-mode questions (remote / hybrid / office)",
        "Matters for distributed teams, and carried real weight in the demonstration"),
    "Psychological-safety and justice scales (>=4 items each)": (
        "ADD", "Psychological safety and fairness (4+ questions each)",
        "Blocks of only 2–3 questions measured least reliably"),
    "ESG-alignment items": (
        "ADD", "Alignment with environmental and social goals (ESG)",
        "An emerging lever in fintech"),
    "Demographics: region, tenure, role level, work mode, department size": (
        "ADD", "Profile: region, tenure, role level, work mode, department size",
        "Needed to find who is at risk; even department sizes were missing"),
    "Redundant sentiment items within 'Working for AccessFintech'": (
        "DROP/MERGE", "Overlapping sentiment questions in “Working for AccessFintech”",
        "They partly measure the outcome itself; the new engagement scale takes over (keep eNPS)"),
    "Single 1-5 scale for every item": (
        "FIX", "One 1–5 scale for every question",
        "Mixing 1–5 and 1–10 distorted the client’s own summary scores"),
    "Anonymised ITEM-LEVEL multi-wave storage": (
        "FIX", "Store anonymised answers per person, per survey",
        "Storing only department averages blocked most analysis; this is the single biggest fix"),
}


st.title("The 2026 Survey Plan")

c1, c2 = st.columns(2)
with c1:
    survey_badge()
with c2:
    synthetic_badge()

st.markdown(
    "**This page combines the two.** Performance comes from the 2024 survey. Importance comes "
    "from the demonstration on page 3."
)

_cap = load_table("tab20_capability_comparison")
_ip4 = load_table("tab21_importance_performance")
_rd = load_table("tab22_survey_redesign")
if _cap is not None:
    _pts = [
        "**Today's survey answers none of the questions stakeholders asked.** Drivers, risk, "
        "segments, trends, every one fails and every failure traces to a data element that "
        "was never collected, not to the choice of method."
    ]
    if _ip4 is not None:
        _un = _ip4.loc[~_ip4["measured_2024"], "importance_%"].sum()
        _pts.append(
            f"**Measure first, act second.** **{_un:.1f}%** of what drives engagement is "
            f"invisible in the 2024 survey, against **{100 - _un:.1f}%** that is measured. "
            f"The biggest win is asking the missing questions."
        )
    if _rd is not None:
        _d = _rd["decision"].value_counts()
        _pts.append(
            "**Extend the survey, do not replace it.** The specification is "
            + ", ".join(f"**{int(n)} {k}**" for k, n in _d.items())
            + ". The 2024 instrument is not wrong, it is incomplete."
        )
    key_takeaways(*_pts)

capability = load_table("tab20_capability_comparison")
require(capability)

# ====================================================================== 1. capability
st.subheader("What today’s survey cannot answer, and why")
cap = pd.DataFrame(
    [CAP_PLAIN.get(str(r.iloc[0]), tuple(r.iloc[:4])) for _, r in capability.iterrows()],
    columns=CAP_COLS,
)
st.dataframe(cap, hide_index=True, use_container_width=True)
with st.expander("Technical detail"):
    st.dataframe(capability, hide_index=True, use_container_width=True)
st.markdown(
    f"""
**Read the last column.** Each question the survey cannot answer fails for one reason: a
**specific piece of data was never collected**, not because the method was weak. All
{len(cap)} rows trace back that way.

So the 2026 survey has one job: collect those missing pieces. Fix them, and every question in
this table becomes answerable.
"""
)

# ====================================================================== 2. where to act
st.subheader("Where to act")
st.markdown(
    """
**What this map does.** It places each driver on two axes and splits them into four boxes:

- **Across: performance.** The driver's score in the 2024 survey. Further right is better.
- **Up: importance.** Its share of what explains engagement, from the demonstration.

The box that matters is top left: **important, but scoring low**. The red panel on the left
holds the drivers the 2024 survey never measured, so they have no score to place, only an
importance value.

**Each measured driver also shows two dots**, joined by a line: its average, and its weakest
question. The longer the line, the more the average is hiding.
"""
)
figure("fig09_importance_performance")

ip = load_table("tab21_importance_performance")
if ip is not None:
    iv = pd.DataFrame({
        "Driver": ip["construct"].map(nice),
        "Importance": ip["importance_%"],
        "2024 score": ip["performance_2024"],
        "Weakest question": ip["weakest_item_score"],
        "Priority items": ip["n_priority_items"],
        "Verdict": ip["quadrant"].map(lambda v: str(v).split(" (")[0]),
    })
    st.dataframe(iv, hide_index=True, use_container_width=True,
                 column_config={
                     "Importance": st.column_config.NumberColumn(format="%.1f%%"),
                     "2024 score": st.column_config.NumberColumn(format="%.2f"),
                     "Weakest question": st.column_config.NumberColumn(format="%.2f"),
                     "Priority items": st.column_config.NumberColumn(format="%.0f"),
                 })

    measured, unmeasured = ip[ip["measured_2024"]], ip[~ip["measured_2024"]]
    lines = [
        f"- **Nearly half of what matters is invisible today:** The {len(unmeasured)} drivers the "
        f"2024 survey never asked about: {', '.join(nice(c) for c in unmeasured['construct'])}, "
        f"carry **{unmeasured['importance_%'].sum():.1f}%** of the explanation. The measured drivers "
        f"carry {measured['importance_%'].sum():.1f}%. Those four cannot even be placed on the "
        "grid, and that absence is the headline.",
        f"- **{nice(unmeasured.iloc[0]['construct'])} is the clearest case:** "
        f"{unmeasured.iloc[0]['importance_%']:.1f}% of the explanation, and not one question "
        "about it in the current survey.",
    ]
    improve = measured[(measured["importance_%"] > measured["importance_%"].median())
                       & (measured["performance_2024"] < measured["performance_2024"].median())]
    if improve.empty:
        lines.append(
            "- **The top-left box is empty and that is arithmetic, not good news:** Among the "
            "measured drivers, the more important ones also score higher, so nothing can land there. "
            "It does not mean nothing needs improving."
        )
    hidden = measured.assign(gap=measured["performance_2024"] - measured["weakest_item_score"])
    hidden = hidden.sort_values("gap", ascending=False).iloc[0]
    lines.append(
        f"- **Averages hide the work:** {nice(hidden['construct'])} scores "
        f"{hidden['performance_2024']:.2f}, which looks safe. Its weakest question scores "
        f"{hidden['weakest_item_score']:.2f} — {hidden['gap']:.2f} points lower. That question, "
        "not the theme, is what an intervention would target."
    )
    st.markdown("**What the map shows:**\n\n" + "\n".join(lines))

# ====================================================================== 3. instrument
st.subheader("Keep, add, drop, fix")
redesign = load_table("tab22_survey_redesign")
if redesign is not None:
    rv = pd.DataFrame(
        [RD_PLAIN.get(str(r.iloc[1]), tuple(r.iloc[:3])) for _, r in redesign.iterrows()],
        columns=RD_COLS,
    )
    st.dataframe(rv, hide_index=True, use_container_width=True)
    with st.expander("Technical detail"):
        st.dataframe(redesign, hide_index=True, use_container_width=True)

    counts = redesign["decision"].value_counts()
    adds, drops = int(counts.get("ADD", 0)), int(counts.get("DROP/MERGE", 0))
    fixes, keeps = int(counts.get("FIX", 0)), int(counts.get("KEEP", 0))
    st.markdown(
        f"""
**The justification column is the deliverable.** Every row points back to the analysis that
produced it, so each recommendation can be checked and challenged on its own. That is what
keeps the whole project logically connected, from the first data check to this table.

**The shape of the advice: {adds} ADD against {drops} DROP.** ({keeps} KEEP, {fixes} FIX.) The
2024 survey is **not wrong, it is incomplete**. It does not need replacing. It needs extending,
with more questions and more detail: the missing drivers, and answers stored per person per wave.
"""
    )

trace = load_table("tab23_traceability")
if trace is not None:
    with st.expander("Traceability: every deliverable back to an objective"):
        tv = trace.copy()
        tv.columns = ["Deliverable", "Refined objective (RO)", "Client objective (O)"][:len(tv.columns)]
        st.dataframe(tv, hide_index=True, use_container_width=True)
        st.caption("The refined objectives RO1–RO4 are defined in the project README.")
