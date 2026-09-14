"""The 2026 Survey Plan — "So what do we change?" (Phase 3 synthesis)."""

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import figure, load_table, require, survey_badge, synthetic_badge  # noqa: E402

st.set_page_config(page_title="2026 Survey Plan", page_icon="📊", layout="wide")

st.title("The 2026 survey plan")

c1, c2 = st.columns(2)
with c1:
    survey_badge()
with c2:
    synthetic_badge()

st.markdown(
    "This page combines the two. **Performance** comes from the survey; **importance** "
    "from the demonstration. They are put side by side only to *prioritise* — never to "
    "assert a new fact about the workforce."
)

capability = load_table("tab20_capability_comparison")
require(capability)

st.subheader("What today's survey cannot answer — and why")
st.dataframe(capability, use_container_width=True)
st.markdown(
    "**Read the last column.** Every question the survey fails to answer is traced to a "
    "specific *missing data element*, not to a shortcoming of method. That is what makes "
    "the 2026 redesign a consequence rather than a preference — each proposed change "
    "exists because a named analysis failed for a named reason."
)

st.subheader("Where to act")
figure("fig09_importance_performance")
ip = load_table("tab21_importance_performance")
if ip is not None:
    st.dataframe(ip, use_container_width=True)
    st.markdown(
        "**Two things to notice.**\n\n"
        "First, constructs marked `MEASURE FIRST` have no performance value at all — the "
        "survey never asked about them. They cannot be placed on the grid, and that "
        "absence *is* the headline: they are invisible to any prioritisation built on "
        "today's survey alone, however careful.\n\n"
        "Second, the `weakest_item` column sits beside each theme mean on purpose. A "
        "theme can look comfortable on average while one of its questions scores far "
        "below the rest, and it is that question an intervention would target."
    )

st.subheader("The instrument: keep, add, drop, fix")
redesign = load_table("tab22_survey_redesign")
if redesign is not None:
    st.dataframe(redesign, use_container_width=True)
    st.markdown(
        "**The justification column is the deliverable.** Every row points back to the "
        "analysis that produced it, so the client can challenge any single "
        "recommendation without having to accept or reject the whole set.\n\n"
        "Note the shape of the advice: mostly **ADD**, with very little dropped. The "
        "survey is *incomplete*, not broken — roughly half the explanatory weight sits in "
        "constructs it never asked about. The instrument needs extending, not replacing, "
        "and the two **FIX** rows (one consistent 1–5 scale; storing item-level responses "
        "across waves) cost almost nothing yet unblock every analysis that failed above."
    )

trace = load_table("tab23_traceability")
if trace is not None:
    with st.expander("Traceability — deliverable → objective"):
        st.dataframe(trace, use_container_width=True)
