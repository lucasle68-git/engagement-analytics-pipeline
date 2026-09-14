"""What Drives Engagement — a demonstration on synthetic data (Phase 2)."""

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import figure, load_table, require, synthetic_badge  # noqa: E402

st.set_page_config(page_title="Drivers (demonstration)", page_icon="📊", layout="wide")

st.title("What drives engagement — a demonstration")
synthetic_badge()

st.markdown(
    """
**Why this page is different from the ones before it.** The survey is *aggregated*: it
reports averages per department, with no individual responses, no outcome question and
no repeat waves. That makes the question "what drives engagement?" unanswerable from it —
not because the method is hard, but because driver analysis needs individual rows and
this data has none.

So the analysis is demonstrated on a **generated panel** — 1,000 employees across 3 waves
— built to match the survey's department averages while adding the individual-level
variation the real data never had. Everything below shows what the 2026 survey would
make possible. None of it is a finding about a real workforce.
"""
)

drivers = load_table("tab16_driver_ranking")
require(drivers)

st.subheader("Ranked drivers")
figure("fig06_driver_ranking")
st.dataframe(drivers, use_container_width=True)
st.markdown(
    "**This is a recovery test, and that is what makes it meaningful.** The generator was "
    "given a known set of driver strengths taken from the engagement literature. The "
    "pipeline is then shown only the simulated questionnaire responses and asked to rank "
    "the drivers. Getting the imposed order back is evidence the method works — on data "
    "where, unusually, the right answer is known in advance.\n\n"
    "`VIF` near 1 means the drivers are not tangled up in each other, so the ranking is "
    "stable rather than an artefact of collinearity. Watch where the constructs the 2024 "
    "survey **never measured** land in the ranking: anything near the top is a driver no "
    "amount of analysis on today's survey could have found."
)

st.subheader("Who is at risk of disengaging")
risk = load_table("tab17_risk_model_comparison")
if risk is not None:
    st.dataframe(risk, use_container_width=True)
    st.markdown(
        "**The column worth reading is not the model name — it is `recall@0.5` against "
        "`recall@tuned`.** Those two numbers come from the *same fitted model*. The only "
        "difference is the cut-off at which a probability is called \"at risk\".\n\n"
        "At the default 0.5 the model finds a small fraction of at-risk employees; tuned "
        "to the operating point the problem actually calls for, it finds most of them. "
        "In a retention setting, where missing someone costs far more than an unnecessary "
        "check-in, choosing the threshold matters more than choosing the algorithm. "
        "Models are compared on PR-AUC rather than accuracy for the same reason: at-risk "
        "employees are the minority, and accuracy rewards predicting \"fine\" for everyone."
    )
    figure("fig07_risk_model_importance")

st.subheader("Do employees fall into segments?")
personas = load_table("tab19_personas")
if personas is not None:
    st.dataframe(personas, use_container_width=True)
    st.markdown(
        "**Held to the same standard as page 2, and it fails in the same way.** The "
        "segments are judged against the same silhouette convention used for the "
        "departments, and they do not clear it. What is demonstrated here is the "
        "machinery — per-segment risk rates and demographic mixes in the form a "
        "stakeholder would act on — not a claim that these personas are real.\n\n"
        "A project that discovers segments only when it needs some has a method problem, "
        "not a finding."
    )

st.subheader("Tracking change across waves")
figure("fig08_wave_trends")
st.markdown(
    "**What this unlocks.** A single survey wave can describe today; it cannot show "
    "movement. Three waves of individual-level data make it possible to see whether an "
    "intervention worked — and to tell a real shift apart from ordinary noise."
)
