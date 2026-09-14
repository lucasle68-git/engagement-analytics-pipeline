"""Hotspots & Risk Areas — "Where should we worry?" (survey data, Phase 1)."""

from pathlib import Path
import sys

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import figure, load_table, require, survey_badge  # noqa: E402

st.set_page_config(page_title="Hotspots & Risk", page_icon="📊", layout="wide")

st.title("Hotspots & Risk Areas")
survey_badge()

disp = load_table("tab06_dispersion_screen")
require(disp)

st.subheader("Priority items for 2026")
st.markdown(
    "**How to read this.** A low score means two very different things depending on how "
    "it is spread:\n\n"
    "- **Company-wide problem** — low nearly everywhere, so it needs a policy-level fix.\n"
    "- **Localised problem** — low because specific departments are dragging it down, so "
    "it needs a targeted intervention.\n\n"
    "The distinction changes what action is appropriate, which is why the screen sorts on "
    "spread as well as on level."
)
flagged = disp[disp["priority_2026"]]
st.dataframe(flagged, use_container_width=True)

st.subheader("Level against spread")
figure("fig02b_dispersion_quadrants")
st.markdown(
    "**The quadrant worth your attention is the top-right**: items with a *high* average "
    "and a *wide* spread — strengths that are delivered unevenly. A strong theme average "
    "can hide the fact that some departments are not getting it at all, and no ranking of "
    "averages will ever surface that."
)

st.subheader("Do departments form groups?")
clusters = load_table("tab08_department_clusters")
if clusters is not None:
    left, right = st.columns([1, 2])
    with left:
        st.dataframe(clusters, use_container_width=True)
    with right:
        figure("fig03a_dendrogram_level")
    st.markdown(
        "**What this means — and what it does not.** The honest answer on this data is "
        "that there are *no distinct kinds of department*. Grouping attempts either fail "
        "the silhouette convention used throughout the project (below 0.25 = no "
        "substantial structure) or improve only by splitting off a single department on "
        "its own, which is an outlier rather than a segment.\n\n"
        "Departments differ in how high they score overall, not in *which* themes they are "
        "weak in. So this is a ranking of who needs attention first — useful — and not a "
        "discovery of department archetypes. Reporting it as the weaker claim is the point."
    )

st.subheader("Where to focus first (importance × performance)")
ipma = load_table("tab09_ipma")
if ipma is not None:
    st.dataframe(ipma, use_container_width=True)
    figure("fig04c_ipma_map")
    st.markdown(
        "**What this means.** Themes flagged `priority` matter most to overall sentiment "
        "yet score lowest today.\n\n"
        "**Three cautions travel with this map, and they are not boilerplate.** The "
        "importance axis is a correlation proxy computed from roughly 19 department units, "
        "so it is weak by construction. The performance axis spans a fraction of a scale "
        "point, so small differences shift a theme between quadrants. And the questions "
        "used to define importance sit inside one of the themes being scored, which "
        "inflates it — the arrow on the chart shows that correction applied to scale. "
        "Treat this as a prioritisation aid, not a finding."
    )
