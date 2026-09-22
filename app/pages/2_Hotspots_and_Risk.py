"""Hotspots & Risk Areas — "Where should we worry?" (survey data, Phase 1).

Every number on this page is read from pipeline outputs (tab06, tab08*, tab09) or
looked up from config. Nothing is re-estimated here.
"""

from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import (ROOT, figure, key_takeaways, load_table,  # noqa: E402
                     require, short_theme, survey_badge)

st.set_page_config(page_title="Hotspots & Risk", page_icon="📊", layout="wide")

try:
    CFG = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
except Exception:
    CFG = {}
P1 = CFG.get("phase1", {})
LOW_Q = P1.get("low_mean_quantile", 0.25)
SPREAD_Q = P1.get("dispersion_high_spread_quantile", 0.75)
ANCHORS = CFG.get("data", {}).get("outcome_anchors", [])


def q(text: str) -> str:
    """Quote a survey question without its trailing full stop."""
    return f"“{str(text).rstrip('.')}”"


st.title("Hotspots & Risk Areas")
survey_badge()

_dp = load_table("tab06_dispersion_screen")
_ks = load_table("tab08a_cluster_k_selection")
_ip = load_table("tab09_ipma")
if _dp is not None:
    _q = _dp["quadrant"].value_counts()
    _pts = [
        f"**{int(_dp['priority_2026'].sum())} of {len(_dp)} questions are flagged for "
        f"2026.** **{int(_q.get('company-wide problem', 0))}** are company-wide, where "
        f"everyone scores low, and **{int(_q.get('localised problem', 0))}** are localised, "
        f"where most departments are fine and a few sit far below. Each needs a different "
        f"response."
    ]
    if _ks is not None:
        _best = max(_ks["silhouette_level"].max(), _ks["silhouette_shape"].max())
        _pts.append(
            f"**Departments do not fall into groups.** The best score the clustering reaches "
            f"is **{_best:.2f}**, below the 0.25 mark that would signal real structure. There "
            f"are no department types to manage, each one has to be read on its own."
        )
    if _ip is not None and "priority" in _ip.columns:
        _names = ", ".join(short_theme(c) for c in _ip.loc[_ip["priority"], _ip.columns[0]])
        _pts.append(
            f"**The priority map points at {_names}** matters a lot, scores badly. But it "
            f"rests on 19 data points, so read it as a pointer for the 2026 survey, not proof."
        )
    key_takeaways(*_pts)

disp = load_table("tab06_dispersion_screen")
require(disp)

low_cut = disp["company_mean"].quantile(LOW_Q)
spread_cut = disp["spread"].quantile(SPREAD_Q)
counts = disp["quadrant"].value_counts()

# ====================================================================== 1. priority items
st.subheader("Priority items for 2026")
st.markdown(
    f"""
**What this table shows:** The {int(disp['priority_2026'].sum())} questions, out of
{len(disp)}, with the lowest company-wide score: the bottom quarter, below **{low_cut:.2f}**.
For each one you can see the company average, the lowest and highest department, and the
**spread**, how far apart the departments are.

**How to read it:** A low score means two different things, depending on the spread:

- **Company-wide problem:** low in nearly every department. It needs a policy-level fix.
- **Localised problem:** low because some departments pull it down. It needs a targeted fix
  in those departments.
"""
)

flagged = disp[disp["priority_2026"]].copy().sort_values("company_mean")
view = pd.DataFrame({
    "Type": flagged["quadrant"].str.capitalize(),
    "Theme": flagged["category"].map(lambda t: short_theme(t, limit=40)),
    "Question": flagged["item"],
    "Company average": flagged["company_mean"],
    "Lowest dept": flagged["dept_min"],
    "Highest dept": flagged["dept_max"],
    "Spread": flagged["spread"],
})
st.dataframe(view, hide_index=True, use_container_width=True,
             column_config={c: st.column_config.NumberColumn(format="%.2f")
                            for c in ["Company average", "Lowest dept", "Highest dept", "Spread"]})
st.caption("Spread = standard deviation of the department scores. Higher means the "
           "departments disagree more.")

cw = flagged[flagged["quadrant"] == "company-wide problem"]
lc = flagged[flagged["quadrant"] == "localised problem"]
notes = []
if not cw.empty:
    top_theme = cw["category"].value_counts()
    notes.append(
        f"- **{len(cw)} company-wide problems.** "
        + (f"{int(top_theme.iloc[0])} of them sit in *{short_theme(top_theme.index[0], 40)}*. "
           if top_theme.iloc[0] > 1 else "")
        + f"The lowest is {q(cw.iloc[0]['item'])} at {cw.iloc[0]['company_mean']:.2f}, and no "
        f"department rises above {cw.iloc[0]['dept_max']:.2f} on it."
    )
if not lc.empty:
    widest = lc.assign(rng=lc["dept_max"] - lc["dept_min"]).sort_values("rng").iloc[-1]
    notes.append(
        f"- **{len(lc)} localised problems.** The widest is {q(widest['item'])}: from "
        f"{widest['dept_min']:.2f} in the weakest department to {widest['dept_max']:.2f} in the "
        f"strongest. Some departments already do this well; the fix is local."
    )
if notes:
    st.markdown("**What the data shows.**\n\n" + "\n".join(notes))

# ====================================================================== 2. level vs spread
st.subheader("Level against spread")
st.markdown(
    f"""
**How this chart is built:** Every one of the {len(disp)} questions is placed by two
numbers:

- **Across:** its company average. Further right is better.
- **Up:** its spread across departments. Higher means the departments disagree more.

Two dashed lines cut the chart into four boxes. The vertical line marks the bottom quarter on
average (**{low_cut:.2f}**); the horizontal line marks the top quarter on spread
(**{spread_cut:.2f}**). Both cut-offs come from this survey itself, not from an outside benchmark.
"""
)
figure("fig02b_dispersion_quadrants")

uneven = disp[disp["quadrant"] == "uneven strength"]
bullets = [
    f"- **Uniform strength** (bottom right, {int(counts.get('uniform strength', 0))} questions): "
    "high and consistent -> Keep doing what works.",
    f"- **Company-wide problem** (bottom left, {int(counts.get('company-wide problem', 0))}): "
    "low everywhere -> A policy-level fix.",
    f"- **Localised problem** (top left, {int(counts.get('localised problem', 0))}): "
    "low, but only in some departments -> A targeted fix.",
    f"- **Uneven strength** (top right, {int(counts.get('uneven strength', 0))}): "
    "high on average, but some departments are left behind.",
]
st.markdown("**The four boxes.**\n\n" + "\n".join(bullets))

if not uneven.empty:
    u = uneven.assign(rng=uneven["dept_max"] - uneven["dept_min"]).sort_values("rng").iloc[-1]
    st.markdown(
        f"""
**The box worth your attention is the top right.** These questions look like strengths,
so no ranking of averages will ever flag them. Take {q(u['item'])}: the company average is
**{u['company_mean']:.2f}**, comfortably high. But departments range from **{u['dept_min']:.2f}**
to **{u['dept_max']:.2f}**, a gap of {u['rng']:.2f} points on the same question. The average
calls it a strength; for the weakest department it is a problem.
"""
    )

# ====================================================================== 3. clustering
st.subheader("Do departments form groups?")
st.markdown(
    """
**What we tried:** Clustering sorts departments so that the most similar ones sit together,
based on their eight theme scores. We used *hierarchical clustering* (Ward's method): it
starts with every department on its own and merges the two most similar, step by step, until
everything is one group. The tree below records every merge.

**The goal** was to find *kinds* of department. For example, "strong on team, weak on
leadership" that could each get their own plan.

**How we judge the result:** A *silhouette score* from 0 to 1 says how clearly the groups
separate. Above 0.50 is a real structure; 0.25–0.50 is weak; **below 0.25 means no real
groups**. We also reject any "group" of just one department: that is an outlier, not a group.
"""
)

ksel = load_table("tab08a_cluster_k_selection")
prof = load_table("tab08c_cluster_profiles")
clusters = load_table("tab08_department_clusters")

left, right = st.columns([1, 2])
with left:
    if clusters is not None:
        grp = clusters.groupby(clusters.columns[1])[clusters.columns[0]].apply(
            lambda s: ", ".join(map(str, s)))
        gtab = pd.DataFrame({"Group": [f"Group {g}" for g in grp.index],
                             "Departments": grp.values,
                             "Count": clusters[clusters.columns[1]].value_counts()
                                                                   .reindex(grp.index).values})
        if prof is not None and "OVERALL" in prof.columns:
            gtab["Average score"] = prof.set_index(prof.columns[0])["OVERALL"].reindex(grp.index).values
        st.markdown("**Best split found (2 groups)**")
        st.dataframe(gtab, hide_index=True, use_container_width=True,
                     column_config={"Average score": st.column_config.NumberColumn(format="%.2f")})
    if ksel is not None:
        kv = pd.DataFrame({
            "Groups tried": ksel["k"],
            "Silhouette (by level)": ksel["silhouette_level"],
            "Silhouette (by pattern)": ksel["silhouette_shape"],
            "Single-dept groups": ksel["n_singletons_level"],
        })
        st.markdown("**Every split tried**")
        st.dataframe(kv, hide_index=True, use_container_width=True,
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["Silhouette (by level)", "Silhouette (by pattern)"]})
with right:
    figure("fig03a_dendrogram_level")
    st.caption("How to read the tree: departments that join lower down are more alike. "
               "The two big branches are the two groups on the left.")

findings = []
if ksel is not None:
    best_lvl = ksel.loc[ksel["silhouette_level"].idxmax()]
    best_shp = ksel.loc[ksel["silhouette_shape"].idxmax()]
    findings.append(
        f"- **No split passes the test.** The best score by overall level is "
        f"**{best_lvl['silhouette_level']:.2f}** (with {int(best_lvl['k'])} groups); by pattern "
        f"it is **{best_shp['silhouette_shape']:.2f}**. Both are below 0.25."
    )
    more = ksel[ksel["k"] > best_lvl["k"]]
    if not more.empty and (more["n_singletons_level"] > 0).any():
        findings.append(
            "- **More groups only make it worse.** Every split beyond "
            f"{int(best_lvl['k'])} starts peeling off single departments, and the score keeps falling."
        )
if prof is not None and "OVERALL" in prof.columns and len(prof) >= 2:
    themes = [c for c in prof.columns if c not in (prof.columns[0], "OVERALL", "n_departments")]
    p = prof.set_index(prof.columns[0])
    lvl_gap = abs(p["OVERALL"].iloc[1] - p["OVERALL"].iloc[0])
    centred = p[themes].sub(p["OVERALL"], axis=0)
    shape_gap = (centred.iloc[1] - centred.iloc[0]).abs().max()
    findings.append(
        f"- **The two groups differ in level, not in pattern.** Their average scores are "
        f"{p['OVERALL'].min():.2f} and {p['OVERALL'].max():.2f}, a gap of {lvl_gap:.2f}. "
        f"Once that is taken out, their theme-by-theme profiles differ by at most "
        f"{shape_gap:.2f}. One group is simply lower across the board."
    )
st.markdown(
    "**What we found:**\n\n" + "\n".join(findings) + "\n\n"
    "**So:** there are no distinct kinds of department here. The split is a ranking of who "
    "needs attention first, not a set of types needing different plans. Reporting the weaker "
    "claim is the honest reading."
)

# ====================================================================== 4. IPMA
st.subheader("Where to focus first (importance × performance)")
anchor_list = " and ".join(q(a) for a in ANCHORS) if ANCHORS else "the two overall questions"
st.markdown(
    f"""
**How this map is built:** Each theme gets two numbers:

- **Performance:** the theme's company average. How well it scores today.
- **Importance:** how closely the theme rises and falls with two overall questions,
  {anchor_list}, across the 19 departments. If departments that score well on a theme also
  score well on those two questions, the theme counts as important.

Both are split at their median, giving four boxes. **Concentrate here** (top left) is the one
that matters: important, but scoring low today.
"""
)

ipma = load_table("tab09_ipma")
if ipma is not None:
    figure("fig04c_ipma_map")
    iv = pd.DataFrame({
        "Theme": ipma["category"].map(lambda t: short_theme(t, limit=40)),
        "Importance": ipma["importance"],
        "Performance": ipma["performance"],
        "Priority": ipma["priority"].map({True: "Concentrate here", False: ""}),
        "Importance before fix": ipma["importance_naive"],
    })
    st.dataframe(iv, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%.2f")
                                for c in ["Importance", "Performance", "Importance before fix"]})

    perf_med = ipma["performance"].median()
    pri = ipma[ipma["priority"]]
    lines = []
    if not pri.empty:
        names = ", ".join(f"*{short_theme(r['category'], 40)}* ({r['importance']:.2f} / "
                          f"{r['performance']:.2f})" for _, r in pri.iterrows())
        lines.append(f"- **{len(pri)} themes land in Concentrate here:** {names}.")
        edge = pri.assign(m=(perf_med - pri["performance"]).abs()).sort_values("m").iloc[0]
        if edge["m"] < 0.05:
            lines.append(
                f"- **One of them is on the edge:** *{short_theme(edge['category'], 40)}* sits only "
                f"{edge['m']:.2f} below the performance line, so a small change could move it "
                "out of the box. Treat it as borderline."
            )
    adj = ipma[ipma["anchor_adjusted"]]
    for _, r in adj.iterrows():
        lines.append(
            f"- **One correction is built in:** The two overall questions sit inside "
            f"*{short_theme(r['category'], 40)}*, so that theme was partly being compared with itself. "
            f"Scoring it without them moves its importance from {r['importance_naive']:.2f} to "
            f"{r['importance']:.2f} (the arrow on the map). It stays in the same box."
        )
    lo_imp = ipma.sort_values("importance").iloc[0]
    lines.append(
        f"- **Low importance is not 'unimportant':** *{short_theme(lo_imp['category'], 40)}* scores "
        f"lowest ({lo_imp['importance']:.2f}). That usually means it does not separate departments "
        "from each other, not that it does not matter."
    )
    st.markdown("**What the map shows:**\n\n" + "\n".join(lines))
    st.caption(
        "Read this map as a prioritisation aid, not a finding. Importance is a correlation "
        "across about 19 departments, so it shows what moves together, not what causes what. "
        "Page 3 tests real drivers on individual-level data."
    )
