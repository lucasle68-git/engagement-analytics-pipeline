"""Current State Explorer — "How does each department compare?" (survey data, Phase 1).

The company-level theme ranking lives on the Home page. This page goes one level
down: pick a department and see where it sits above or below the company, theme by
theme. Everything is read from pipeline outputs; nothing is recomputed.
"""

from pathlib import Path
import math
import sys

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _shared import (figure, key_takeaways, load_table, require,  # noqa: E402
                     short_theme, survey_badge)

st.set_page_config(page_title="Current State", page_icon="📊", layout="wide")

BELOW, ABOVE = "#c0392b", "#1f77b4"      # red = under the company, blue = over
GAP_LINE = -0.20                          # a theme counts as "below" past this gap
STANDOUT_DEPTH, STANDOUT_MARGIN = -0.50, 0.40


def profile(row: pd.Series) -> tuple[str, str]:
    """Label the SHAPE of a department's gaps, in the terms used in NB02.

    - Above average: ahead of the company almost everywhere.
    - Hot spot: one theme sits far below the department's own typical gap.
    - Broad gap: most themes are below the company together.
    - Both can apply at once; "Mixed" covers everything else.
    """
    deepest = row.min()
    below = int((row < GAP_LINE).sum())
    standout = deepest <= STANDOUT_DEPTH and (deepest - row.median()) <= -STANDOUT_MARGIN
    broad = below >= 5
    worst = short_theme(row.idxmin(), limit=40)

    if row.mean() >= 0.10 and deepest > GAP_LINE:
        return ("Above average",
                "This department sits above the company on almost every theme. Treat it as "
                "an internal benchmark: the useful question is *how* it does this.")
    if broad and standout:
        return ("Broad gap + hot spot",
                f"Most themes ({below} of {len(row)}) sit below the company, and "
                f"**{worst}** falls well below even that. Two problems at once: a "
                "department-wide climate issue, and one area that needs its own owner.")
    if standout:
        return ("Hot spot",
                f"One theme, **{worst}**, sits far below the rest of this department's "
                "profile. That is a local problem with a named owner, not a department-wide one.")
    if broad:
        return ("Broad gap",
                f"{below} of {len(row)} themes sit below the company, with no single theme "
                "standing out. That points to a department-wide climate issue rather than "
                "one broken area.")
    return ("Mixed",
            "Some themes sit above the company and some below, with no large gap. Nothing "
            "here calls for urgent action; watch the lowest bar over time.")


st.title("Current State Explorer")
survey_badge()

_gp = load_table("tab04_gap_matrix")
_tg = load_table("tab05_top_gaps")
_rk1 = load_table("tab03_theme_ranking")
if _gp is not None:
    _g = _gp.set_index(_gp.columns[0])
    _hi, _lo = _g.mean(axis=1).idxmax(), _g.mean(axis=1).idxmin()
    _pts = [
        f"**A department tends to be strong or weak across the board.** Department "
        f"**{_hi}** is above the company average on **{int((_g.loc[_hi] > 0).sum())} of "
        f"{_g.shape[1]}** themes; department **{_lo}** is below on "
        f"**{int((_g.loc[_lo] < 0).sum())} of {_g.shape[1]}**. The gap runs along "
        f"departments, not along topics."
    ]
    if _tg is not None and not _tg.empty:
        _w = _tg.loc[_tg["gap"].idxmin()]
        _tm = ""
        if _rk1 is not None:
            _hit = _rk1[_rk1.iloc[:, 0] == _w["category"]]
            if not _hit.empty:
                _tm = (f", a theme whose company average is "
                       f"{float(_hit['mean'].iloc[0]):.2f}")
        _pts.append(
            f"**Theme averages hide single questions.** The largest single gap is "
            f"**{_w['gap']:+.2f}**, department {_w['department']} on one question inside "
            f"*{short_theme(_w['category'])}*{_tm}. Averaging makes a gap like this disappear."
        )
    _pts.append(
        "**The label tells you which fix applies.** A *broad gap* is a department-level "
        "conversation; a *hot spot* is one specific question to chase. Different problems, "
        "different responses."
    )
    key_takeaways(*_pts)
st.markdown(
    "The company-level theme ranking is on the **Home** page. This page looks one level "
    "down: how each department compares with the company, theme by theme."
)

gaps = load_table("tab04_gap_matrix")
require(gaps)
gm = gaps.set_index(gaps.columns[0])

# ------------------------------------------------------------------ department view
st.subheader("A department against the company average")
st.markdown(
    "Pick a department to see how it scores on each of the eight themes, measured as a "
    "**gap to the company average**. Red bars are themes where it sits below the company, "
    "blue bars where it sits above. The line above the chart gives its overall score and "
    "its **rank among all departments**, so you can see at once whether it is a leader, "
    "in the middle, or near the bottom and which theme pulls it down most."
)

composite_all = load_table("tab10_composite_index")
if composite_all is not None:
    with st.expander("See the full department ranking"):
        rk = composite_all[composite_all.iloc[:, 0] != "Company"].copy()
        rk = rk.sort_values("composite_index", ascending=False).reset_index(drop=True)
        rk.insert(0, "Rank", range(1, len(rk) + 1))
        rk.columns = ["Rank", "Department", "Overall score (1–5)", "Gap to company"]
        st.dataframe(rk, hide_index=True, use_container_width=True,
                     column_config={
                         "Overall score (1–5)": st.column_config.NumberColumn(format="%.2f"),
                         "Gap to company": st.column_config.NumberColumn(format="%+.2f"),
                     })
        st.caption("Overall score = the average of a department's eight theme scores.")

dept = st.selectbox("Choose a department", gm.index)
row = gm.loc[dept]

# one-line summary: composite, rank, deepest gap
composite = load_table("tab10_composite_index")
summary = []
if composite is not None:
    ci = composite.set_index(composite.columns[0])["composite_index"]
    company = ci.get("Company")
    depts = ci.drop("Company", errors="ignore").sort_values(ascending=False)
    if dept in depts.index and company is not None:
        rank = list(depts.index).index(dept) + 1
        summary.append(f"composite **{depts[dept]:.2f}** vs company **{company:.2f}**")
        summary.append(f"ranks **{rank} of {len(depts)}**")
if row.min() < 0:
    summary.append(f"deepest gap: **{short_theme(row.idxmin(), limit=40)} ({row.min():+.2f})**")
else:
    summary.append(f"ahead on every theme; smallest lead: "
                   f"**{short_theme(row.idxmin(), limit=40)} ({row.min():+.2f})**")
st.markdown(f"**{dept}** · " + " · ".join(summary))

label, explanation = profile(row)
st.info(f"**Profile: {label}.** {explanation}")

# diverging horizontal bars, sorted, fixed axis across every department
lo = math.floor(gm.min().min() * 10) / 10 - 0.1
hi = math.ceil(gm.max().max() * 10) / 10 + 0.1
chart_df = pd.DataFrame({
    "theme": [short_theme(t, limit=40) for t in row.index],
    "gap": row.values,
})
chart_df["side"] = chart_df["gap"].map(lambda g: "Below company" if g < 0 else "Above company")
order = chart_df.sort_values("gap")["theme"].tolist()

bars = alt.Chart(chart_df).mark_bar(size=22).encode(
    y=alt.Y("theme:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
    x=alt.X("gap:Q", scale=alt.Scale(domain=[lo, hi], nice=False),
            title="Gap to company average (points on a 1–5 scale)"),
    color=alt.Color("side:N", scale=alt.Scale(domain=["Below company", "Above company"],
                                              range=[BELOW, ABOVE]),
                    legend=alt.Legend(title=None, orient="bottom")),
    tooltip=[alt.Tooltip("theme:N", title="Theme"),
             alt.Tooltip("gap:Q", title="Gap", format="+.2f")],
)
labels = alt.Chart(chart_df).mark_text(fontSize=12, dx=alt.expr("datum.gap < 0 ? -22 : 22")).encode(
    y=alt.Y("theme:N", sort=order),
    x="gap:Q",
    text=alt.Text("gap:Q", format="+.2f"),
)
zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeWidth=1.5, color="#888").encode(x="x:Q")
zero_label = alt.Chart(pd.DataFrame({"x": [0], "t": ["company average"]})).mark_text(
    align="left", dx=4, dy=-8, color="#888", fontSize=11
).encode(x="x:Q", y=alt.value(0), text="t:N")

st.altair_chart((bars + labels + zero + zero_label).properties(height=320),
                use_container_width=True)
st.caption(
    "The axis is the same for every department, so bar lengths can be compared as you "
    "switch between them. Zero is the company average."
)

with st.expander("How the profile label is decided"):
    st.markdown(
        f"""
- **Above average:** the department's average gap is +0.10 or more, and no theme is
  more than {abs(GAP_LINE):.2f} below the company.
- **Hot spot:** the deepest theme is at least {abs(STANDOUT_DEPTH):.2f} below the company
  *and* at least {STANDOUT_MARGIN:.2f} below the department's own typical (median) gap.
- **Broad gap:** 5 or more of the 8 themes are more than {abs(GAP_LINE):.2f} below the company.
- **Broad gap + hot spot:** both of the above.
- **Mixed:** none of the above.

These are simple reading aids, not statistical tests. They put a name on the two
patterns described in notebook 02: one deep problem with a clear owner, versus a
department that is below the company on most things.
"""
    )

st.markdown("**Every department by profile**")
LABEL_ORDER = ["Above average", "Broad gap", "Broad gap + hot spot", "Hot spot", "Mixed"]
by_label = {lab: [] for lab in LABEL_ORDER}
for d, r in gm.iterrows():
    by_label[profile(r)[0]].append(str(d))
profile_table = pd.DataFrame({
    "Profile": LABEL_ORDER,
    "Departments": [", ".join(by_label[lab]) or "—" for lab in LABEL_ORDER],
    "Count": [len(by_label[lab]) for lab in LABEL_ORDER],
})
st.dataframe(profile_table, hide_index=True, use_container_width=True)
st.caption(f"Department **{dept}**, selected above, is in the **{label}** row.")

# ------------------------------------------------------------------ full grid
st.subheader("The full department × theme grid")
figure("fig02_department_gap_heatmap")
st.markdown(
    "**What this means:** Each row is a department, each column a theme. Read along a row "
    "to see whether a department is weak everywhere or in one place. Departments that "
    "score well are internal benchmarks: colleagues to ask *how*, not a league table."
)

# ------------------------------------------------------------------ item level
top = load_table("tab05_top_gaps")
if top is not None:
    st.subheader("Largest individual gaps")

    # theme-level gap for the same department and theme, looked up from tab04, so each
    # question can be read against the average it belongs to
    top = top.copy()
    top["theme_gap"] = [gm.loc[d, c] if (d in gm.index and c in gm.columns) else float("nan")
                        for d, c in zip(top["department"], top["category"])]
    top["hidden_by_average"] = top["gap"] - top["theme_gap"]

    pos, neg = top[top["gap"] > 0], top[top["gap"] < 0]
    lead_pos = pos["department"].value_counts()
    lead_neg = neg["department"].value_counts()

    view = top[["department", "category", "item", "gap", "theme_gap", "hidden_by_average"]].copy()
    view["category"] = view["category"].map(lambda t: short_theme(t, limit=40))
    view.columns = ["Department", "Theme", "Question", "Question gap",
                    "Theme gap", "Hidden by the average"]
    st.dataframe(view, hide_index=True, use_container_width=True,
                 column_config={c: st.column_config.NumberColumn(format="%+.2f")
                                for c in ["Question gap", "Theme gap", "Hidden by the average"]})
    st.caption(
        "Question gap = the department's score on that one question minus the company's. "
        "Theme gap = the same, averaged over the whole theme. Hidden by the average = how "
        "much further the question falls than its theme suggests."
    )

    lines = []
    if not lead_pos.empty:
        d, n = lead_pos.index[0], int(lead_pos.iloc[0])
        lines.append(
            f"- **The strengths cluster in one place:** {n} of the {len(pos)} largest positive "
            f"gaps belong to department **{d}**. That makes {d} the internal benchmark to learn from."
        )
    if not lead_neg.empty:
        d, n = lead_neg.index[0], int(lead_neg.iloc[0])
        worst = neg.sort_values("gap").iloc[0]
        rank_note = ""
        if composite_all is not None:
            rk_ = composite_all[composite_all.iloc[:, 0] != "Company"]
            rk_ = rk_.sort_values("composite_index", ascending=False).iloc[:, 0].tolist()
            if d in rk_:
                rank_note = f" It also ranks {rk_.index(d) + 1} of {len(rk_)} overall."
        lines.append(
            f"- **So do the problems:** {n} of the {len(neg)} largest negative gaps belong to "
            f"department **{d}**, down to {worst['gap']:+.2f} on "
            f"\"{str(worst['item']).rstrip('.')}\".{rank_note}"
        )

    ranking = load_table("tab03_theme_ranking")
    best_theme = ranking.iloc[0, 0] if ranking is not None else None
    in_best = neg[neg["category"] == best_theme] if best_theme else neg.iloc[0:0]
    if not in_best.empty:
        r = in_best.sort_values("gap").iloc[0]
        lines.append(
            f"- **Big gaps hide inside the best theme:** *{short_theme(best_theme, limit=40)}* "
            f"is the company's highest-scoring theme, yet department **{r['department']}** "
            f"sits {r['gap']:+.2f} on \"{str(r['item']).rstrip('.')}\". That is "
            f"{r['hidden_by_average']:+.2f} further than its theme gap of {r['theme_gap']:+.2f}."
        )

    # a gap the top-gaps table cannot show: a deep question inside a theme that looks fine
    try:
        import pandas as _pd
        from _shared import ROOT
        mat = _pd.read_csv(ROOT / "data" / "processed" / "dept_item_matrix.csv", index_col=0)
        disp = load_table("tab06_dispersion_screen")
        if disp is not None and "Company" in mat.index:
            item_theme = dict(zip(disp["item"], disp["category"]))
            item_gap = mat.drop(index="Company").sub(mat.loc["Company"], axis=1)
            cands = []
            for d_, r_ in item_gap.iterrows():
                for q, v in r_.items():
                    th = item_theme.get(q)
                    if th in gm.columns and d_ in gm.index:
                        tg = gm.loc[d_, th]
                        if v <= -0.5 and abs(tg) < 0.25:
                            cands.append((v - tg, d_, th, q, v, tg))
            if cands:
                _, d_, th, q, v, tg = sorted(cands)[0]
                lines.append(
                    f"- **Some gaps never reach this table:** Department **{d_}** looks close to "
                    f"the company on *{short_theme(th, limit=40)}* ({tg:+.2f}), yet one question in "
                    f"that theme, \"{str(q).rstrip('.')}\", sits at {v:+.2f}. A theme average of "
                    f"{tg:+.2f} would never flag it."
                )
    except Exception:
        pass

    st.markdown(
        "**Why item level, not theme level:** A theme score averages several questions, so one "
        "weak question can disappear inside it. The numbers above show this happening:\n\n"
        + "\n".join(lines)
        + "\n\nAveraging hides exactly the questions an intervention would target."
    )
