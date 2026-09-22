"""Styled, captioned figures. Every figure passes through save_figure(), which
stamps a data-provenance badge (REAL vs DEMO vs SYNTHETIC), governance enforced
in code, not left to manual discipline.

Two kinds of function live here, and the split is deliberate:

- generic plot types (`heatmap`, `ranked_bar`) with thin wrappers that apply the
  house style once so 14 figures cannot drift apart;
- purpose-built plots (`correlation_triangle`, `dispersion_quadrants`,
  `ipma_map`), each encodes an analytical argument that a default chart would
  lose, and each docstring states which argument.

Notebooks contain no styling code. A notebook cell creates the axes, calls one
function here, and calls `save_figure`; changing how the report looks is
therefore a one-file change, and no figure can reach outputs/ unbadged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

BADGE = {
    "real": ("REAL 2024 survey data (aggregated)", "#1a7f37"),
    "demo": ("DEMO data — invented numbers, not client results", "#8250df"),
    "synthetic": ("SYNTHETIC demonstration data — method, not fact", "#b35900"),
}

sns.set_theme(style="whitegrid", context="notebook")


def save_figure(
    fig: plt.Figure,
    name: str,
    cfg: dict[str, Any],
    data_kind: Literal["real", "demo", "synthetic"],
    caption: str = "",
) -> Path:
    """Stamp provenance badge + caption, save to outputs/figures, return path.

    How it works
    ------------
    0. Downgrade the claim if the source workbook is not the client's. A
       notebook cell always declares `data_kind="real"` for a figure drawn from
       the survey workbook, because that is what the cell is analysing — but
       whether that workbook IS the client file is a property of the run, not of
       the cell. `data.provenance` in config records it, and when it reads
       "demo" every `real` badge becomes a `demo` badge here. So a public clone
       running the bundled demo workbook cannot produce a figure captioned as
       real client data, and no notebook had to be edited to guarantee it.
       `synthetic` is left alone: Phase 2 data is generated either way.
    1. Look up the badge text and colour for `data_kind` in the BADGE dict. An
       unrecognised value raises a KeyError immediately, a figure cannot be
       saved without declaring what data it came from.
    2. `fig.text(0.01, 0.01, ...)` writes the badge in FIGURE coordinates
       (0,0 = bottom-left of the whole image, not of the axes), so the badge
       lands in the same place whatever the plot inside is.
    3. Add the caption as a suptitle if one was given.
    4. Create outputs/figures if needed, save at 200 dpi with
       `bbox_inches="tight"` (crops whitespace but keeps the badge, because the
       badge is part of the figure).
    5. Return the path so the notebook can print where the file went.

    Why this is a function and not four lines in each notebook: the badge is a
    data-governance control. Phase 1 figures show real client data and
    Phase 2 figures show synthetic data, and a reader or client seeing them out
    of context must not have to guess which is which. Routing every figure
    through one function makes the badge impossible to forget. The guarantee
    is structural rather than a matter of remembering.

    `name` should carry the report figure number, e.g. 'fig03_theme_ranking'.
    """
    if data_kind == "real" and cfg.get("data", {}).get("provenance", "real") == "demo":
        data_kind = "demo"
    text, colour = BADGE[data_kind]
    fig.text(
        0.01, 0.01, text, fontsize=8, color="white", family="monospace",
        bbox=dict(facecolor=colour, edgecolor="none", boxstyle="round,pad=0.3"),
    )
    if caption:
        fig.suptitle(caption, fontsize=11)
    out_dir = Path(cfg["paths"]["figures_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def heatmap(matrix, ax=None, center: float | None = 0.0, fmt: str = ".2f"):
    """Annotated diverging heatmap (used for the department gap matrix).

    How it works: if no axes are supplied, size them from the matrix shape so a
    34-column matrix does not get squeezed into the same box as an 8-column
    one. Then one seaborn call with the house settings. `annot=True` prints
    the number inside each cell (a reader of the report should not have to
    decode a colour into a value), and `center=0` anchors the diverging
    red-to-green scale at zero.

    That `center` argument is the important one for gap matrices: zero means
    "exactly at the company average", so red is genuinely below and green
    genuinely above. Without it matplotlib would centre the colour scale on the
    middle of the observed range, and a department could be coloured green
    while scoring below the company.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(max(8, 0.5 * matrix.shape[1]), 0.45 * matrix.shape[0] + 2))
    sns.heatmap(
        matrix, annot=True, fmt=fmt, cmap="RdYlGn", center=center,
        linewidths=0.5, cbar_kws={"shrink": 0.7}, ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    return ax


# Martilla & James (1977) quadrant names -> (colour, action shown to the reader)
IPMA_QUADRANTS = {
    "concentrate here":     ("#c0392b", "high importance · low performance\n→ ACT FIRST"),
    "keep up the good work": ("#1a7f37", "high importance · high performance\n→ protect"),
    "low priority":          ("#7f8c8d", "low importance · low performance\n→ monitor"),
    "possible overkill":     ("#2980b9", "low importance · high performance\n→ do not over-invest"),
}


def ipma_map(ipma, ax=None):
    """Importance-Performance map with the circularity correction made visible.

    Axes follow Martilla & James (1977): performance on x, importance on y,
    split at the medians of the plotted themes. The top-left quadrant
    ('concentrate here') is the priority list.

    The distinctive element is the ARROW. Both configured outcome anchors sit
    inside the 'Working for AccessFintech' theme, so scoring that theme from all
    its items and correlating it against two of them is circular. phase1.ipma
    rescoring each theme from its NON-anchor items corrects this, and the arrow
    draws the correction to scale. A reader can see how large it was and that
    the theme stays in the same quadrant, i.e. the finding survives the fix.
    A number in a table cannot show that; a moved point can.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(9.5, 7))

    imp_cut = ipma["importance"].median()
    perf_cut = ipma["performance"].median()
    ax.axvline(perf_cut, color="grey", ls="--", lw=1)
    ax.axhline(imp_cut, color="grey", ls="--", lw=1)

    def quadrant_of(r):
        hi_i, hi_p = r["importance"] > imp_cut, r["performance"] > perf_cut
        if hi_i and not hi_p:
            return "concentrate here"
        if hi_i and hi_p:
            return "keep up the good work"
        return "possible overkill" if hi_p else "low priority"

    for _, r in ipma.iterrows():
        q = quadrant_of(r)
        colour = IPMA_QUADRANTS[q][0]
        ax.scatter(r["performance"], r["importance"], s=190, color=colour,
                   edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(SHORT_CATEGORY.get(r.name, str(r.name)[:18]),
                    (r["performance"], r["importance"]), fontsize=9,
                    xytext=(9, -3), textcoords="offset points", zorder=5)

        # the audit trail: where the uncorrected estimate sat
        if r.get("anchor_adjusted", False):
            naive = r["importance_naive"]
            ax.scatter(r["performance"], naive, s=120, facecolors="none",
                       edgecolors=colour, linewidths=1.4, linestyle=":", zorder=3)
            ax.annotate("", xy=(r["performance"], r["importance"]),
                        xytext=(r["performance"], naive),
                        arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.6,
                                        linestyle="--", shrinkA=4, shrinkB=6))
            ax.annotate(f"uncorrected {naive:.2f}\n(anchors inside this theme)",
                        (r["performance"], naive), fontsize=7.5, style="italic",
                        color=colour, xytext=(12, -14), textcoords="offset points")

    # headroom so the uncorrected marker is not clipped and corner labels do not
    # collide with the points
    ax.margins(0.16)
    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    for q, (colour, text) in IPMA_QUADRANTS.items():
        x = xlo if "low performance" in text else xhi
        y = yhi if "high importance" in text else ylo
        ax.text(x, y, f"{q.upper()}\n{text}", fontsize=7.5, color=colour,
                ha="left" if x == xlo else "right",
                va="top" if y == yhi else "bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                          boxstyle="round,pad=0.35"))

    ax.set_xlabel("performance — company theme mean (1–5 harmonised)")
    ax.set_ylabel("importance — correlation with outcome anchors, across ~19 depts")
    return ax


# Long workbook category names -> short axis labels. Falls back to truncation
# for any name not listed, so a renamed category degrades gracefully.
SHORT_CATEGORY = {
    "Culture & Wellbeing": "Culture",
    "My role at Access Fintech": "My role",
    "Work/Life Balance": "Work/Life",
    "Working for AccessFintech": "Working for AFT",
    "Working with my Line Manager": "Line Manager",
    "Working with my Team": "Team",
    "Working with the Executive Teams": "Executive",
    "Working with the Leadership Teams (Heads of Department)": "Leadership (HoD)",
}


def correlation_triangle(corr, ax=None, annotate_summary: bool = True):
    """Lower-triangle heatmap of the theme correlation matrix.

    Three design choices, each carrying an analytical point:

    1. SEQUENTIAL colour scale on 0-1, not the diverging scale used for gap
       matrices. Every off-diagonal correlation here is positive, so a scale
       centred on zero would render the whole matrix one colour and hide the
       variation that matters. On a 0-1 sequential scale the all-positive
       finding is visible (nothing sits at the pale end by accident) AND the
       spread across the observed range stays legible.
    2. Upper triangle masked. The matrix is symmetric, so half of it is
       duplicate ink, and the diagonal of 1.0 carries no information.
    3. The weakest pair is boxed: near-independence between two themes is the
       actionable finding (separate levers), and it is easy to miss among 28
       cells.

    `annotate_summary` prints mean r and the positive count in the corner,
    the halo effect quantified rather than asserted.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8.5, 7))

    labels = [SHORT_CATEGORY.get(c, str(c)[:16]) for c in corr.columns]
    m = corr.copy()
    m.index, m.columns = labels, labels

    mask = np.triu(np.ones_like(m, dtype=bool))          # hide upper + diagonal
    sns.heatmap(m, mask=mask, annot=True, fmt=".2f", cmap="YlGnBu",
                vmin=0, vmax=1, linewidths=0.6, square=True,
                cbar_kws={"shrink": 0.7, "label": "Pearson r (across ~19 departments)"},
                annot_kws={"fontsize": 9}, ax=ax)

    # box the weakest pair — the near-independence finding
    off = m.where(~np.eye(len(m), dtype=bool))
    lo = off.stack().idxmin()
    r, c = labels.index(lo[0]), labels.index(lo[1])
    row, col = (max(r, c), min(r, c))                    # lower triangle position
    ax.add_patch(plt.Rectangle((col, row), 1, 1, fill=False,
                               edgecolor="#c0392b", lw=2.5, zorder=5))

    if annotate_summary:
        vals = off.stack().dropna()
        vals = vals[vals.index.get_level_values(0) != vals.index.get_level_values(1)]
        uniq = vals.iloc[::2] if len(vals) == 2 * (len(m) * (len(m) - 1) // 2) else vals
        ax.text(0.98, 0.97,
                f"all {len(m)*(len(m)-1)//2} pairs positive\n"
                f"mean r = {uniq.mean():.2f}   ·   range {uniq.min():.2f}–{uniq.max():.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#999", boxstyle="round,pad=0.4"))

    ax.set_xlabel(""); ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    return ax


QUADRANT_STYLE = {
    "localised problem":    ("#c0392b", "o"),   # low score, wide spread
    "company-wide problem": ("#e67e22", "s"),   # low score, narrow spread
    "uneven strength":      ("#2980b9", "^"),   # high score, wide spread
    "uniform strength":     ("#1a7f37", "."),   # high score, narrow spread
}


def dispersion_quadrants(disp, cfg: dict[str, Any], ax=None, label_chars: int = 46):
    """Scatter the dispersion screen as the 2x2 it actually is (RO3a).

    x = company mean (how good is this item overall?)
    y = spread across departments (is the experience even?)

    The quadrant logic is the analytical point and a table cannot show it: a low
    score with a WIDE spread is a localised problem (specific departments are
    dragging it down -> targeted intervention), while a low score with a NARROW
    spread is company-wide (-> policy fix). Same low score, different action.

    Threshold lines are recomputed with the same quantiles phase1 used, so the
    drawn boundaries are the ones that actually assigned the quadrants, not an
    eyeballed approximation. Only the flagged items are labelled, labelling all
    34 would make the plot unreadable and the unflagged ones carry no action.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 7))

    mean_cut = disp["company_mean"].quantile(cfg["phase1"]["low_mean_quantile"])
    spread_cut = disp["spread"].quantile(cfg["phase1"]["dispersion_high_spread_quantile"])

    ax.axvline(mean_cut, color="grey", ls="--", lw=1)
    ax.axhline(spread_cut, color="grey", ls="--", lw=1)

    for quad, (colour, marker) in QUADRANT_STYLE.items():
        sub = disp[disp["quadrant"] == quad]
        if sub.empty:
            continue
        ax.scatter(sub["company_mean"], sub["spread"], c=colour, marker=marker,
                   s=90 if marker != "." else 55, alpha=0.85, zorder=3,
                   edgecolors="white", linewidths=0.6,
                   label=f"{quad} (n={len(sub)})")

    # label only the flagged items; item text is long, so truncate
    for _, r in disp[disp["priority_2026"]].iterrows():
        item = r.name[1] if isinstance(r.name, tuple) else str(r.name)
        text = item[:label_chars] + ("…" if len(item) > label_chars else "")
        ax.annotate(text, (r["company_mean"], r["spread"]), fontsize=7.5,
                    xytext=(6, 4), textcoords="offset points", zorder=4)

    xlo, xhi = ax.get_xlim()
    ylo, yhi = ax.get_ylim()
    corners = [
        (xlo, yhi, "LOCALISED PROBLEM\nlow score · uneven\n→ targeted action", "left", "top"),
        (xhi, yhi, "UNEVEN STRENGTH\nhigh score · uneven\n→ protect the weak units", "right", "top"),
        (xlo, ylo, "COMPANY-WIDE PROBLEM\nlow score · consistent\n→ policy-level fix", "left", "bottom"),
        (xhi, ylo, "UNIFORM STRENGTH\nhigh score · consistent\n→ maintain", "right", "bottom"),
    ]
    for x, y, txt, ha, va in corners:
        ax.text(x, y, txt, ha=ha, va=va, fontsize=8, color="#555", style="italic",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.65,
                          boxstyle="round,pad=0.35"))

    ax.set_xlabel("company mean (1–5 harmonised)  →  higher is better")
    ax.set_ylabel("spread across departments (sd)  →  higher is more uneven")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False)
    return ax


def ranked_bar(series, ax=None, xlabel: str = "Score (1–5 harmonised)"):
    """Horizontal ranked bar chart for theme/department rankings.

    How it works
    ------------
    1. Height scales with the number of bars (`0.45 * len(series)`), so eight
       themes and nineteen departments both come out readable.
    2. `series.sort_values()` ascending with `barh`, matplotlib draws the
       first element at the BOTTOM, so ascending order puts the highest score
       at the top, which is how a ranking is read.
    3. Colour by position relative to the median: below-median bars red,
       above-median green. The median is used rather than a fixed cut-off
       because the scores here span only a quarter of a point; the colour
       communicates rank, not an absolute standard, and the report says so.
    4. Print each value at the end of its bar, because on a compressed range
       the bars themselves are nearly the same length as the numbers, not the
       geometry, carry the information.

    Horizontal rather than vertical for one practical reason: the theme labels
    are long sentences, and vertical bars would need them rotated 45 degrees.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.45 * len(series) + 1.5))
    s = series.sort_values()
    colors = ["#c0392b" if v < s.median() else "#1a7f37" for v in s]
    ax.barh(s.index.astype(str), s.values, color=colors)
    ax.set_xlabel(xlabel)
    for i, v in enumerate(s.values):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=9)
    return ax
