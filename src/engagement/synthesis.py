"""Phase 3 — synthesis: Phase 1 vs Phase 2 comparison -> 2026 survey + actions.

Implements methodology §4.4 and nothing more:
  §4.4.1 capability_comparison: what Phase 1 could answer vs what Phase 2
          unlocked, each gap mapped to a specific missing data element (RO4
          evidence base).
  §4.4.3 importance_performance: Phase 2 driver importance (relative weights,
          synthetic) x Phase 1 real 2024 performance -> priority interventions.
          The headline is structural: the TOP drivers are constructs the 2024
          survey does not measure at all.
  §4.4.2 survey_redesign: keep / add / drop / fix table, every row
          traced to an analytical result or literature factor.
  §4.4.4 traceability: deliverable -> refined objective -> client objective.
        

Claim boundary: importance comes from the synthetic demonstration (method
capability); performance comes from real 2024 data. The two are combined only
to PRIORITISE, never to assert new facts about the workforce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Phase-2 construct -> 2024 survey categories that (partially) measure it.
# Empty list = the construct is NOT measured by the 2024 instrument.
CATEGORIES_OF_CONSTRUCT = {
    "personal": [],
    "development": ["My role at Access Fintech"],
    "social": ["Working with my Line Manager", "Working with my Team",
               "Culture & Wellbeing"],
    "balance": ["Work/Life Balance"],
    "justice": ["Working with the Leadership Teams (Heads of Department)",
                "Working with the Executive Teams"],
    "technology": [],
    "psych_safety": [],
    "esg": [],
}


def importance_performance(drivers: pd.DataFrame,
                           theme_means: pd.Series,
                           dispersion: pd.DataFrame | None = None) -> pd.DataFrame:
    """Driver importance (Phase 2, synthetic) x current performance (Phase 1, real).

    Parameters
    ----------
    drivers : output of models.driver_analysis (index = construct,
        column 'relative_weight_%').
    theme_means : real 2024 harmonised category means
        (phase1.theme_ranking(...)['mean']).
    dispersion : optional phase1.dispersion_screen output (MultiIndex
        category/item). Theme-mean performance averages a whole theme, and
        across the constructs the 2024 survey measures it typically spans a
        fraction of a scale point, too flat an axis to separate anything. The
        item-level screen restores the resolution that the average hides: how
        many of the theme's items are flagged priority-risk, and how far the
        weakest one falls.

    Notes
    -----
    ALL FOUR cells of the 2x2 are named. An earlier version named only the
    'improve' cell and collapsed the other three into 'maintain', producing a
    table that could not be read without the figure beside it.

    `.attrs` records the honest diagnostics of the map itself:
      rank_alignment_spearman: how importance and performance rank against
        each other across measured constructs. At rho = 1.0 the top-half of
        importance and the bottom-half of performance are disjoint SETS, so the
        IMPROVE cell is empty by arithmetic necessity rather than by chance.
      improve_cell_empty_by_construction: that disjointness, stated directly.
      performance_span. The width of the performance axis, so a degenerate
        axis is visible rather than implied.
    These are findings to report, not results to tune away.

    How it works
    ------------
    1. For each construct, look up which 2024 categories (if any) measure it in
       `CATEGORIES_OF_CONSTRUCT`. An empty list is not missing data, it is the
       finding that the instrument never asked.
    2. Performance = mean of those categories' real 2024 theme means; NaN where
       the construct is unmeasured. `measured_2024` records which case applies.
    3. If a dispersion screen is supplied, add item-level resolution per
       construct: how many of its items are flagged priority, the weakest item
       and its score. This is what stops a comfortable theme mean from hiding a
       much weaker item inside it.
    4. Medians are taken over the MEASURED constructs only, an unmeasured
       construct has no performance value and must not shift the axis.
    5. `np.select` labels all four cells plus 'MEASURE FIRST' for the unmeasured
       ones. Note `lo_perf` is False for NaN, so unmeasured constructs cannot
       fall into a performance-based cell by accident; the first condition
       catches them first anyway.
    6. Record the three diagnostics in `.attrs`.

    Reading order for the output: check `.attrs` BEFORE the quadrant column. If
    the IMPROVE cell is empty because importance and performance rank
    identically, that is arithmetic, not evidence that nothing needs improving
    and the map should be reported that way (essay §5.2).
    """
    rows = []
    for con in drivers.index:
        cats = CATEGORIES_OF_CONSTRUCT.get(con, [])
        perf = float(theme_means.loc[cats].mean()) if cats else np.nan
        rows.append({"construct": con,
                     "importance_%": float(drivers.loc[con, "relative_weight_%"]),
                     "performance_2024": round(perf, 3) if cats else np.nan,
                     "measured_2024": bool(cats)})
    out = pd.DataFrame(rows).set_index("construct")

    # ---- item-level resolution ---------------------------------------------
    if dispersion is not None:
        d = dispersion.reset_index()
        flagged, worst_txt, worst_val = {}, {}, {}
        for con in out.index:
            block = d[d["category"].isin(CATEGORIES_OF_CONSTRUCT.get(con, []))]
            if block.empty:
                flagged[con], worst_txt[con], worst_val[con] = np.nan, "", np.nan
                continue
            w = block.loc[block["company_mean"].idxmin()]
            flagged[con] = int(block["priority_2026"].sum())
            worst_txt[con] = str(w["item"])[:70]
            worst_val[con] = round(float(w["company_mean"]), 3)
        out["n_priority_items"] = pd.Series(flagged)
        out["weakest_item_score"] = pd.Series(worst_val)
        out["weakest_item"] = pd.Series(worst_txt)

    # ---- four-cell labelling ------------------------------------------------
    measured = out[out["measured_2024"]]
    imp_med = measured["importance_%"].median()
    perf_med = measured["performance_2024"].median()
    hi_imp = out["importance_%"] > imp_med
    lo_perf = out["performance_2024"] < perf_med          # NaN -> False
    out["quadrant"] = np.select(
        [~out["measured_2024"], hi_imp & lo_perf, hi_imp & ~lo_perf, ~hi_imp & lo_perf],
        ["MEASURE FIRST (not in 2024 survey)",
         "IMPROVE (high importance, low performance)",
         "PROTECT (high importance, high performance)",
         "MONITOR (low importance, low performance)"],
        default="DE-PRIORITISE (low importance, high performance)")

    hi_set = set(measured.index[measured["importance_%"] > imp_med])
    lo_set = set(measured.index[measured["performance_2024"] < perf_med])
    out.attrs["rank_alignment_spearman"] = round(float(
        measured["importance_%"].corr(measured["performance_2024"],
                                      method="spearman")), 3)
    out.attrs["improve_cell_empty_by_construction"] = hi_set.isdisjoint(lo_set)
    out.attrs["performance_span"] = round(float(
        measured["performance_2024"].max() - measured["performance_2024"].min()), 3)
    return out.sort_values("importance_%", ascending=False)


def capability_comparison() -> pd.DataFrame:
    """§4.4.1 the headline table: each client question, what each phase could
    do about it, and the missing data element that made the difference.

    Essay §5.1, Table 9 (tab20).

    Why this is a hard-coded table rather than a computation: the content is an
    argument, not a measurement. What makes it evidence is the fourth column —
    every row names the specific data element whose absence caused the Phase 1
    limit, so a reader can check the claim against the notebook cited beside it.
    Written as code so it is produced with the results and versioned with them;
    NB05 points here instead of restating it, so the two cannot drift apart.
    """
    rows = [
        ["What drives engagement? (O2)",
         "Prohibited: aggregated means, no outcome variable",
         "Ranked, VIF-checked, triangulated driver table (tab16)",
         "Individual responses + UWES outcome block"],
        ["Who is at risk? (O3)",
         "Department-level screening only (9 items flagged, tab06)",
         "Individual risk scores, temporally validated, explainable (tab17-18)",
         "Individual responses + demographics"],
        ["Are the scales sound?",
         "Untestable (no item-level data); scale-mixing defect found (tab01)",
         "Alpha per construct; short-scale weakness exposed (tab15)",
         "Item-level responses"],
        ["Are there segments? (O1/O3)",
         "No archetypes recoverable (silhouette < 0.25, NB03)",
         "Persona machinery with risk & demographic profiles (tab19)",
         "Demographics + department sizes"],
        ["Is engagement moving? (O1)",
         "Single 2024 snapshot",
         "Wave trajectories (fig08)",
         "Multi-wave collection, comparable instrument"],
    ]
    return pd.DataFrame(rows, columns=[
        "client question", "Phase 1 (real 2024 data)",
        "Phase 2 (synthetic demonstration)", "missing data element -> 2026"]
    ).set_index("client question")


def survey_redesign() -> pd.DataFrame:
    """§4.4.2: keep / add / drop-merge / fix, one-line justification each.

    Essay §6.3, Table 10 (tab22).

    Rule applied to every row: the justification must cite either an analytical
    result in this project (a tabNN or a notebook) or a named literature source.
    A recommendation with neither is an opinion, and none is included.

    The shape of the table is itself the finding: six ADDs against one
    DROP/MERGE says the 2024 instrument is incomplete rather than wrong, which
    is the conclusion the importance map reaches independently.
    """
    rows = [
        ["KEEP", "Manager / team / leadership relationship blocks",
         "Strong social-resource coverage; aligns with Saks (2006); Phase 1 shows these are the company's strengths (tab03)"],
        ["KEEP", "eNPS / recommendation and retention items",
         "Comparable outcome anchors across waves (used as IPMA anchors, tab09)"],
        ["ADD", "UWES-9 engagement block (vigour / dedication / absorption)",
         "No direct outcome existed; precondition for any driver or risk model (Schaufeli & Bakker 2004; demonstrated in NB05)"],
        ["ADD", "Personal-resources scale (>=4 items)",
         "Strongest meta-analytic correlate (r=.48, Mazzetti 2021) and TOP-ranked driver in the demonstration — entirely absent from 2024"],
        ["ADD", "Technology & work-mode items + work-mode field",
         "Distinct driver for distributed teams (Choudhary & Jain 2024); demonstrated importance in tab16"],
        ["ADD", "Psychological-safety and justice scales (>=4 items each)",
         "Block length drives reliability: 2-item blocks reached alpha .62 and 3-item blocks only .686-.747 in NB05, the measured case for >=4 items per modelled construct"],
        ["ADD", "ESG-alignment items",
         "Emerging fintech-relevant lever (Gannon & Hieker 2022)"],
        ["ADD", "Demographics: region, tenure, role level, work mode, department size",
         "Required for the segmentation/risk objectives; dept sizes unknown even at aggregate level in 2024 (tab10 note)"],
        ["DROP/MERGE", "Redundant sentiment items within 'Working for AccessFintech'",
         "Outcome-adjacent block (IPMA circularity, tab09); UWES replaces its measurement role:keep eNPS, merge the rest"],
        ["FIX", "Single 1-5 scale for every item",
         "Mixed 1-5/1-10 scales corrupted the workbook's own roll-ups (tab01) and echoed into correlations (NB02)"],
        ["FIX", "Anonymised ITEM-LEVEL multi-wave storage",
         "Aggregated deltas destroyed every inferential option Phase 1 documented; storage format is the single highest-leverage fix"],
    ]
    return pd.DataFrame(rows, columns=["decision", "element", "justification (traced)"])


def traceability() -> pd.DataFrame:
    """§4.4.4 — deliverable -> refined objective -> client objective.

    Essay §6.1 (tab23). The closing audit, read in both directions: every
    deliverable exists for a stated objective, and every objective has evidence
    behind it. A deliverable with no objective is scope creep; an objective with
    no deliverable is an unmet brief. Neither appears here.
    """
    rows = [
        ["Corrected theme ranking + department benchmark (tab03-04, fig01-02)", "RO1", "O1"],
        ["Priority-risk screen, localised vs company-wide (tab06)", "RO3a", "O3"],
        ["One-dimension (halo) finding, 3 converging analyses (NB02-03)", "RO1", "O1"],
        ["Calibrated synthetic panel + two quality gates (tab13-14, fig05)", "RO2 prep", "O2"],
        ["Ranked driver table, triangulated (tab16, fig06)", "RO2", "O2"],
        ["Temporally-validated risk model + explanations (tab17-18, fig07)", "RO3b", "O3"],
        ["Persona machinery + honest structure verdict (tab19)", "RO3b", "O3"],
        ["Importance x performance priorities (tab21, fig09)", "RO4", "O4"],
        ["Keep/add/drop/fix 2026 instrument (tab22)", "RO4", "O4"],
    ]
    return pd.DataFrame(rows, columns=["deliverable", "refined objective",
                                       "client objective"])
