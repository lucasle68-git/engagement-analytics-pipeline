"""Fabricate the aggregated survey workbook the pipeline expects, with invented numbers.

Why this script exists
----------------------
The 2024 survey belongs to the client and is not redistributable, so the real
workbook is absent from this repository. Without *some* workbook at
`data/raw/`, every notebook stops at the first cell and a reader can see the
code but never a result.

This script writes a stand-in at `data/raw/engagement_survey_2024.xlsx` that is
structurally identical to the client file — same sheet name, same 22 columns,
same 42 rows, same 8 category headers, same two 1-10 items, same
delta-from-company-mean encoding, same empty `manager` column — and entirely
invented numerically. The whole pipeline therefore runs end to end on a public
clone, and every figure it produces is stamped DEMO rather than REAL (see
`engagement.viz.save_figure` and `data.provenance` in config).

What is deliberately reproduced
-------------------------------
The demo is not uniform noise. Three properties of the real file are imitated,
because they are what the analysis is written to detect, and a flat random
workbook would make notebooks 01-03 render empty arguments:

1. A *theme ordering* — team-level themes score above leadership and executive
   themes, the pattern engagement surveys usually show.
2. A *department general factor* ("halo"): departments differ more in overall
   level than themes differ from each other, so the clustering and PCA sections
   have the structure they are written to find and argue about.
3. *Mixed scales inside a roll-up*: the two 1-10 items sit inside otherwise 1-5
   themes, so `quality.validate_structure` finds the same arithmetic-versus-
   comparability defect on the demo file that it finds on the real one, and
   'Work/Life Balance' appears strongest purely as a scale artefact.

What is NOT reproduced: any real score. No value here is derived from, fitted
to, or calibrated against the client workbook. Department labels A-R carry no
correspondence to the real departments, and the named executives in the real
instrument's trust item are replaced with a generic phrase.

How it works
------------
1. Lay out the instrument: eight categories, each with its ordered items, and a
   flag for the two items measured 1-10 rather than 1-5.
2. Draw a company-level mean per item: the theme's base level plus a small
   item-specific offset, on that item's own scale.
3. Draw a department effect per (department, item): a department-wide level
   shift, plus a department-by-theme shape shift, plus item noise scaled by a
   per-item "contestedness" factor so some items divide departments more than
   others. Effects on a 1-10 item are widened by 9/4 to keep them proportionate
   to that item's ruler.
4. Add the effect to the company mean to get a department's *absolute* score,
   clip that to the item's response scale, and only then subtract the company
   mean back off to get the delta the workbook stores. Doing it in this order
   matters: a delta is a difference between two bounded quantities, so applying
   the bound to the delta directly can put a reconstructed score above 5 on a
   1-5 item — which is what a naive generator does and what the real file, being
   real, never does. Clipping also introduces a mild ceiling effect on the
   strongest items, as real Likert data has.
5. Compute each category header as the exact mean of its member items — for the
   company column and for every department column alike, which is how the real
   workbook is built and what the structure check audits.
5. Round to two decimals (the source file's precision) and write the sheet.

Rounding is applied after the header means are computed, so a header can differ
from the mean of its rounded items by at most ~0.01 — inside the 0.05 tolerance
in `config.quality.structure_tolerance`, and the same order of rounding
residual the real file carries.

Usage
-----
    python scripts/make_demo_workbook.py            # write the demo workbook
    python scripts/make_demo_workbook.py --force    # overwrite an existing file

The seed is fixed, so the numbers are reproducible: two people regenerating this
workbook get identical cell values and therefore identical figures and tables.
(The .xlsx *file* is not byte-identical between runs — openpyxl stamps its zip
container with metadata — but every value inside it is.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "engagement_survey_2024.xlsx"
SHEET = "results"

# Fixed independently of config.project.seed: the demo data must not move when
# someone re-seeds the analysis to test its stability.
DEMO_SEED = 20260914

LABEL_COL = "Category / Question"
COMPANY_COL = "Company Overall (107)"
DEPT_COLS = ["Other (24)", *list("ABCDEFGHIJKLMNOPQR")]   # 19 reporting units

# The two items the client measures 1-10; every other item is a 1-5 Likert.
HAPPINESS_10 = "How happy are you at AccessFintech on a scale of 1 to 10?"
BALANCE_10 = (
    "On a scale of one to ten, how well are you able to balance your work "
    "and personal commitments?"
)
TEN_POINT = {HAPPINESS_10, BALANCE_10}

# The instrument: category -> ordered items. Question wording follows the real
# survey because it is what the analysis code and config key on; the trust item's
# parenthetical list of named executives is replaced, as it is personal data.
INSTRUMENT: dict[str, list[str]] = {
    "Working for AccessFintech": [
        "I can see myself working for Access Fintech for the next 2 years.",
        "I am optimistic about Access Fintech's long-term success.",
        "As an employee, I get sufficient and relevant information about company "
        "updates/org changes from the executive teams.",
        "The day-to-day decisions here demonstrate that we continuously learn from "
        "experiences and aim for improvement.",
        "I would recommend AFT to my friends.",
        HAPPINESS_10,
    ],
    "My role at Access Fintech": [
        "My job gives me professional challenges and that is exciting to me.",
        "I understand how my work contributes to the company's success.",
        "AFT places great importance on the growth and development of its people.",
        "I believe I have an equal opportunity within the team to have a "
        "stable/growing career path in the organization.",
        "I am recognized for my accomplishments and contributions to the company.",
        "I think I have been offered the support and flexibility necessary to thrive at work.",
    ],
    "Culture & Wellbeing": [
        "I feel Accessfintech's culture is collaborative and empowering.",
        "I feel comfortable sharing my ideas at work with peers and management.",
        "I think my work environment reflects the company culture.",
        "I feel the HR team at Access Fintech care about my wellbeing.",
        "The HR team are attentive, involved and helpful, I feel like I have "
        "someone to turn to.",
    ],
    "Work/Life Balance": [
        "I feel the amount of work assigned to me is manageable",
        "I feel AFT is supportive of a healthy work-life balance.",
        BALANCE_10,
    ],
    "Working with my Line Manager": [
        "My manager keeps me informed about what is happening within my team and "
        "the wider business.",
        "I receive regular constructive feedback from my manager.",
        "My manager is helpful and empathetic and I feel comfortable approaching "
        "them on personal matters.",
        "My manager helps me identify the best resources to develop my skills and "
        "knowledge as a professional",
        "My work is appreciated by my direct manager.",
    ],
    "Working with my Team": [
        "My team cooperates and helps each other.",
        "Me and my team members contribute to the success of our team and the "
        "success of the company.",
        "Collaboration between departments/teams is done well at Access Fintech.",
    ],
    "Working with the Leadership Teams (Heads of Department)": [
        "The Leadership team openly and transparently updates employees on "
        "decisions and changes in the company.",
        "The leadership team values all employees and treats them equally with "
        "fairness and respect.",
        "My work is appreciated by my Head of Department.",
    ],
    "Working with the Executive Teams": [
        "I trust the Executive team to lead the business to success.",
        "The executive team works as a cohesive team.",
        "The executive team contributes to a positive work culture.",
    ],
}

# Invented theme levels on the 1-5 metric. Ordered to give the diagnostic
# notebooks a real gradient to describe: immediate team strongest, distant
# leadership weakest.
THEME_BASE_1_5: dict[str, float] = {
    "Working for AccessFintech": 4.00,
    "My role at Access Fintech": 3.85,
    "Culture & Wellbeing": 3.78,
    "Work/Life Balance": 3.58,
    "Working with my Line Manager": 3.92,
    "Working with my Team": 4.26,
    "Working with the Leadership Teams (Heads of Department)": 3.30,
    "Working with the Executive Teams": 3.12,
}

# Invented company means for the 1-10 items, on their own ruler. Both sit high
# enough that averaging them into a 1-5 theme visibly inflates it.
TEN_POINT_BASE: dict[str, float] = {
    HAPPINESS_10: 7.40,
    BALANCE_10: 7.55,
}

ITEM_OFFSET_SD = 0.18      # item-to-item variation inside a theme (1-5 metric)
DEPT_LEVEL_SD = 0.30       # department-wide level shift ("halo")
DEPT_THEME_SD = 0.20       # department-by-theme shape shift
ITEM_NOISE_SD = 0.20       # residual department x item noise
CONTESTED_RANGE = (0.55, 1.85)   # per-item multiplier on the noise term
TEN_POINT_WIDEN = 9 / 4    # a 1-5 effect expressed on a 1-10 ruler

# Absolute department scores are held inside the usable part of each ruler. The
# margin off the endpoints reflects that a department mean of exactly 1.00 or
# 5.00 would mean every respondent in it picked the same extreme.
BOUNDS_1_5 = (1.40, 4.92)
BOUNDS_1_10 = (2.20, 9.60)


def build_demo_frame(seed: int = DEMO_SEED) -> pd.DataFrame:
    """Return the wide demo heatmap: 42 rows x 22 columns, ready to write.

    Rows appear in instrument order — each category header immediately followed
    by its items — because `engagement.io.tidy` assigns items to themes by
    reading top to bottom, exactly as a person reads the spreadsheet.
    """
    rng = np.random.default_rng(seed)

    themes = list(INSTRUMENT)
    theme_index = {t: i for i, t in enumerate(themes)}

    # Department structure, drawn once and shared by every item.
    dept_level = rng.normal(0.0, DEPT_LEVEL_SD, size=len(DEPT_COLS))
    dept_theme = rng.normal(0.0, DEPT_THEME_SD, size=(len(DEPT_COLS), len(themes)))

    company_mean: dict[str, float] = {}
    dept_delta: dict[str, np.ndarray] = {}

    for theme, items in INSTRUMENT.items():
        t = theme_index[theme]
        for item in items:
            ten_point = item in TEN_POINT
            widen = TEN_POINT_WIDEN if ten_point else 1.0

            base = (
                TEN_POINT_BASE[item]
                if ten_point
                else THEME_BASE_1_5[theme] + rng.normal(0.0, ITEM_OFFSET_SD)
            )
            company_mean[item] = float(base)

            # Some items divide departments far more than others; the dispersion
            # screen in notebook 02 exists to find exactly these.
            contested = rng.uniform(*CONTESTED_RANGE)
            noise = rng.normal(0.0, ITEM_NOISE_SD * contested, size=len(DEPT_COLS))
            effect = (dept_level + dept_theme[:, t] + noise) * widen

            # Bound the absolute score, then express it back as a delta, so no
            # reconstructed score can fall outside the item's response scale.
            lo, hi = BOUNDS_1_10 if ten_point else BOUNDS_1_5
            absolute = np.clip(base + effect, lo, hi)
            dept_delta[item] = absolute - base

    # Assemble rows: header first, then its items; header value is the exact
    # mean of its members, for the company column and every department alike.
    records: list[dict[str, object]] = []
    for theme, items in INSTRUMENT.items():
        header: dict[str, object] = {LABEL_COL: theme, "manager": None}
        header[COMPANY_COL] = float(np.mean([company_mean[i] for i in items]))
        member_deltas = np.vstack([dept_delta[i] for i in items]).mean(axis=0)
        for col, val in zip(DEPT_COLS, member_deltas):
            header[col] = float(val)
        records.append(header)

        for item in items:
            row: dict[str, object] = {LABEL_COL: item, "manager": None}
            row[COMPANY_COL] = company_mean[item]
            for col, val in zip(DEPT_COLS, dept_delta[item]):
                row[col] = float(val)
            records.append(row)

    df = pd.DataFrame.from_records(records, columns=[LABEL_COL, COMPANY_COL, *DEPT_COLS, "manager"])

    # Two decimals is the precision of the source file. Rounding after the header
    # means are taken leaves the same small residual the real workbook carries.
    numeric = [COMPANY_COL, *DEPT_COLS]
    df[numeric] = df[numeric].astype(float).round(2)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite an existing workbook (refuses by default, so a real "
             "client file placed here is never silently replaced)",
    )
    parser.add_argument("--seed", type=int, default=DEMO_SEED, help=f"default {DEMO_SEED}")
    args = parser.parse_args(argv)

    if OUT_PATH.exists() and not args.force:
        print(
            f"{OUT_PATH.relative_to(ROOT)} already exists — leaving it alone.\n"
            "Pass --force to overwrite (do NOT do this if you have placed the real "
            "client workbook there).",
            file=sys.stderr,
        )
        return 1

    df = build_demo_frame(args.seed)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_PATH, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=SHEET, index=False)

    n_items = sum(len(v) for v in INSTRUMENT.values())
    print(
        f"Wrote {OUT_PATH.relative_to(ROOT)}\n"
        f"  sheet '{SHEET}': {len(df)} rows x {len(df.columns)} columns\n"
        f"  {len(INSTRUMENT)} category headers + {n_items} items, "
        f"{len(DEPT_COLS)} reporting units, seed {args.seed}\n"
        "  DEMO DATA — invented numbers, no client values."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
