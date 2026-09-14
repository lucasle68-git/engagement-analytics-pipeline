# Employee Engagement Analytics — diagnostic, synthetic demonstration, survey redesign

A two-phase analytics consultancy project for a fintech client: diagnose what an aggregated
employee-engagement survey **can** support, demonstrate on synthetic data what better data
**would** unlock, and convert the gap between the two into a redesigned 2026 instrument.

Built as an end-to-end Python package with a one-command pipeline, a tested synthetic-data
generator with blocking quality gates, and data-provenance controls enforced in code.

> **Runs out of the box.** The client's survey is not redistributable, so this repository ships a
> fabricated stand-in with the same structure and invented numbers. `make all` reproduces every
> figure and table below in about two minutes. See [Data governance](#data-governance).

<p align="center">
  <img src="outputs/figures/fig01_theme_ranking.png" width="49%" alt="Ranked engagement themes">
  <img src="outputs/figures/fig06_driver_ranking.png" width="49%" alt="Ranked engagement drivers">
</p>

---

## The problem

The client supplied a single spreadsheet: one wave, 107 respondents, **already aggregated**. Item
means for the company, and every department stored as a *delta* from that mean. No individual
responses, no demographics, no time dimension, and an empty `manager` column.

Stakeholders wanted drivers of engagement, attrition risk scores, employee segments, and trends.
**None of those are answerable from that file** — not because of method, but because the data
elements they require were never collected.

The project's argument is built on taking that seriously rather than working around it.

| Phase | Question | Data | What it may claim |
|---|---|---|---|
| **1 — Diagnostic** | Where does engagement stand, and where are the gaps? | Real aggregated survey | Description only, bounded explicitly |
| **2 — Demonstration** | What would individual-level, multi-wave data unlock? | Synthetic panel, 1,000 employees × 3 waves | Method capability, never company fact |
| **3 — Synthesis** | What must the 2026 survey collect? | Both | A specification, traceable to a measured gap |

The deliverable that matters is Phase 3: every proposed survey change is traceable to a specific
analysis that failed for a specific missing data element — a consequence, not a preference.

## What this demonstrates

| Area | In this repository |
|---|---|
| **Data engineering** | Schema validation that fails loudly on a changed file; delta→absolute reconstruction; mixed-scale harmonisation; tidy reshaping; a config-driven package with zero magic numbers in analysis code |
| **Statistics** | Dispersion screening, hierarchical clustering with silhouette-based model selection, PCA, importance–performance analysis with a circularity correction, Cronbach's α, correction for measurement attenuation |
| **Machine learning** | Johnson relative-weights driver analysis, imbalanced classification with temporal validation and threshold tuning, SHAP explanations, k-means segmentation held to a stated standard |
| **Synthetic data** | Gaussian-copula generator calibrated to real aggregates, with **two blocking quality gates** — fidelity and plausibility — that fail the run rather than warn |
| **Software practice** | `src/` package, `pytest` suite on the correctness-critical transforms, one-command reproducible pipeline, pinned dependencies, generated provenance manifest |
| **Judgement & communication** | Every analysis preceded by a decision box (*what · why this method · what was rejected · what it feeds*); claim boundaries enforced by a function, not by discipline |

## Three things worth looking at

**1. A guard that cannot be forgotten.** Every figure in the project is written to disk by one
function, `viz.save_figure`, which stamps a provenance badge and refuses to save a figure that has
not declared where its data came from. Phase 1 figures are badged REAL, Phase 2 SYNTHETIC — and
when the pipeline runs on the demo workbook, `real` is automatically downgraded to **DEMO** from
config, so a public clone *cannot* produce a chart captioned as real client data. The guarantee is
structural rather than a matter of remembering.

**2. Quality gates that block.** The synthetic generator is not trusted because it looks
reasonable. Gate 1 checks every department × construct cell against the calibration tolerance *and*
requires the department ranking to be reproduced exactly (ρ = 1.00) — a generator that matched the
means while scrambling the order would be useless. Gate 2 checks that the correlations imposed from
the literature survive discretisation, correcting for measurement attenuation before comparing.
Both halt the pipeline on failure. `tests/test_synthetic.py` tests that they halt it.

**3. Negative results reported as results.** The clustering section found no department archetypes,
and says so. The persona section is held to the *same* silhouette standard as Phase 1 and fails it
in the same way, and says so. An importance–performance map built on ~19 aggregated units is
reported as exploratory with its three weaknesses named. A project that discovers structure only
when it needs some has a method problem, not a finding.

---

## Quick start

Python 3.10+. Three commands, about two minutes.

```bash
pip install -r requirements.txt && pip install -e .
make demo-data        # write the demo survey workbook to data/raw/
make test             # 6 tests on the correctness-critical transforms
make pipeline         # execute notebooks 01-06, export HTML, write the manifest
```

`make all` runs the last three in order. Outputs land in `outputs/figures/`,
`outputs/tables/` and `outputs/notebooks_html/`.

**To read rather than run:** open `outputs/notebooks_html/` in a browser — six self-contained HTML
files, one per notebook, no installation required. Start at `01`, then `03`, then `06` for a
ten-minute route.

## The six notebooks

| # | Notebook | Question |
|---|---|---|
| 01 | `01_data_loading_quality.ipynb` | Is the data loaded correctly, and what is wrong with it? |
| 02 | `02_phase1_diagnostic.ipynb` | Where does engagement stand, and where are the gaps? |
| 03 | `03_phase1_multivariate.ipynb` | Do departments form groups? How many things does the survey measure? |
| 04 | `04_synthetic_generation_validation.ipynb` | Can we build the data the survey is missing — and prove it is sound? |
| 05 | `05_phase2_modelling.ipynb` | What would better data unlock: drivers, risk, segments, trends? |
| 06 | `06_phase3_synthesis.ipynb` | What must the 2026 survey collect? |

Each notebook follows the same shape: a header stating what it does, a **decision box** before
every analysis, the code and its results, then a short reading of the result. Notebooks contain
almost no logic — the analysis lives in `src/engagement/` and is called from them.

## How the code is organised

```
config/config.yaml    every parameter: paths, seed, thresholds, provenance
src/engagement/       all analysis code
notebooks/            the six notebooks; they call src/
scripts/              run_pipeline.py · make_demo_workbook.py
tests/                6 tests on the transforms where a silent error would spoil everything
outputs/              figures, tables, rendered HTML, run manifest
data/                 raw (demo workbook), processed, synthetic
```

| Module | Responsibility |
|---|---|
| `config.py` | Loads and validates `config.yaml`; resolves paths from the repo root |
| `io.py` | Reads the workbook, validates its structure, reshapes it to tidy long format |
| `quality.py` | Delta→absolute reconstruction, scale harmonisation, structural audit, Cronbach's α |
| `phase1.py` | Rankings, gap matrix, dispersion screen, clustering, PCA, IPMA, composite index |
| `synthetic.py` | Copula generator, calibration, and the two blocking quality gates |
| `models.py` | Reliability, driver analysis, risk models, SHAP explanations |
| `personas.py` | Segmentation, held to the Phase 1 standard |
| `synthesis.py` | Phase comparison, priority map, the 2026 instrument specification |
| `viz.py` | House chart style and the provenance badge every figure passes through |

Every function carries a plain-English **"How it works"** description written as numbered steps.

## Tests

```bash
make test
```

Six tests, one per claim the analysis makes about its own data — chosen for the steps where a
silent error would invalidate everything built on top of it, and each guard tested on both its
accepting and its rejecting path.

| Test | What it proves |
|---|---|
| Reconstruction | Delta + company mean recovers the absolute score, and the step loses no information |
| Harmonisation | The 1-10 items land on the 1-5 scale and nothing else is touched |
| Validation | A renamed column or reworded heading stops the run instead of producing a wrong answer |
| Calibration | Synthetic department means match their targets, and re-running is deterministic |
| Correlations | The relationships imposed from the literature are the ones recovered |
| Gates block | The quality gates fail the run — they do not merely print a warning |

## Reproducibility

Seeded (`project.seed` in config), pinned (`requirements.txt`), and recorded: every pipeline run
rewrites `outputs/run_manifest.json` with the timestamp, seed, Python and package versions,
per-notebook runtime and output counts, so the provenance record cannot go stale. The demo
workbook has its own fixed seed, independent of the analysis seed, so re-seeding the analysis to
test its stability does not move the data underneath it.

## Data governance

The 2024 survey belongs to the client and is not redistributable, so **it is not in this
repository** — and neither is anything derived from it. What is here instead is
`data/raw/engagement_survey_2024.xlsx`, written by `scripts/make_demo_workbook.py`: the same sheet
name, the same 22 columns and 42 rows, the same category headers, the same two 1-10 items, the
same delta-from-company-mean encoding — and entirely invented numbers.

The demo is not uniform noise. It deliberately reproduces the three *structural* properties the
analysis is written to detect — a theme ordering, a department-level general factor, and mixed
response scales inside a roll-up — so the notebooks demonstrate real arguments rather than render
empty ones. No value in it is derived from, fitted to, or calibrated against the client's file.

Consequently:

- Every figure and table in `outputs/` is computed from demo data; Phase 1 figures are badged
  **DEMO**, Phase 2 figures **SYNTHETIC**.
- Passages that describe what was actually found on the client's data say so explicitly and point
  to the report section that carries the figures, rather than repeating them.
- The survey instrument's question wording is retained because the analysis code and config key on
  it; the named individuals in one item of the real instrument are replaced with a generic phrase.
- `data.provenance` in `config/config.yaml` is the switch. Set it to `real` with the client
  workbook in place and the badges change back — the analysis code itself is identical either way.

Applying the confidentiality boundary in the published artefact, rather than only asserting it in
the report, is the same governance control the project argues for.

## Context

Produced for **MGT5496P Business Analytics Consultancy** (University of Glasgow, Adam Smith
Business School), assessed as an individual consultancy report. The written report, which contains
the client findings, is not distributed here.

**Lucas Le** · [github.com/lucasle68-git](https://github.com/lucasle68-git)

Code released under the MIT Licence (see `LICENSE`). The survey instrument wording remains the
property of the client.
