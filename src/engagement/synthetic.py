"""Phase 2 input — calibrated synthetic-data generator + validators.

Purpose (methodology §4.3): build an individual-level, multi-wave,
demographically rich dataset that is CONSISTENT WITH the real 2024 aggregates,
so the analytic pipeline the real data cannot support can be demonstrated.

Design
------
1. Marginal calibration — for the 34 real items, each department's simulated
   wave-1 mean matches the reconstructed real absolute score (native 1-5 or
   1-10 scale) within `synthetic.calibration_tolerance`, enforced by an
   iterative mean-matching loop after ordinal discretisation.
2. Theory-informed dependence — construct latents are drawn from a Gaussian
   copula whose correlations with the UWES-style engagement latent are set in
   `config.yaml -> synthetic.target_correlations` (Mazzetti et al. 2021;
   Bakker & Demerouti 2017; Saks 2006; Edmondson 1999; Choudhary & Jain 2024;
   Gannon & Hieker 2022).
3. Instrument-complete variables — constructs the 2024 survey lacks are added:
   UWES engagement block (outcome), personal resources, technology/work-mode,
   justice, psychological safety, ESG alignment.
4. Demographics & waves — region, tenure, role level, work mode;
   `n_waves` panel waves with small controlled construct-level drift.
   (A 'function' field was removed: with anonymised single-letter departments
   there is no real information to derive it from, and a fabricated copy of
   `department` would silently duplicate a model feature.)
5. Reproducibility — single seed from config; parameters saved with the data.

Claim boundary: outputs demonstrate METHOD CAPABILITY, never facts about
AccessFintech's workforce. Correlational structure is partly imposed.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from engagement.quality import cronbach_alpha

# Real survey categories -> generating construct (JD-R reading of the 2024 instrument)
CONSTRUCT_OF_CATEGORY = {
    "My role at Access Fintech": "development",
    "Working with my Line Manager": "social",
    "Working with my Team": "social",
    "Culture & Wellbeing": "social",
    "Work/Life Balance": "balance",          # balance = reverse of demands (high balance = low demands)
    "Working for AccessFintech": "satisfaction",
    "Working with the Leadership Teams (Heads of Department)": "justice",
    "Working with the Executive Teams": "justice",
}

# New instrument-complete blocks: code prefix -> (construct, n_items, company mean)
NEW_BLOCKS = {
    "uwes": ("engagement", 3, 3.80),      # vigour / dedication / absorption
    "pers": ("personal", 3, 3.75),
    "tech": ("technology", 3, 3.90),
    "just": ("justice", 2, 3.70),
    "psys": ("psych_safety", 2, 3.85),
    "esg":  ("esg", 2, 3.60),
}

DEMOGRAPHICS = {
    "region": (["UK", "US", "Europe", "Israel", "India"], [0.30, 0.25, 0.15, 0.15, 0.15]),
    "tenure": (["<1y", "1-3y", "3-5y", ">5y"], [0.25, 0.40, 0.20, 0.15]),
    "role_level": (["junior", "mid", "senior", "lead"], [0.30, 0.40, 0.20, 0.10]),
    "work_mode": (["onsite", "hybrid", "remote"], [0.20, 0.55, 0.25]),
}


def _construct_corr_matrix(targets: dict[str, float]) -> tuple[list[str], np.ndarray]:
    """Correlation matrix over [engagement + constructs], PD-corrected.

    How it works
    ------------
    1. Start from a k x k matrix filled with 0.30 and a diagonal of 1.0.
    2. Overwrite row/column 0 with the configured engagement<->construct
       correlations. These are the values taken from the literature.
    3. Overwrite the `demands` row/column with -0.15 against every other
       construct, then restore its own engagement cell from config.
    4. Eigen-decompose. If the smallest eigenvalue is (near) negative the matrix
       is not a valid correlation matrix, so clip the eigenvalues, rebuild, and
       rescale the diagonal back to 1.
    5. Warn loudly if step 4 fired, reporting how far the applied matrix moved
       from the configured one.

    Where 0.30 and -0.15 come from. Only the engagement<->construct cells are
    evidenced individually; the construct<->construct cells are not the object
    of the demonstration, and setting them all to zero would be the stronger
    assumption. Job resources are known to inter-correlate moderately, and
    demands relate negatively to resources (JD-R: Bakker & Demerouti, 2017).
    0.30 and -0.15 are deliberately modest placeholders: large enough to avoid
    an implausibly orthogonal world, small enough not to manufacture the
    multicollinearity that NB05's driver analysis then tests for. They are
    disclosed, not hidden, and the driver results are checked against VIF.

    Why PD matters: a Cholesky factorisation exists only for a positive-definite
    matrix, so without step 4 `generate` would fail outright, better a
    documented, reported correction than a crash or a silent fudge.

    The final matrix is DISCLOSED by validate_plausibility, which reads the
    targets from what this function RETURNED rather than from config, so the
    disclosure stays truthful even when step 4 has adjusted the values.
    """
    names = ["engagement"] + list(targets)
    k = len(names)
    R = np.full((k, k), 0.30)
    np.fill_diagonal(R, 1.0)
    for j, c in enumerate(targets, start=1):
        R[0, j] = R[j, 0] = targets[c]
    d = names.index("demands")
    for j in range(1, k):
        if j != d:
            R[d, j] = R[j, d] = -0.15
    R[0, d] = R[d, 0] = targets["demands"]  # defensive: survives a range(1,k)->range(k) edit
    # PD correction via eigenvalue clipping
    w, V = np.linalg.eigh(R)
    if w.min() < 1e-6:
        # NOT silent: if this branch runs, the applied matrix deviates from the
        # configured targets, and the disclosure table must report the APPLIED
        # values — validate_plausibility therefore reads targets from this
        # returned matrix, never from config directly.
        R_intended = R.copy()
        min_eig = float(w.min())
        w = np.clip(w, 1e-6, None)
        R = V @ np.diag(w) @ V.T
        Dinv = np.diag(1 / np.sqrt(np.diag(R)))
        R = Dinv @ R @ Dinv
        warnings.warn(
            f"Target correlation matrix was not positive-definite "
            f"(min eigenvalue {min_eig:.4f}); corrected by eigenvalue clipping. "
            f"Max deviation from configured values: "
            f"{np.abs(R - R_intended).max():.3f}. The disclosure table reports "
            f"the CORRECTED (applied) matrix.")
    return names, R


def _discretise(x: np.ndarray, scale: int) -> np.ndarray:
    return np.clip(np.rint(x), 1, scale)


def generate(cfg: dict[str, Any], real_targets: pd.DataFrame,
             item_meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the synthetic panel.

    How it works (seven stages, in order)
    ------------------------------------
    1. **Item map.** Give every item a code: `q01`..`q34` for the real items
       (carrying their real text, category, scale and JD-R construct) and
       `uwes_1`, `pers_1`, ... for the new blocks. Everything downstream refers
       to codes, so item text is stored exactly once.
    2. **Employees.** Draw a department for each of the n employees with
       probabilities matching the real department sizes ('Other (24)' really is
       24 people), then draw the four demographics independently.
    3. **Latents (the copula).** `cholesky(R)` factorises the correlation matrix,
       and multiplying independent normals by it produces correlated ones:
       `Z = randn(n, k) @ L.T` gives each employee a set of construct scores with
       exactly the intended dependence. `balance` is then set to `-demands`
       (reverse-keyed items). A department 'halo' offset taken from the real
       composite deviation, is added to every construct, reproducing the general
       factor Phase 1 found. Three small demographic effects are injected so
       segmentation has something real to recover.
    4. **Waves.** Draw a small construct-level drift per wave, with wave 1 pinned
       to zero so it stays anchored to 2024.
    5. **Items from latents.** For each item: `core = loading * latent +
       sqrt(1 - loading^2) * noise`, the standard one-factor measurement model,
       weighted so the item's total variance stays 1. Real items are then centred
       on their department's real target; new items on a fixed block mean.
    6. **Discretise + calibrate.** Round to the ordinal scale, then iterate a
       per-department shift: measure the error against the real target, move the
       shift 70% of the way back (damped, because rounding makes the shift->mean
       map a step function), repeat. The `for/else` logs honestly when the loop
       stops short instead of converging.
    7. **Repair pass (discrete raking).** Rounding leaves small residual errors,
       especially in small departments. Nudge individual wave-1 responses by +-1
       within scale bounds until each department-item mean is inside tolerance.
       This is why gate 1 passes on the repaired data where the rounding loop
       on its own would have left it outside tolerance.

    Nothing here is trusted on faith: stages 6-7 are audited by
    `validate_fidelity` and stage 3 by `validate_plausibility`, both blocking.

    Parameters
    ----------
    cfg : loaded config.
    real_targets : department x item matrix of reconstructed ABSOLUTE scores
        on native scales (from quality.to_matrix(absolute, "abs_value"),
        Company row excluded).
    item_meta : columns [item, category, scale] for the 34 real items.

    Returns
    -------
    (data, item_map)
    data : one row per employee x wave; columns = employee_id, department,
        demographics, wave, then item codes q01..q34 + new-block items.
    item_map : code -> item text, category, scale, construct (incl. new items).
    """
    s_cfg = cfg["synthetic"]
    rng = np.random.default_rng(cfg["project"]["seed"])
    n, n_waves = s_cfg["n_employees"], s_cfg["n_waves"]
    loading, within_sd = s_cfg["item_loading"], s_cfg["within_dept_sd"]

    # ---- item map (codes for real items + new blocks) -----------------------
    real_items = list(real_targets.columns)
    meta_ix = item_meta.set_index("item")        # index ONCE, not per lookup
    rows = [{"code": f"q{i+1:02d}", "item": it,
             "category": meta_ix.loc[it, "category"],
             "scale": int(meta_ix.loc[it, "scale"]),
             "construct": CONSTRUCT_OF_CATEGORY[meta_ix.loc[it, "category"]]}
            for i, it in enumerate(real_items)]
    for pre, (con, k, _) in NEW_BLOCKS.items():
        for j in range(1, k + 1):
            rows.append({"code": f"{pre}_{j}", "item": f"[NEW] {con} item {j}",
                         "category": f"NEW: {con}", "scale": 5, "construct": con})
    item_map = pd.DataFrame(rows).set_index("code")

    # ---- employees: departments + demographics ------------------------------
    depts = list(real_targets.index)
    w = np.array([24 if d == "Other (24)" else 83 / (len(depts) - 1) for d in depts], float)
    dept_of = rng.choice(depts, size=n, p=w / w.sum())
    data = pd.DataFrame({"employee_id": [f"E{i:04d}" for i in range(n)], "department": dept_of})
    for col, (cats, p) in DEMOGRAPHICS.items():
        data[col] = rng.choice(cats, size=n, p=p)

    # ---- construct latents via Gaussian copula ------------------------------
    names, R = _construct_corr_matrix(dict(s_cfg["target_correlations"]))
    L = np.linalg.cholesky(R)
    Z = rng.standard_normal((n, len(names))) @ L.T          # employee-level latents
    lat = pd.DataFrame(Z, columns=names)
    lat["balance"] = -lat["demands"]        # balance items are reverse-keyed demands
    names = names + ["balance"]
    # halo: department general offset (from real composite deviation) enters
    # every construct weakly, mimicking the general factor found in Phase 1.
    # Note: the offset uses the raw (mixed-scale) row mean; acceptable because
    # subtracting the overall mean cancels most of the 1-10 items' level
    # inflation and only relative deviation is used (config: halo_strength).
    dept_dev = (real_targets.mean(axis=1) - real_targets.values.mean())
    halo = data["department"].map(dept_dev).to_numpy()
    halo_strength = s_cfg.get("halo_strength", 0.5)
    for c in names:
        lat[c] = lat[c] + halo_strength * halo

    # small demographic effects (so segmentation has something to find)
    lat.loc[data["work_mode"].eq("remote").to_numpy(), "technology"] -= 0.35
    lat.loc[data["tenure"].eq("<1y").to_numpy(), "engagement"] += 0.15
    lat.loc[data["role_level"].eq("lead").to_numpy(), "engagement"] += 0.20

    # ---- wave drift (construct-level, controlled) ---------------------------
    drift = rng.normal(0, s_cfg["wave_drift_sd"], size=(n_waves, len(names)))  # incl. balance
    drift[0] = 0.0                                          # wave 1 anchored to 2024

    frames = []
    for wv in range(1, n_waves + 1):
        f = data.copy()
        f["wave"] = wv
        eps = rng.standard_normal((n, len(item_map)))
        for j, code in enumerate(item_map.index):
            meta = item_map.loc[code]
            con = meta["construct"]
            latv = lat[con].to_numpy() + drift[wv - 1, names.index(con)]
            core = loading * latv + np.sqrt(1 - loading**2) * eps[:, j]
            if code.startswith("q"):
                target = real_targets[meta["item"]]
                mu = f["department"].map(target).to_numpy()
                sd = within_sd * (meta["scale"] / 5)
                f[code] = mu + sd * core
            else:
                mu0 = NEW_BLOCKS[code.split("_")[0]][2]
                f[code] = mu0 + within_sd * core
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)

    # ---- discretise + wave-1 mean-matching calibration ----------------------
    codes = list(item_map.index)
    scales = item_map["scale"].to_dict()
    shift = {c: pd.Series(0.0, index=depts) for c in codes if c.startswith("q")}
    w1 = out["wave"] == 1
    for _ in range(s_cfg["max_calibration_iter"]):
        worst = 0.0
        for c in shift:
            cont = out.loc[w1, c] + out.loc[w1, "department"].map(shift[c])
            disc = _discretise(cont.to_numpy(), scales[c])
            err = (pd.Series(disc, index=out.loc[w1, "department"]).groupby(level=0).mean()
                   - real_targets[item_map.loc[c, "item"]])
            shift[c] = shift[c] - 0.7 * err.reindex(depts).fillna(0)   # damped (rounding makes the map a step function)
            worst = max(worst, float(err.abs().max()))
        if worst <= s_cfg["calibration_tolerance"] * 0.9:
            break
    else:  # attached to the for-loop: runs only when it did NOT break
        # Transparency (not an error): with ordinal rounding the shift->mean
        # map is a step function; the department-level loop lands ~97% of
        # cells inside tolerance but cannot converge on a few stubborn
        # near-ceiling cells — the individual-level raking below finishes those.
        print(f"[calibration] department-level loop stopped at worst error "
              f"{worst:.4f} (> tolerance {s_cfg['calibration_tolerance']}); "
              f"individual-level raking handles the residual cells.")
    for c in codes:
        if c.startswith("q"):
            out[c] = _discretise((out[c] + out["department"].map(shift[c])).to_numpy(), scales[c])
        else:
            out[c] = _discretise(out[c].to_numpy(), scales[c])

    # ---- repair pass (discrete raking): rounding leaves residual mean errors,
    # especially in small departments; nudge individual wave-1 responses by ±1
    # (within scale bounds) until every department-item mean is inside tolerance.
    tol = s_cfg["calibration_tolerance"]
    w1_idx = out.index[out["wave"] == 1]
    for c in [c for c in codes if c.startswith("q")]:
        target = real_targets[item_map.loc[c, "item"]]
        top = scales[c]
        for dept in depts:
            ridx = w1_idx[out.loc[w1_idx, "department"] == dept]
            n_d = len(ridx)
            if n_d == 0:
                continue
            for _ in range(3 * n_d):
                err = out.loc[ridx, c].mean() - target[dept]
                if abs(err) <= tol * 0.6:
                    break
                if err > 0:   # too high -> lower one respondent above floor
                    cand = ridx[out.loc[ridx, c] > 1]
                    if len(cand) == 0:
                        break
                    out.loc[rng.choice(cand), c] -= 1
                else:         # too low -> raise one respondent below ceiling
                    cand = ridx[out.loc[ridx, c] < top]
                    if len(cand) == 0:
                        break
                    out.loc[rng.choice(cand), c] += 1
    return out, item_map


# --------------------------------------------------------------- validators
def validate_fidelity(data: pd.DataFrame, item_map: pd.DataFrame,
                      real_targets: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Quality gate 1: wave-1 department x item means vs real targets.

    Returns dict with per-item report, worst error, and pass flag. The
    pipeline/notebook HALTS if `passed` is False (ILO1 applied to generated
    data). Also checks the department composite RANKING is reproduced
    (Spearman), tying Phase 2 back to the Phase 1 composite index.

    How it works
    ------------
    1. Keep wave 1 only, it is the wave calibrated to 2024; later waves are
       supposed to drift.
    2. Average the real-item columns by department and rename them back to item
       text, so the simulated matrix has the same shape and labels as
       `real_targets` and pandas can subtract them cell by cell.
    3. `err = |sim - real|`, then report the worst and mean error per item.
    4. Rank the two composite indices and correlate the ranks (Spearman). This
       is a second, independent question: cell-level accuracy could be fine while
       the department ORDER, the thing Phase 1 actually reported, was wrong.
    5. `passed` is the single worst cell against the tolerance, not the average.
       An average would let a few badly-off departments hide behind 600 good
       cells.
    """
    w1 = data[data["wave"] == 1]
    qcodes = [c for c in item_map.index if c.startswith("q")]
    sim = w1.groupby("department")[qcodes].mean()
    sim.columns = [item_map.loc[c, "item"] for c in qcodes]
    err = (sim - real_targets).abs()
    comp_sim = sim.mean(axis=1).rank()
    comp_real = real_targets.mean(axis=1).rank()
    report = pd.DataFrame({
        "max_abs_error": err.max(), "mean_abs_error": err.mean()
    }).sort_values("max_abs_error", ascending=False)
    tol = cfg["synthetic"]["calibration_tolerance"]
    return {
        "report": report.round(4),
        "sim_means": sim,                       # kept for the fidelity scatter
        "worst_error": float(err.to_numpy().max()),
        "tolerance": tol,
        "passed": bool(err.to_numpy().max() <= tol),
        "ranking_spearman": float(comp_sim.corr(comp_real, method="spearman")),
    }


def validate_plausibility(data: pd.DataFrame, item_map: pd.DataFrame,
                          cfg: dict[str, Any]) -> dict[str, Any]:
    """Quality gate 2: ranges, skew, achieved-vs-target correlations — BLOCKING.

    Three checks feed `passed` (all must hold):
      1. every response within its ordinal scale bounds, no NaNs;
      2. engagement-block alpha >= synthetic.min_engagement_alpha;
      3. worst |disattenuated_r - target_r| <= synthetic.max_corr_gap — the
         check that would catch a broken copula (e.g. the classic `@ L` vs
         `@ L.T` mistake), which range checks alone can never detect.

    Reliability (alpha) and correlation are both computed on HARMONISED item
    blocks (`_harmonised_block`) so the two sides of the disattenuation
    formula share one metric. The NB01 rule applied to the generator's audit.

    Targets are read from the APPLIED correlation matrix returned by
    `_construct_corr_matrix`, not from config, so the disclosure stays correct
    even if the PD correction adjusted the configured values.

    How it works
    ------------
    1. Range check: for every item, is the minimum below 1 or the maximum above
       its scale, or are there NaNs? Collect the offenders.
    2. Score each construct (mean of its harmonised items) and compute alpha for
       the engagement block.
    3. Rebuild the APPLIED correlation matrix and read the engagement row from
       it, these are the targets the disclosure table reports.
    4. For each construct: correlate its score with engagement, flipping the
       sign for `demands` (stored reverse-keyed as `balance`); compute its alpha;
       divide by sqrt(alpha_x * alpha_eng) to disattenuate.
    5. Compute per-item skew (informational, a ceiling-constrained item is
       expected here, not a fault).
    6. `passed` = no range violations AND engagement alpha above its floor AND
       worst correlation gap inside its tolerance.

    Why the disattenuation step exists: observed correlations between short
    imperfect scales are systematically SMALLER than the latent correlations
    that generated them (observed r ~= latent r * sqrt(alpha_x * alpha_y)).
    Comparing raw observed r against the imposed latent target would fail a
    perfectly good generator. Correcting first tests what is actually claimed
    that the copula survived discretisation.

    Also returns a per-item skew report (informational, not gating).
    """
    w1 = data[data["wave"] == 1]
    bad = {}
    for c in item_map.index:
        v = w1[c]
        if v.min() < 1 or v.max() > item_map.loc[c, "scale"] or v.isna().any():
            bad[c] = (float(v.min()), float(v.max()))

    scores = construct_scores(w1, item_map)
    eng = scores["engagement"]
    a_eng = cronbach_alpha(_harmonised_block(w1, item_map, "engagement"))

    names, R = _construct_corr_matrix(dict(cfg["synthetic"]["target_correlations"]))
    applied = {con: float(R[0, j]) for j, con in enumerate(names) if con != "engagement"}

    rows = []
    for con, t in applied.items():
        col, sign = (("balance", -1) if con == "demands" else (con, 1))
        if col not in scores:
            continue
        r_obs = sign * float(scores[col].corr(eng))
        a_x = cronbach_alpha(_harmonised_block(w1, item_map, col))
        r_dis = r_obs / np.sqrt(max(a_x, 1e-6) * max(a_eng, 1e-6))
        rows.append({"construct": con, "target_latent_r": round(t, 3),
                     "achieved_observed_r": round(r_obs, 3),
                     "alpha": round(a_x, 2),
                     "disattenuated_r": round(r_dis, 3)})
    ach = pd.DataFrame(rows).set_index("construct")

    skew = w1[item_map.index].skew()
    skew_report = pd.DataFrame({
        "skew": skew.round(3),
        "flag": np.where(skew.abs() > 2.0, "extreme (ceiling-constrained)", ""),
    }).sort_values("skew")

    corr_gap = float((ach["disattenuated_r"] - ach["target_latent_r"]).abs().max())
    alpha_ok = a_eng >= cfg["synthetic"].get("min_engagement_alpha", 0.60)
    corr_ok = corr_gap <= cfg["synthetic"].get("max_corr_gap", 0.10)
    return {"range_violations": bad, "achieved_vs_target": ach,
            "engagement_alpha": round(a_eng, 2),
            "worst_corr_gap": round(corr_gap, 3),
            "skew_report": skew_report,
            "n_extreme_skew": int((skew.abs() > 2.0).sum()),
            "passed": (not bad) and alpha_ok and corr_ok}


def _harmonised_block(data: pd.DataFrame, item_map: pd.DataFrame,
                      con: str) -> pd.DataFrame:
    """Item block for one construct, with 1-10 items rescaled onto 1-5.

    Shared by construct_scores() and every alpha computation, so reliability
    and correlation are always estimated on the SAME metric with the NB01 rule
    (never average or weight across mixed scales) enforced in one place.
    """
    cols = item_map.index[item_map["construct"] == con]
    block = data[list(cols)].copy()
    for c in cols:
        if item_map.loc[c, "scale"] == 10:
            block[c] = 1 + (block[c] - 1) * (4 / 9)
    return block


def construct_scores(data: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Mean harmonised item score per construct per row (validators & models).

    How it works: for each distinct construct in the item map, take its item
    block through `_harmonised_block` (so any 1-10 item is rescaled first) and
    average across columns, one score per employee per construct.

    This is the single place construct scores are built. Models, validators and
    personas all call it, so none of them can quietly use a different definition
    of, say, `social`.
    """
    return pd.DataFrame(
        {con: _harmonised_block(data, item_map, con).mean(axis=1)
         for con in item_map["construct"].unique()},
        index=data.index)


def save(data: pd.DataFrame, item_map: pd.DataFrame, cfg: dict[str, Any]) -> Path:
    """Persist dataset + generator parameters next to each other (provenance).

    Three files, deliberately in one folder: the panel, the item map (codes ->
    real item text), and `generator_params.json` holding the seed and the whole
    `synthetic` config block. Anyone who receives the CSV also receives exactly
    what produced it, so the dataset can be regenerated rather than trusted.
    """
    d = Path(cfg["paths"]["synthetic_dir"]); d.mkdir(parents=True, exist_ok=True)
    data.to_csv(d / "synthetic_panel.csv", index=False)
    item_map.to_csv(d / "synthetic_item_map.csv")
    (d / "generator_params.json").write_text(json.dumps(
        {"seed": cfg["project"]["seed"], **cfg["synthetic"]}, indent=2, default=str))
    return d / "synthetic_panel.csv"
