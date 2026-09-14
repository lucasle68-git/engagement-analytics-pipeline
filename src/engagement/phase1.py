"""Phase 1 — diagnostic analysis of the real 2024 aggregated data (RO1, RO3a).

Unit of analysis: ~19 aggregated department units. All functions here are
DESCRIPTIVE / EXPLORATORY. Inferential, predictive and psychometric modelling
are deliberately excluded (see methodology §4.2.6) and live in Phase 2.

Pipeline position:  quality.harmonise_scales --> harmonised long df --> HERE

Every function in this module takes the SAME input — the harmonised tidy
dataframe — and differs only in how it aggregates it. Reading the module is
therefore a matter of asking, for each function, two questions: what does it
group by, and what does it compare against?

    function              groups by                compares against
    -------------------------------------------------------------------------
    theme_ranking         category (Company only)  the other themes
    gap_matrix            department x category    the company mean
    top_gaps              department x item        the company mean
    dispersion_screen     item                     the spread across depts
    theme_correlation     department x category    the other themes
    -------------------------------------------------------------------------
    (used by NB02 — the four above the line map to essay §3.1-§3.4)

    department_matrix / _prepare_matrix / select_k / cluster_departments /
    cluster_profiles / compare_clusterings / pca_departments / ipma /
    composite_index                                      (used by NB03)

None of them re-derive the data: reconstruction and harmonisation happened once
in quality.py, and any function here that needed to repeat that work would be a
sign the pipeline had been split in the wrong place.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from engagement.quality import to_matrix


# ------------------------------------------------------------------- internals
def department_matrix(harmonised: pd.DataFrame) -> pd.DataFrame:
    """Department x item matrix with the 'Company' row removed.

    Reuses quality.to_matrix (single source of the pivot) and applies the one
    Phase-1-specific rule: 'Company' is an aggregate, not a department, so it
    must be excluded before any technique that treats each row as an
    observation to be compared with the others (clustering, PCA, distance).
    Including it would add a phantom 20th 'department' at the centre of the
    space and distort both cluster membership and the principal components.
    """
    return to_matrix(harmonised[harmonised["department"] != "Company"], "harmonised")


# ---------------------------------------------------------------- RO1: profile
def theme_ranking(harmonised: pd.DataFrame) -> pd.DataFrame:
    """Company-level ranked strength-weakness profile of the 8 themes.

    How it works
    ------------
    1. Keep only the 'Company' rows. The company mean per item is given in the
       workbook, so no averaging across departments is needed (and averaging
       19 unequally-sized departments would give the wrong answer anyway).
    2. `groupby("category")` on the harmonised item scores, then `.agg` two
       things at once: the mean of the theme's items, and how many items it has
       (`n_items` is reported because a 3-item theme is a less stable estimate
       than a 6-item one).
    3. Sort descending and attach an explicit `rank` column, so the ordering
       survives being copied into the report.

    Why recomputed rather than read: the workbook already contains a header
    value for each theme, but `quality.validate_structure` showed two of those
    headers average 1-10 and 1-5 items together. Those headers are therefore
    never used; every theme score here is rebuilt from harmonised items.

    Worked example: 'Working with my Line Manager' = mean of its 5 harmonised
    items = 4.166, rank 1. The whole ranking spans 3.913 to 4.166, a quarter
    of a scale point, which is the finding reported in essay §3.1.
    """
    company = harmonised[harmonised["department"] == "Company"]
    out = (
        company.groupby("category")["harmonised"]
        .agg(mean="mean", n_items="count")
        .sort_values("mean", ascending=False)
        .round(3)
    )
    out["rank"] = range(1, len(out) + 1)
    return out


# ------------------------------------------------------ RO1/RO3a: benchmarking
def gap_matrix(harmonised: pd.DataFrame) -> pd.DataFrame:
    """Department x category matrix of gaps vs the company mean (harmonised).

    How it works
    ------------
    1. `groupby(["department", "category"])` averages the items inside each
       theme, for every department AND for the 'Company' row — one number per
       department-theme pair.
    2. `.unstack("category")` turns the category level of that index into
       columns, giving a rectangle: 20 rows (19 departments + Company) x 8
       theme columns.
    3. `cat.drop(index="Company")` removes the reference row from the body,
       and `- cat.loc["Company"]` subtracts it from every remaining row.
       Pandas aligns on column NAMES here, not position, so the subtraction is
       safe even if the column order ever changed.

    Positive = above company average, negative = below, so a cell reading -1.09
    puts that department more than a full scale point below the company on that
    theme. This matrix is essay Figure 3 (fig02).

    Note the deltas are NOT simply the workbook's original delta column: those
    were per item and on mixed scales. These are recomputed from harmonised
    item scores, so themes are comparable with each other.
    """
    cat = (
        harmonised.groupby(["department", "category"])["harmonised"]
        .mean()
        .unstack("category")
    )
    return (cat.drop(index="Company") - cat.loc["Company"]).round(3)


def top_gaps(harmonised: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Largest positive and negative department-item gaps vs company mean.

    Where `gap_matrix` works at THEME level (8 columns), this works at ITEM
    level (34 items x 19 departments = 646 gaps) and returns only the extremes.
    Theme averaging hides single items; this function is what recovers them.

    How it works
    ------------
    1. Split the frame: department rows in `items`, the Company rows reshaped
       into a lookup `company` indexed by item.
    2. `merge(company, on="item")` puts each item's company score on the same
       row as the department score, which is the standard tidy way to compare a value
       with its own benchmark.
    3. `gap = harmonised - company_score`, rounded to 3dp.
    4. `nlargest(n, "gap")` and `nsmallest(n, "gap")` take the n biggest gaps in
       each direction; `.assign(direction=...)` labels them before they are
       concatenated, so a reader of the CSV never has to infer the sign.

    Worked example (top row of tab05): department R scores 5.0 on the manager
    resources item against a company score of 3.9, a gap of +1.1, the largest
    positive gap in the dataset. Essay §3.2, Table 3 (tab05).

    Caveat carried into the report: with departments this small, a single
    respondent can move an item this far, so the extremes are read as leads to
    follow up, not as established facts about a team.
    """
    items = harmonised[harmonised["department"] != "Company"].copy()
    company = (
        harmonised[harmonised["department"] == "Company"]
        .set_index("item")["harmonised"]
        .rename("company_score")
    )
    items = items.merge(company, on="item")
    items["gap"] = (items["harmonised"] - items["company_score"]).round(3)
    cols = ["department", "category", "item", "harmonised", "company_score", "gap"]
    worst = items.nsmallest(n, "gap")[cols]
    best = items.nlargest(n, "gap")[cols]
    return pd.concat([best.assign(direction="above"), worst.assign(direction="below")])


# ------------------------------------------------- RO3a: dispersion screening
def dispersion_screen(harmonised: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Classify every item by (company mean x departmental spread).

    The reasoning this function exists for: a low score alone does not tell you
    what to do about it. A low score that every department shares is a policy
    problem; the same low score driven by three departments is a targeted one.
    Level and consistency are two different axes, so the screen uses both.

    Quadrant logic:
      low mean + wide spread   -> 'localised problem'  (specific depts struggling)
      low mean + narrow spread -> 'company-wide problem'
      high mean + wide spread  -> 'uneven strength'
      high mean + narrow spread-> 'uniform strength'

    How it works
    ------------
    1. Drop the Company row, then `groupby(["category", "item"])` and `.agg`
       the standard deviation across departments plus the min and max, the
       spread of experience for that one question.
    2. Join the company mean for the same item back on (MultiIndex
       category/item on both sides, so the join needs no key juggling).
    3. Turn both axes into booleans using QUANTILES from config, not fixed
       numbers: `low_mean` is at/below the low-mean quantile, `wide` is at/above
       the high-spread quantile. Quantiles are used because the scores are
       compressed, an absolute cut-off such as "below 3.5" would flag nothing
       at all on this data. The trade-off is stated in the report: a relative
       threshold always flags roughly a fixed proportion of items, so this is a
       prioritisation device, not an absolute standard.
    4. `np.select` maps the two booleans onto the four quadrant names in one
       vectorised pass (the fourth is the `default`).
    5. `priority_2026` is set to `low_mean` alone: both low-scoring quadrants
       need action, and the spread axis decides WHICH action, not whether.

    Both cut-offs are quantiles of the survey itself rather than fixed values,
    so the screen adapts to an instrument whose overall level has shifted and
    what it flags stays a statement about THIS survey's internal spread rather
    than about an external benchmark. Essay §3.3: Figure 4 (fig02b) and
    Table 4 (tab06).
    """
    depts = harmonised[harmonised["department"] != "Company"]
    stats = depts.groupby(["category", "item"])["harmonised"].agg(
        spread="std", dept_min="min", dept_max="max"
    )
    company = (
        harmonised[harmonised["department"] == "Company"]
        .set_index(["category", "item"])["harmonised"]
        .rename("company_mean")
    )
    out = stats.join(company)
    low_mean = out["company_mean"] <= out["company_mean"].quantile(cfg["phase1"]["low_mean_quantile"])
    wide = out["spread"] >= out["spread"].quantile(cfg["phase1"]["dispersion_high_spread_quantile"])
    out["quadrant"] = np.select(
        [low_mean & wide, low_mean & ~wide, ~low_mean & wide],
        ["localised problem", "company-wide problem", "uneven strength"],
        default="uniform strength",
    )
    out["priority_2026"] = low_mean
    return out.round(3).sort_values(["priority_2026", "spread"], ascending=[False, False])


# ------------------------------------------- exploratory: theme co-movement
def theme_correlation(harmonised: pd.DataFrame) -> pd.DataFrame:
    """8x8 Pearson correlation of category scores ACROSS DEPARTMENTS.

    How it works
    ------------
    1. Drop the Company row (it is an average of the others; leaving it in
       would let the same information count twice).
    2. Average items into themes per department and `unstack`. The same
       19 x 8 rectangle `gap_matrix` builds, but used untouched rather than
       differenced.
    3. `.corr()` correlates the COLUMNS with each other. Each correlation is
       computed over 19 points, one per department, and answers: when a
       department scores highly on theme A, does it also score highly on B?

    Read the unit of analysis carefully, because it is the main way this table
    gets misused. The correlation is between DEPARTMENTS, not employees. A high
    value means departments that do well on one theme do well on another; it
    says nothing about whether an individual who rates their manager highly
    also rates work/life balance highly. Inferring the second from the first is
    the ecological fallacy (Robinson, 1950), and the report states explicitly
    that this table cannot support it.

    What to look for: near-universally positive pairs. Everything moving
    together is the signature of a halo /
    common-method factor, departments that feel good about one thing feel good
    about everything, which is one of the three converging pieces of evidence
    for that finding in the report (essay §3.4, Figure 5 / fig02c).

    Caveats reported verbatim: ~19 units -> very low power, so individual
    coefficients are unstable; treated as hypothesis generation for Phase 2,
    never as a result in itself.
    """
    cat = (
        harmonised[harmonised["department"] != "Company"]
        .groupby(["department", "category"])["harmonised"]
        .mean()
        .unstack("category")
    )
    return cat.corr().round(3)


# ------------------------------------------- multivariate (feasibility-bounded)
def _prepare_matrix(
    harmonised: pd.DataFrame, mode: str = "level"
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the clustering input under one of two explicit framings.

    mode='level' : departments compared on their scores as reported.
        Distance is dominated by *how high* a department scores, so the
        solution answers "which departments are struggling overall?".

    mode='shape' : each department's row is centred on its own mean before
        standardisation (profile analysis / ipsative centring). Overall
        favourability is removed, so distance reflects *relative* strengths and
        weaknesses, and the solution answers "in what way are they struggling?".
        Note this costs one degree of freedom (each centred row sums to zero).

    Both then z-score each item (column) so no item dominates on spread alone.

    Returns (matrix used, standardised array).
    """
    matrix = department_matrix(harmonised)
    if mode == "shape":
        matrix = matrix.sub(matrix.mean(axis=1), axis=0)
    elif mode != "level":
        raise ValueError(f"mode must be 'level' or 'shape', got {mode!r}")

    z = StandardScaler().fit_transform(matrix)

    # Guard: standardisation divides by each item's standard deviation, so a
    # zero-variance item would silently produce inf/NaN and corrupt every
    # distance downstream. Verified explicitly rather than assumed — this also
    # distinguishes a genuine numerical problem from the spurious
    # "divide by zero encountered in matmul" RuntimeWarning that some
    # numpy/BLAS builds emit during matrix multiplication.
    if not np.isfinite(z).all():
        zero_var = matrix.columns[matrix.std(ddof=0) == 0].tolist()
        raise ValueError(
            "Standardised matrix contains non-finite values. "
            f"Zero-variance items: {zero_var or 'none — check for NaNs in input'}"
        )
    return matrix, z


def select_k(
    harmonised: pd.DataFrame, cfg: dict[str, Any], mode: str = "level"
) -> pd.DataFrame:
    """Silhouette scan over the candidate k range (config: phase1.n_clusters_range).

    Replaces an arbitrary choice of k with a stated criterion. Reports the
    silhouette score, the smallest cluster size and the number of singleton
    clusters for each k, because a high silhouette achieved by isolating one
    department is not a segmentation.

    Interpretation convention (Kaufman & Rousseeuw, 1990): > 0.50 reasonable
    structure; 0.25-0.50 weak, may be artificial; < 0.25 no substantial
    structure. Reporting a low score honestly is itself a finding.

    How it works
    ------------
    1. Build the standardised matrix for the requested mode.
    2. `pdist` computes the distance between every pair of departments; `linkage`
       with `method="ward"` merges them bottom-up, always choosing the merge that
       adds least within-cluster variance. This is done ONCE, the hierarchy does
       not depend on k.
    3. For each candidate k, `fcluster(..., criterion="maxclust")` cuts that same
       tree into k groups. Cutting one tree at several heights is why the
       solutions are nested and why the scan is cheap.
    4. `silhouette_score` scores the cut: for each department, how much closer it
       is to its own cluster than to the nearest other one, averaged. +1 is
       perfect separation, 0 is on the boundary.
    5. Alongside the score, record the smallest cluster size and the number of
       singletons. The guard against a "good" solution that is really one
       department pushed into a cluster of its own.

    Read the silhouette and the membership column together: a run whose score
    improves only because it has isolated one department has found an outlier,
    not a segment (essay §3.4).
    """
    _, z = _prepare_matrix(harmonised, mode)
    lnk = linkage(pdist(z, metric="euclidean"), method="ward")
    lo, hi = cfg["phase1"]["n_clusters_range"]
    rows = []
    for k in range(lo, hi + 1):
        labels = fcluster(lnk, t=k, criterion="maxclust")
        sizes = pd.Series(labels).value_counts()
        rows.append(
            {
                "k": k,
                "silhouette": round(float(silhouette_score(z, labels)), 3),
                "smallest_cluster": int(sizes.min()),
                "n_singletons": int((sizes == 1).sum()),
            }
        )
    return pd.DataFrame(rows).set_index("k")


def cluster_departments(
    harmonised: pd.DataFrame,
    cfg: dict[str, Any],
    k: int | None = None,
    mode: str = "level",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Ward hierarchical clustering of departments (see _prepare_matrix for modes).

    Returns (assignments dataframe, linkage matrix for dendrogram plotting).

    Why Ward at n~19: deterministic, stable at small n, and the dendrogram is
    readable by non-technical stakeholders. k-means is rejected here (sensitive
    to initialisation at this sample size) and reserved for Phase 2 personas,
    where n is adequate.

    If k is not supplied it is chosen by the highest silhouette score over the
    configured range, preferring solutions with no singleton clusters.

    How it works
    ------------
    1. Build the matrix and rebuild the same Ward hierarchy `select_k` scored
       (deterministic, so the two agree by construction).
    2. If the caller did not fix k: run the scan, keep only rows with
       `n_singletons == 0`, and take the best silhouette among those. If NO k is
       free of singletons, fall back to the full scan rather than returning
       nothing. The notebook then reports that fact as the result.
    3. Cut the tree at k and label the departments.
    4. Name the label column `cluster_level` or `cluster_shape` after the mode,
       so the two runs can be joined side by side later (`compare_clusterings`)
       without ambiguity.

    Returns the linkage matrix as well as the assignments because the dendrogram
    is drawn from it: the notebook shows the whole hierarchy, not just the cut.
    """
    matrix, z = _prepare_matrix(harmonised, mode)
    lnk = linkage(pdist(z, metric="euclidean"), method="ward")
    if k is None:
        scan = select_k(harmonised, cfg, mode)
        viable = scan[scan["n_singletons"] == 0]
        k = int((viable if not viable.empty else scan)["silhouette"].idxmax())
    labels = fcluster(lnk, t=k, criterion="maxclust")
    assign = pd.DataFrame(
        {"department": matrix.index, f"cluster_{mode}": labels}
    ).set_index("department")
    return assign, lnk


def cluster_profiles(
    harmonised: pd.DataFrame, assign: pd.DataFrame, relative: bool = False
) -> pd.DataFrame:
    """Mean theme profile of each cluster, for interpretation.

    relative=False : absolute theme means + OVERALL mean (reads the LEVEL).
    relative=True  : each theme minus the cluster's own overall mean (reads the
        SHAPE, which themes are distinctively strong/weak for that cluster,
        independent of how favourable it is overall).

    How it works
    ------------
    1. Average items into themes per department, `unstack` to the 19 x 8
       rectangle, drop 'Company'.
    2. `.join(assign)` attaches the cluster label to each department, then
       `.groupby(col).mean()` averages the departments inside each cluster.
    3. `overall = prof.mean(axis=1)` is each cluster's mean across its own eight
       themes, the level it sits at.
    4. `relative=True` subtracts that level from every theme (`prof.sub(overall,
       axis=0)`); `relative=False` keeps the raw means and appends the level as
       an OVERALL column.
    5. Attach `n_departments` so no one reads a cluster of two as if it were a
       cluster of thirteen.

    Why both views are printed in NB03: the pair IS the test. If the clusters
    differ only in overall favourability, the relative table is near zero
    everywhere while the level difference between them is large, and it is that
    contrast which supports or refutes the halo reading in essay §3.4.
    """
    col = assign.columns[0]
    cat = (
        harmonised.groupby(["department", "category"])["harmonised"]
        .mean()
        .unstack("category")
        .drop(index="Company")
    )
    prof = cat.join(assign).groupby(col).mean()
    overall = prof.mean(axis=1)
    if relative:
        out = prof.sub(overall, axis=0)
    else:
        out = prof.copy()
        out["OVERALL"] = overall
    out["n_departments"] = assign[col].value_counts().sort_index()
    return out.round(3)


def compare_clusterings(
    assign_level: pd.DataFrame, assign_shape: pd.DataFrame
) -> pd.DataFrame:
    """Cross-tabulate the two solutions.

    Departments sharing a level-cluster but sitting in different shape-clusters
    are the analytically interesting cases: equally low engagement, different
    underlying problem, therefore different intervention.

    How it works: `join` puts both label columns on the same index (departments
    match by name), then `pd.crosstab` counts how many departments fall into
    each level x shape combination. Off-diagonal mass means the two framings
    disagree, which is the point of running the clustering twice.
    """
    j = assign_level.join(assign_shape)
    return pd.crosstab(j.iloc[:, 0], j.iloc[:, 1])


def pca_departments(harmonised: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """PCA of departments in item space (structure-revealing, not confirmatory).

    What PCA is doing here, in one sentence: instead of asking which departments
    group together (clustering), it asks how few underlying dimensions are
    needed to reproduce the 34 item scores — and what each of those dimensions
    is made of.

    How it works
    ------------
    1. Reuse `_prepare_matrix(..., "level")`, the same standardised 19 x 34
       matrix the clustering used, so the two analyses cannot silently disagree
       about their input.
    2. Cap the number of components at `min(config value, n_rows - 1)`. The
       second term is a hard mathematical ceiling: n observations can support at
       most n-1 components.
    3. `pca.fit_transform(z)` returns the SCORES, where each department sits on
       each new axis.
    4. `pca.components_.T` gives the LOADINGS, how much each original item
       contributes to each axis. Transposed so rows are items, which is the
       orientation a reader interprets from.
    5. `explained_variance_ratio_` reports how much of the total variation each
       axis carries.

    Scores vs loadings, the distinction that makes the output readable:
    loadings say WHAT an axis measures (read the items at each end), scores say
    WHERE each department falls on it. The biplot in NB03 draws both together.

    Boundary. p (34 items) > n (19 units), so components are unstable and the
    result is structure-revealing only: it may suggest how many dimensions exist,
    it cannot confirm a factor structure. That test needs individual-level data
    and is deferred to Phase 2, part of the RO4 argument.

    The reading of PC1 is tested rather than asserted: it is correlated against
    the composite index built independently in `composite_index`, and a
    correlation near 1 means PC1 simply IS overall favourability. PC2 typically
    opposes proximal items against distal ones. Essay §3.4, Figures 8-9.
    """
    matrix, z = _prepare_matrix(harmonised, "level")
    n_comp = min(cfg["phase1"]["pca_max_components"], matrix.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=cfg["project"]["seed"])
    scores = pca.fit_transform(z)
    return {
        "scores": pd.DataFrame(
            scores, index=matrix.index, columns=[f"PC{i+1}" for i in range(n_comp)]
        ),
        "loadings": pd.DataFrame(
            pca.components_.T,
            index=matrix.columns,
            columns=[f"PC{i+1}" for i in range(n_comp)],
        ),
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def ipma(harmonised: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Importance-Performance map data (the priority-setting output).

    Importance proxy: correlation of each theme with the outcome anchors
    (recommendation / happiness items) ACROSS departments. Performance: theme
    company mean. High-importance + low-performance = priority quadrant.

    ANCHOR EXCLUSION (circularity correction)
    -----------------------------------------
    Both configured anchors are items *inside* 'Working for AccessFintech'.
    Scoring that theme from all of its items and then correlating it against a
    mean of two of those same items is circular. The theme is partly
    correlated with itself, which inflated its importance to .92 and put it top
    of the priority list. Each theme's score is therefore computed from its
    NON-ANCHOR items only, so no item appears on both sides of any correlation.

    The uncorrected value is retained as `importance_naive` and the affected
    theme carries `anchor_adjusted = True`, so the size of the correction is
    auditable rather than silently applied.

    Performance is left over ALL items: it is a mean, not a correlation, so it
    carries no circularity, and the client should see the theme score as
    reported.

    How it works
    ------------
    1. Drop 'Company'; mark which rows are anchor items (`is_anchor`).
    2. `anchors` = each department's mean over the anchor items, one number per
       department, the outcome to correlate against.
    3. `_cat_scores` is defined once and applied twice: to ALL items (`cat_all`)
       and to non-anchor items only (`cat_adj`). Writing it as a local helper
       makes it self-evident that the two tables differ ONLY in which items went
       in.
    4. `corrwith(anchors)` correlates each theme column with the anchor Series
       across the ~19 departments. Pandas aligns on the department index, so
       the 8 correlations come back in one call.
    5. Performance is the company theme mean, over all items.
    6. Assemble, flag `anchor_adjusted`, record `importance_shift` (corrected −
       naive), and mark the priority quadrant: importance above its median AND
       performance below its median.

    What matters is not the size of the shift but whether a theme crosses a
    quadrant boundary because of it. A conclusion that survives its own
    correction can be reported; one that does not, cannot. Essay §3.5,
    Figure 10.

    Small-n caveat (~19 departments) applies; reported as exploratory.
    """
    depts = harmonised[harmonised["department"] != "Company"]
    is_anchor = depts["item"].isin(cfg["data"]["outcome_anchors"])

    anchors = (
        depts[is_anchor].groupby("department")["harmonised"].mean().rename("anchor")
    )

    def _cat_scores(frame: pd.DataFrame) -> pd.DataFrame:
        return (
            frame.groupby(["department", "category"])["harmonised"].mean().unstack("category")
        )

    cat_all = _cat_scores(depts)
    # reindex: a theme made up ENTIRELY of anchor items would vanish here;
    # it must surface as NaN importance, not disappear from the map.
    cat_adj = _cat_scores(depts[~is_anchor]).reindex(columns=cat_all.columns)

    importance = cat_adj.corrwith(anchors).rename("importance")
    importance_naive = cat_all.corrwith(anchors).rename("importance_naive")
    performance = (
        harmonised[harmonised["department"] == "Company"]
        .groupby("category")["harmonised"]
        .mean()
        .rename("performance")
    )

    out = pd.concat([importance, importance_naive, performance], axis=1).round(3)
    out["anchor_adjusted"] = out.index.isin(depts.loc[is_anchor, "category"].unique())
    out["importance_shift"] = (out["importance"] - out["importance_naive"]).round(3)
    out["priority"] = (out["importance"] > out["importance"].median()) & (
        out["performance"] < out["performance"].median()
    )
    return out.sort_values(["priority", "importance"], ascending=[False, False])


def composite_index(harmonised: pd.DataFrame, weights: dict[str, float] | None = None) -> pd.DataFrame:
    """Literature-weighted composite engagement index per department.

    Default: equal weights (transparent baseline). Pass a weights dict keyed by
    category to apply literature-informed weights (justified in the report from
    the consolidated factor set, not assigned arbitrarily).

    How it works
    ------------
    1. Build the department x theme table (Company kept this time, it is the
       benchmark every department is compared against).
    2. No weights: `cat.mean(axis=1)`, the plain average of the eight themes.
    3. With weights: `reindex(cat.columns)` puts the dict in column order and
       fills any theme the caller forgot with 0; `w / w.sum()` normalises so the
       index stays on the 1-5 scale whatever numbers were passed; `cat.mul(w,
       axis=1).sum(axis=1)` is the weighted mean.
       The `fillna(0)` is deliberate: a missing theme is dropped rather than
       silently given a share, and normalising afterwards keeps the arithmetic
       honest.
    4. Add `vs_company` and the department's distance from the Company row then
       sort, so the output reads as a ranking.

    Why equal weights are the default: with eight strongly inter-correlated
    themes, unit weights perform close to optimally and are fully transparent
    (OECD/JRC, 2008; Wainer, 1976; Dawes, 1979). A differential weight set would
    need evidence this dataset does not contain. The `weights` argument exists
    so the sensitivity of the ranking can be TESTED rather than assumed away.

    The comparison the index exists to support is the spread BETWEEN
    departments against the spread between themes. When the first is clearly
    the larger, the actionable variation is departmental rather than thematic,
    and benchmarking beats theme ranking as a way in.
    Also used in NB03 as the independent check on PC1.
    """
    cat = (
        harmonised.groupby(["department", "category"])["harmonised"].mean().unstack("category")
    )
    if weights:
        w = pd.Series(weights).reindex(cat.columns).fillna(0)
        w = w / w.sum()
        idx = cat.mul(w, axis=1).sum(axis=1)
    else:
        idx = cat.mean(axis=1)
    out = idx.rename("composite_index").to_frame().round(3)
    out["vs_company"] = (out["composite_index"] - out.loc["Company", "composite_index"]).round(3)
    return out.sort_values("composite_index", ascending=False)
