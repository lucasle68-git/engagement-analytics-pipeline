"""Phase 2 segmentation — engagement personas (RO3b/RO1).

k-means on standardised construct scores, k selected by silhouette scan
(same discipline as Phase 1: criterion stated before the solution, singleton
guard) with an agglomerative cross-check (adjusted Rand index). Personas are
profiled in plain-language terms — share, distinctive traits, risk rate —
because the client-facing value is the *description*, not the labels.

Demonstrates the demographic targeting the aggregated real data cannot
deliver. All outputs are method demonstrations on synthetic data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from engagement.models import DRIVER_CONSTRUCTS, _quiet_blas


def build_personas(model_df: pd.DataFrame, cfg: dict[str, Any],
                   k: int | None = None) -> dict[str, Any]:
    """Cluster wave-1 employees into personas on construct profiles.

    Returns dict: k_scan, assignments, profiles (workforce z-units), demographic
    mix per persona, agreement (ARI with hierarchical cross-check).

    How it works
    ------------
    1. Wave 1 only (one row per employee, later waves would count people twice),
       standardised so no construct dominates on spread alone.
    2. Finiteness guard before clustering, so any BLAS warning during k-means can
       be ruled out as spurious rather than assumed to be (same discipline as
       `phase1._prepare_matrix`).
    3. Silhouette scan across the configured k range, recording the smallest
       cluster and singleton count. The criterion is stated before any solution
       is shown, exactly as in Phase 1.
    4. Fit at the chosen k, then fit agglomerative clustering at the same k and
       compare with the adjusted Rand index. ARI is chance-corrected: 0 means no
       better than random agreement, 1 identical. Two different algorithms
       agreeing is evidence the grouping is in the data, not in the algorithm.
    5. Profile each persona as z-scores against the workforce: "how far from the
       average employee, per construct", then attach share and risk rate.
    6. Cross-tabulate demographics per persona, normalised by row, which is the
       targeting the aggregated real data cannot produce.

    Why k-means here and Ward in Phase 1: at n≈19 k-means is unstable across
    initialisations, so Ward's deterministic hierarchy was used; at n=1,000 with
    `n_init=20` that objection disappears and k-means scales better. Same
    standard, different sample size. The selection rule is identical in
    both, which is what makes the low silhouette here reportable rather than
    embarrassing.
    """
    seed = cfg["project"]["seed"]
    w1 = model_df[model_df["wave"] == 1].reset_index(drop=True)
    X = StandardScaler().fit_transform(w1[DRIVER_CONSTRUCTS + ["engagement_score"]])

    # Guard: verify the standardised matrix is finite BEFORE clustering, so
    # any numerical RuntimeWarning from the BLAS layer during k-means can be
    # ruled out as spurious (same discipline as phase1._prepare_matrix).
    if not np.isfinite(X).all():
        raise ValueError("Standardised persona matrix contains non-finite values "
                         "— check construct scores for NaNs or zero variance.")

    lo, hi = cfg["phase1"]["n_clusters_range"]
    rows = []
    # _quiet_blas(): the finiteness guard above has already ruled out a genuine
    # numerical fault, so the BLAS matmul flags k-means raises here are the
    # spurious kind documented in models._quiet_blas.
    with _quiet_blas():
        for kk in range(lo, hi + 1):
            lab = KMeans(n_clusters=kk, n_init=20, random_state=seed).fit_predict(X)
            sizes = pd.Series(lab).value_counts()
            rows.append({"k": kk,
                         "silhouette": round(float(silhouette_score(X, lab)), 3),
                         "smallest_cluster": int(sizes.min()),
                         "n_singletons": int((sizes == 1).sum())})
        scan = pd.DataFrame(rows).set_index("k")
        if k is None:
            viable = scan[scan["n_singletons"] == 0]
            k = int((viable if not viable.empty else scan)["silhouette"].idxmax())

        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        w1 = w1.assign(persona=km.fit_predict(X) + 1)
        hier = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    ari = round(float(adjusted_rand_score(w1["persona"], hier)), 3)

    cols = DRIVER_CONSTRUCTS + ["engagement_score"]
    prof_abs = w1.groupby("persona")[cols].mean()
    # profile = persona mean expressed in workforce z-units (how far this
    # persona sits from the average employee, per construct) — comparable
    # across constructs and readable for non-technical audiences
    profiles = ((prof_abs - w1[cols].mean()) / w1[cols].std(ddof=0)).round(2)
    profiles["share_%"] = (w1["persona"].value_counts(normalize=True)
                           .sort_index() * 100).round(1)
    profiles["risk_rate_%"] = (w1.groupby("persona")["at_risk"].mean() * 100).round(1)

    demo = {c: (pd.crosstab(w1["persona"], w1[c], normalize="index") * 100).round(1)
            for c in ["work_mode", "tenure", "role_level", "region"]}
    return {"k": k, "k_scan": scan, "assignments": w1[["employee_id", "persona"]],
            "profiles": profiles, "demographics": demo, "hierarchical_ari": ari}
