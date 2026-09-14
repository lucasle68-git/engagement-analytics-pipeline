"""Phase 2 modelling: drivers, risk, explainability (RO2, RO3b).

Every function operates on the SYNTHETIC panel and demonstrates method
capability, never company fact. Design principles (methodology §4.3.5):
an interpretable baseline is always estimated alongside a performance model;
models are selected on recall and PR-AUC (missing an at-risk employee costs
more than a false alert); temporal validation (train early waves, test the
latest) demonstrates real-world applicability; explanations are triangulated.

Performance model: histogram gradient boosting as implemented in scikit-learn
(HistGradientBoostingClassifier, the LightGBM algorithm family) rather than an
external XGBoost/LightGBM dependency: same model class, one fewer moving part
for markers to install. Deep learning is deliberately excluded (unjustified for
tabular survey data of this size). SHAP is used when available, with
permutation importance as the always-available fallback and triangulation.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import cross_val_predict
from statsmodels.stats.outliers_influence import variance_inflation_factor

from engagement.quality import cronbach_alpha
from engagement.synthetic import _harmonised_block, construct_scores

# Constructs used as PREDICTORS. 'satisfaction' is excluded by design: it is
# outcome-adjacent (engagement<->satisfaction r=.60 imposed), and Phase 1's
# IPMA circularity finding showed what happens when an outcome-region variable
# is allowed to pose as a driver.
DRIVER_CONSTRUCTS = ["personal", "development", "social", "balance",
                     "justice", "technology", "psych_safety", "esg"]
DEMOGRAPHIC_COLS = ["region", "tenure", "role_level", "work_mode"]


@contextmanager
def _quiet_blas():
    """Silence the spurious BLAS floating-point flags raised during matmul.

    On some numpy builds (notably macOS/Accelerate) the FP status flags set
    inside the BLAS call are read back by numpy afterwards and reported as
    divide-by-zero, overflow AND invalid-value on the same line. All three
    at once, which no genuine numerical fault produces.

    Suppressed HERE, inside the library, rather than in the notebook, because
    a notebook-level `warnings.filterwarnings` does not survive scikit-learn's
    internal `catch_warnings` blocks. `np.errstate` acts at the numpy level,
    below the warnings machinery, so it holds for the whole call.

    Safe because the conditions these flags would indicate are checked
    directly: models operate on bounded ordinal construct scores, and
    personas.build_personas raises if its standardised matrix is non-finite.
    """
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"), \
            warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*encountered in matmul.*",
                                category=RuntimeWarning)
        yield


def prepare(data: pd.DataFrame, item_map: pd.DataFrame,
            cfg: dict[str, Any]) -> pd.DataFrame:
    """Employee-wave modelling table: construct scores + demographics + targets.

    engagement_score : mean of the UWES block (the direct outcome the 2024
        survey lacks, the reason it must be added in 2026).
    at_risk : engagement in (approximately) the bottom
        `models.risk_threshold_quantile` of the TRAINING waves' distribution.
        The threshold is learned on waves < max (no test leakage), and the cut
        uses a strict `<` against the lower-interpolated quantile: the
        engagement score takes only ~13 discrete values, so `<= quantile`
        would sweep in every employee tied at the cut value and inflate
        prevalence (measured: 30.4% instead of the intended 20%).
    """
    scores = construct_scores(data, item_map)
    out = pd.concat([data[["employee_id", "department", "wave"] + DEMOGRAPHIC_COLS]
                     .reset_index(drop=True),
                     scores.reset_index(drop=True)], axis=1)
    out["engagement_score"] = scores["engagement"].to_numpy()
    train_mask = out["wave"] < out["wave"].max()
    train = out.loc[train_mask, "engagement_score"]
    thr = float(np.quantile(train, cfg["models"]["risk_threshold_quantile"],
                            method="lower"))
    out["at_risk"] = (out["engagement_score"] < thr).astype(int)  # tie-safe `<`
    return out


def construct_validation(data: pd.DataFrame, item_map: pd.DataFrame) -> pd.DataFrame:
    """Cronbach's alpha per construct (wave 1), the reliability audit the real
    data could not support, demonstrating how 2026 scales would be verified.

    Alphas are computed on HARMONISED item blocks (1-10 items rescaled onto
    1-5): alpha weights items by variance, so a raw 1-10 item would dominate
    its block. The NB01 mixed-scale rule applies to reliability too.
    """
    w1 = data[data["wave"] == 1]
    rows = []
    for con in item_map["construct"].unique():
        block = _harmonised_block(w1, item_map, con)
        rows.append({"construct": con, "n_items": block.shape[1],
                     "alpha": round(cronbach_alpha(block), 3)})
    return (pd.DataFrame(rows).set_index("construct")
            .sort_values("alpha", ascending=False))


def driver_analysis(model_df: pd.DataFrame) -> pd.DataFrame:
    """OLS driver ranking with VIF and Johnson relative weights (RO2).

    Predictors standardised so coefficients are comparable; relative weights
    decompose R² into non-negative shares that are robust to the predictor
    inter-correlations imposed by the copula (Johnson, 2000). The check that
    betas alone are not artefacts of collinearity.
    """
    X = model_df[DRIVER_CONSTRUCTS].apply(lambda s: (s - s.mean()) / s.std(ddof=0))
    y = (model_df["engagement_score"] - model_df["engagement_score"].mean()) \
        / model_df["engagement_score"].std(ddof=0)
    with _quiet_blas():
        ols = sm.OLS(y, sm.add_constant(X)).fit()

        Xc = sm.add_constant(X)
        vif = {c: variance_inflation_factor(Xc.to_numpy(), i)
               for i, c in enumerate(Xc.columns) if c != "const"}

        # Johnson's relative weights via SVD of the predictor correlation structure
        U, S, Vt = np.linalg.svd(X.to_numpy(), full_matrices=False)
        Z = U @ Vt                                   # orthogonal approximation of X
        lam = Vt.T @ np.diag(S) @ Vt / np.sqrt(len(X))   # X = Z @ lam
        beta_z = Z.T @ y.to_numpy() / len(X)
        raw = (lam**2) @ (beta_z**2)
        rw = 100 * raw / raw.sum()

    out = pd.DataFrame({
        "beta_std": ols.params.drop("const").round(3),
        "p_value": ols.pvalues.drop("const").round(4),
        "VIF": pd.Series(vif).round(2),
        "relative_weight_%": pd.Series(rw, index=X.columns).round(1),
    }).sort_values("relative_weight_%", ascending=False)
    out.attrs["r_squared"] = round(float(ols.rsquared), 3)
    return out


def risk_models(model_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Baseline vs performance models on the at-risk flag, with temporal
    validation: train on earlier waves, test on the latest (RO3b)."""
    seed = cfg["project"]["seed"]
    # .astype(float): get_dummies returns BOOL columns (pandas >= 2.0); mixed
    # with the float construct scores the frame converts to an OBJECT array,
    # which scikit-learn must re-cast on every fit and every CV fold.
    feats = pd.get_dummies(model_df[DRIVER_CONSTRUCTS + DEMOGRAPHIC_COLS],
                           columns=DEMOGRAPHIC_COLS, drop_first=True).astype(float)
    y = model_df["at_risk"]
    test = model_df["wave"] == model_df["wave"].max()
    Xtr, Xte, ytr, yte = feats[~test], feats[test], y[~test], y[test]

    models = {
        "logistic (baseline)": LogisticRegression(
            class_weight="balanced", max_iter=3000, random_state=seed),
        "random forest (cross-check)": RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=seed),
        "hist gradient boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=seed),
    }
    rows, fitted = [], {}
    with _quiet_blas():
        for name, m in models.items():
            m.fit(Xtr, ytr)
            proba = m.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            # Screening threshold tuned for >=75% recall on OUT-OF-FOLD train
            # probabilities (cross_val_predict), never in-sample ones: a random
            # forest memorises its training set (~100% train recall at any cut),
            # so an in-sample threshold is meaningless on new data — the audit
            # measured recall@tuned collapsing to 0.02 with the in-sample version.
            # n_jobs left serial: joblib workers are separate processes and do
            # NOT inherit the parent's numpy error state, so parallelism
            # reintroduces the spurious BLAS warnings _quiet_blas() suppresses.
            # At n~2,000 x 20 features the serial CV costs a fraction of a second.
            ptr = cross_val_predict(m, Xtr, ytr, cv=5,
                                    method="predict_proba")[:, 1]
            thr_grid = np.unique(ptr)
            ok = [t for t in thr_grid if recall_score(ytr, ptr >= t) >= 0.75]
            thr = max(ok) if ok else 0.5
            pred_t = (proba >= thr).astype(int)
            rows.append({
                "model": name,
                "recall@0.5": recall_score(yte, pred),
                "precision@0.5": precision_score(yte, pred),
                "recall@tuned": recall_score(yte, pred_t),
                "precision@tuned": precision_score(yte, pred_t),
                "pr_auc": average_precision_score(yte, proba),
                "roc_auc": roc_auc_score(yte, proba),
                "brier": brier_score_loss(yte, proba),
            })
            fitted[name] = m
    metrics = pd.DataFrame(rows).set_index("model").round(3)
    # Purpose-aligned selection: PR-AUC alone rewards ranking quality; an HR
    # screen also needs a USABLE operating point, so the winner must clear a
    # 60% tuned-recall floor (falls back to plain PR-AUC if none does).
    usable = metrics[metrics["recall@tuned"] >= 0.60]
    best = (usable if not usable.empty else metrics)["pr_auc"].idxmax()
    return {"metrics": metrics, "fitted": fitted, "best": best,
            "X_test": Xte, "y_test": yte, "feature_names": list(feats.columns),
            "train_prevalence": round(float(ytr.mean()), 3)}


def explain(rm: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Global driver importance + one-employee story for the selected model.

    SHAP is attempted with an EXPLICIT TreeExplainer first (the generic
    shap.Explainer auto-detection raises TypeError on some model/version
    combinations), then the generic explainer; permutation importance
    (recall-scored) is the always-available triangulation.

    `example_profile` is computed regardless of SHAP: the flagged employee's
    construct scores in workforce z-units ("1.4 sd below average on
    development ..."), so the per-employee narrative exhibit exists even where
    SHAP cannot run — with the substitution stated honestly.
    """
    model, X, y = rm["fitted"][rm["best"]], rm["X_test"], rm["y_test"]
    out: dict[str, Any] = {}
    with _quiet_blas():
        perm = permutation_importance(
            model, X, y, scoring="recall", n_repeats=15,
            random_state=cfg["project"]["seed"])
        out["permutation"] = (pd.Series(perm.importances_mean, index=X.columns)
                              .sort_values(ascending=False).round(4))
        try:
            import shap
            try:                                 # explicit first (avoids auto-detect TypeError)
                sv = shap.TreeExplainer(model).shap_values(X)
                if isinstance(sv, list):         # older shap: [class0, class1]
                    sv = sv[1]
                if getattr(sv, "ndim", 2) == 3:  # newer shap: (n, features, classes)
                    sv = sv[:, :, 1]
            except Exception:
                sv = shap.Explainer(model, X)(X).values
            out["shap_global"] = (pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
                                  .sort_values(ascending=False).round(4))
        except Exception as e:                    # shap optional at runtime
            out["shap_error"] = (f"SHAP unavailable ({type(e).__name__}); "
                                 "permutation importance + z-score profile used instead.")

        # one-employee story (always available): highest-risk test employee,
        # construct scores vs the workforce, in sd units
        proba = model.predict_proba(X)[:, 1]
    i = int(np.argmax(proba))
    cons = [c for c in DRIVER_CONSTRUCTS if c in X.columns]
    z = ((X.iloc[i][cons] - X[cons].mean()) / X[cons].std(ddof=0))
    out["example_profile"] = z.astype(float).sort_values().round(2)
    out["example_risk_proba"] = round(float(proba[i]), 3)
    return out
