"""Three checks on the synthetic generator described in report Section 4.1.

One test per claim made about the generated data:

    1. it reproduces the real 2024 department means, and does so identically
       on every re-run (quality gate 1)
    2. the correlations it imposes from the literature are the ones that come
       back out (quality gate 2)
    3. both gates BLOCK — they fail, rather than merely report, when a
       threshold is violated

The fixture is a small FAKE targets matrix with the same structure the real
pipeline produces, so the tests never depend on (or ship) client data. Real
category names are used because the generator maps categories -> constructs.
"""

import numpy as np
import pandas as pd
import pytest

from engagement import synthetic

CATS = ["My role at Access Fintech", "Working with my Team",
        "Work/Life Balance", "Work/Life Balance", "Working for AccessFintech"]
ITEMS = ["item role 1", "item team 1", "item balance (1-10)",
         "item balance 2", "item overall 1"]
SCALES = [5, 5, 10, 5, 5]
DEPTS = ["X", "Y", "Z"]


@pytest.fixture
def small_cfg() -> dict:
    return {
        "project": {"seed": 123},
        "paths": {"synthetic_dir": "/tmp/engagement_test_synth"},
        "synthetic": {
            "n_employees": 400, "n_waves": 2,
            "calibration_tolerance": 0.05,
            "target_correlations": {
                "personal": 0.48, "development": 0.45, "social": 0.36,
                "demands": -0.30, "satisfaction": 0.60, "justice": 0.30,
                "technology": 0.35, "psych_safety": 0.35, "esg": 0.25,
            },
            "item_loading": 0.70, "within_dept_sd": 0.85,
            "wave_drift_sd": 0.06, "max_calibration_iter": 25,
            "halo_strength": 0.5,
            "min_engagement_alpha": 0.60,
            # looser than production: construct scores on n=400 with 2-4 item
            # scales carry sampling noise the production n=1000 does not
            "max_corr_gap": 0.15,
        },
    }


@pytest.fixture
def small_targets() -> pd.DataFrame:
    vals = {"item role 1": [3.8, 4.1, 3.5],
            "item team 1": [4.2, 4.0, 3.9],
            "item balance (1-10)": [7.5, 8.0, 6.9],
            "item balance 2": [3.9, 4.1, 3.5],
            "item overall 1": [4.0, 4.3, 3.6]}
    return pd.DataFrame(vals, index=DEPTS)


@pytest.fixture
def small_meta() -> pd.DataFrame:
    return pd.DataFrame({"item": ITEMS, "category": CATS, "scale": SCALES})


def test_generator_is_calibrated_and_reproducible(small_cfg, small_targets, small_meta):
    """Gate 1. The claim is that aggregating the synthetic individuals back up
    recovers the real department means. If that fails, nothing built on this
    data means anything.

    Reproducibility is checked alongside it: the seed in config must make two
    runs byte-identical, otherwise no figure in the report can be re-created.
    """
    data, item_map = synthetic.generate(small_cfg, small_targets, small_meta)
    fidelity = synthetic.validate_fidelity(data, item_map, small_targets, small_cfg)
    assert fidelity["passed"], f"worst error {fidelity['worst_error']}"

    again, _ = synthetic.generate(small_cfg, small_targets, small_meta)
    pd.testing.assert_frame_equal(data, again)


def test_imposed_correlations_come_back_out(small_cfg, small_targets, small_meta):
    """Gate 2. The generator's whole purpose is to carry a known dependence
    structure, so the structure has to survive the copula, the discretisation
    and the calibration loop.

    Two levels of check. The structural one catches the classic Cholesky
    mistake, `@ L` instead of `@ L.T`; the statistical one cannot, because the
    first column of L equals the first row of R, so the engagement row alone
    still looks roughly right. The statistical check then confirms the
    correlations survive the rest of the pipeline.
    """
    _, R = synthetic._construct_corr_matrix(
        dict(small_cfg["synthetic"]["target_correlations"]))
    L = np.linalg.cholesky(R)
    assert np.allclose(L @ L.T, R, atol=1e-10)
    assert not np.allclose(L.T @ L, R, atol=1e-3)   # documents the failure mode

    data, item_map = synthetic.generate(small_cfg, small_targets, small_meta)
    plausibility = synthetic.validate_plausibility(data, item_map, small_cfg)
    assert plausibility["worst_corr_gap"] <= small_cfg["synthetic"]["max_corr_gap"], \
        plausibility["achieved_vs_target"].to_string()
    assert plausibility["passed"]      # also covers scale bounds and reliability


def test_the_gates_block_rather_than_report(small_cfg, small_targets, small_meta):
    """A quality gate that only prints a warning is documentation, not control.

    Forcing an impossible reliability threshold must flip `passed` to False —
    which is what makes the notebook's `assert` halt the run.
    """
    data, item_map = synthetic.generate(small_cfg, small_targets, small_meta)
    impossible = {**small_cfg,
                  "synthetic": {**small_cfg["synthetic"], "min_engagement_alpha": 0.99}}
    plausibility = synthetic.validate_plausibility(data, item_map, impossible)
    assert not plausibility["passed"]
