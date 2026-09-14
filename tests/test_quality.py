"""Three checks on the data preparation described in report Section 2.2.

One test per preparation step, so a failure names the step that broke:

    1. reconstruction   department delta + company mean -> absolute score
    2. harmonisation    the two 1-10 items rescaled onto the 1-5 metric
    3. validation       a malformed workbook fails loudly instead of quietly

The fixture is a small FAKE heatmap with the same structure as the client
workbook, so the tests never depend on (or ship) client data.
"""

import pandas as pd
import pytest

from engagement import io, quality

CFG = {
    "paths": {"sheet_name": "results"},
    "data": {
        "company_col": "Company Overall (107)",
        "categories": ["Theme A", "Theme B"],
        "ten_point_items": ["Q3 (1-10)"],
        "outcome_anchors": ["Q1"],
    },
    "quality": {"structure_tolerance": 0.05, "harmonisation": "rescale_1_5"},
}


@pytest.fixture
def fake_raw() -> pd.DataFrame:
    """2 categories, 3 items; Q3 is 1-10; two departments stored as deltas."""
    return pd.DataFrame(
        {
            "Category / Question": ["Theme A", "Q1", "Q2", "Theme B", "Q3 (1-10)"],
            "Company Overall (107)": [4.0, 4.5, 3.5, 8.0, 8.0],
            "X": [0.1, 0.2, 0.0, -0.5, -0.5],
            "Y": [-0.1, -0.2, 0.0, 1.0, 1.0],
            "manager": [None] * 5,
        }
    )


def test_reconstruction_recovers_absolute_scores(fake_raw):
    """Step 1. Every department score in the workbook is a DELTA from the
    company mean, so the whole analysis rests on adding the two back together.

    Two things are checked: the arithmetic on named cells, and the invariant
    that the step is information-preserving — subtracting the company mean from
    a reconstructed score must return the original delta, for every cell.
    """
    tidy = io.tidy(fake_raw, CFG)
    abs_df = quality.reconstruct_absolute(tidy)

    # X on Q1: 4.5 + 0.2 = 4.7 ; Y on Q3: 8.0 + 1.0 = 9.0
    q1x = abs_df[(abs_df["item"] == "Q1") & (abs_df["department"] == "X")]
    q3y = abs_df[(abs_df["item"] == "Q3 (1-10)") & (abs_df["department"] == "Y")]
    assert q1x["abs_value"].iloc[0] == pytest.approx(4.7)
    assert q3y["abs_value"].iloc[0] == pytest.approx(9.0)

    # Company rows are already absolute and must pass through untouched
    company_q1 = abs_df[(abs_df["department"] == "Company") & (abs_df["item"] == "Q1")]
    assert company_q1["abs_value"].iloc[0] == 4.5

    # Round trip: absolute - company mean == the delta we started from
    depts = abs_df[abs_df["department"] != "Company"]
    recovered = depts["abs_value"] - depts["company_mean"]
    pd.testing.assert_series_equal(recovered, depts["value"], check_names=False)


def test_harmonisation_rescales_only_the_ten_point_items(fake_raw):
    """Step 2. Two questions use a 1-10 scale and the rest use 1-5. Averaging
    them untreated inflates any theme that mixes them, which is the defect
    reported in Table 2.

    The map is v5 = 1 + (v10 - 1) * 4/9, and it must touch nothing else.
    """
    tidy = io.tidy(fake_raw, CFG)
    harmonised = quality.harmonise_scales(quality.reconstruct_absolute(tidy), CFG)

    company = harmonised[harmonised["department"] == "Company"]
    q3 = company[company["item"] == "Q3 (1-10)"]["harmonised"].iloc[0]
    q1 = company[company["item"] == "Q1"]["harmonised"].iloc[0]

    assert q3 == pytest.approx(1 + 7 * 4 / 9)   # 8.0 on 1-10 -> 4.11 on 1-5
    assert q1 == pytest.approx(4.5)             # already 1-5, unchanged
    assert harmonised["harmonised"].between(1, 5).all()


def test_a_malformed_workbook_fails_loudly(fake_raw):
    """Step 3. The 2026 file will be produced by someone else. If a column is
    renamed or a category header reworded, the pipeline must stop rather than
    analyse a file it has misunderstood — the guarantee behind recommendation
    C3 in the report.

    Three failure modes, each of which would otherwise produce a plausible but
    wrong result.
    """
    # (a) a well-formed file passes silently
    io.validate_raw_structure(fake_raw, CFG)

    # (b) no company column -> delta reconstruction would be meaningless
    renamed = fake_raw.rename(columns={"Company Overall (107)": "Overall"})
    with pytest.raises(ValueError, match="Expected company column"):
        io.validate_raw_structure(renamed, CFG)

    # (c) a reworded category header -> io.tidy would mis-assign every item
    #     beneath it to the wrong theme, silently
    reworded = fake_raw.copy()
    reworded.loc[0, "Category / Question"] = "Theme A renamed"
    with pytest.raises(ValueError, match="Category headers missing"):
        io.validate_raw_structure(reworded, CFG)

    # (d) a corrupted header value is caught by the arithmetic check, which is
    #     what produced the mixed-scale finding in Table 2
    corrupted = fake_raw.copy()
    corrupted.loc[0, "Company Overall (107)"] = 4.9      # true mean is 4.0
    report = quality.validate_structure(io.tidy(corrupted, CFG), CFG)
    assert not report[report["category"] == "Theme A"]["within_tolerance"].iloc[0]
