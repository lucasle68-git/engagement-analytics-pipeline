"""Shared helpers for every page of the stakeholder app: provenance banners and
table/figure loading.

Why this module exists
----------------------
Two reasons, and the first is the governance one.

1. **The banner is a data-governance control, not decoration.** Every page of this
   app states what kind of data it is showing before it shows anything, exactly as
   every figure in `outputs/` carries a provenance badge stamped by
   `engagement.viz.save_figure`. Putting the banner text in one place means a page
   cannot quietly disagree with the rest of the project about what it is displaying
   — and, critically, `survey_badge()` reads `data.provenance` from
   `config/config.yaml`, so when the pipeline has been run on the bundled demo
   workbook the app says DEMO rather than claiming real client data. The same
   switch, the same guarantee, in the app as in the figures.

2. Four pages were each loading CSVs with their own copy of the same six lines.

The app only ever READS what the pipeline wrote. It recomputes nothing, so a
number shown here cannot drift from the number in the notebook that produced it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"


def provenance() -> str:
    """Return "demo" or "real" — what kind of workbook the pipeline last ran on.

    Falls back to "real" if config cannot be read, because that is the claim that
    demands the most caution from the reader, not the least.
    """
    try:
        cfg = yaml.safe_load((ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
        return str(cfg.get("data", {}).get("provenance", "real")).lower()
    except Exception:
        return "real"


def survey_badge() -> None:
    """Banner for a page built from the survey workbook (Phase 1)."""
    if provenance() == "demo":
        st.info(
            "🟣 **Demo data** — invented numbers with the same structure as the client's "
            "survey. The client file is not distributed with this project, so every figure "
            "on this page demonstrates the analysis rather than reporting a company result."
        )
    else:
        st.success(
            "🟢 **Real 2024 survey data** — aggregated and anonymised; no individual "
            "is identifiable."
        )


def synthetic_badge() -> None:
    """Banner for a page built from the generated panel (Phase 2)."""
    st.warning(
        "🟠 **Synthetic demonstration data** — shows what the analysis *can do* once the "
        "right data exists. Not facts about any real workforce."
    )


def load_table(name: str) -> pd.DataFrame | None:
    """Read `outputs/tables/<name>.csv`, or None if the pipeline has not been run."""
    path = TABLES / f"{name}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def figure(name: str, caption: str = "") -> None:
    """Show `outputs/figures/<name>.png` if the pipeline has produced it."""
    path = FIGURES / f"{name}.png"
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)


def require(*frames: pd.DataFrame | None) -> None:
    """Stop the page with an actionable message if any required table is missing."""
    if any(f is None for f in frames):
        st.error(
            "Pipeline outputs not found. From the project root run:\n\n"
            "```\nmake demo-data\nmake pipeline\n```\n\nthen reload this page."
        )
        st.stop()


_THEME_PREFIXES = (
    "Working with the ", "Working with my ", "Working with ", "Working for ",
)


def short_theme(name: str, limit: int = 20) -> str:
    """Shorten a theme label for tight UI slots such as a metric tile.

    The survey's theme names are full phrases — "Working with the Leadership Teams
    (Heads of Department)" — and a metric tile truncates on rendered *width*, so even
    a 20-character name is clipped mid-word in a third-width column. Every theme
    shares one of a few leading relational phrases, and dropping it leaves the part
    that actually distinguishes the theme: "Leadership Teams", "Executive Teams",
    "Team". The parenthetical goes too, and only then does length-based clipping
    apply as a backstop.

    "My role at <company>" is special-cased: stripping its prefix would leave the
    company name, which reads as a different theme entirely.
    """
    s = str(name).strip()
    if s.lower().startswith("my role at"):
        return "My role"
    for prefix in _THEME_PREFIXES:
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    s = s.split(" (")[0].strip()
    return s if len(s) <= limit else s[: limit - 1].rstrip() + "\u2026"
