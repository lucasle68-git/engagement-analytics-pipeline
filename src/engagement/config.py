"""Configuration loader, a single source of truth for every parameter.

Why: no magic numbers in analysis code. Markers (or the client) can re-run the
entire pipeline under different assumptions by editing config/config.yaml only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repository root = two levels above this file (src/engagement/config.py)
ROOT = Path(__file__).resolve().parents[2]

_REQUIRED_TOP_KEYS = {"project", "paths", "data", "quality", "phase1", "synthetic", "models"}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate config.yaml.

    Parameters
    ----------
    path : optional explicit path; defaults to <repo>/config/config.yaml.

    How it works
    ------------
    1. Locate the file: the caller's path if given, otherwise
       `<repo>/config/config.yaml` (ROOT is derived from this file's own
       location, so it works from any working directory).
    2. Stop immediately if it is missing.
    3. Read the YAML into a nested Python dictionary.
    4. Check all seven required sections are present, using set subtraction including
       a typo'd or deleted section fails here, at the gate, rather than
       surfacing as a confusing error mid-analysis.
    5. Turn every `*_dir` / `*_workbook` entry into an absolute path.

    Every other module receives this dictionary and trusts it: configuration is
    read and checked in exactly one place.

    Returns
    -------
    dict with all configuration sections; path values resolved against repo root.

    Raises
    ------
    FileNotFoundError, ValueError on missing file / missing required sections.
    """
    cfg_path = Path(path) if path else ROOT / "config" / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    missing = _REQUIRED_TOP_KEYS - cfg.keys()
    if missing:
        raise ValueError(f"config.yaml missing sections: {sorted(missing)}")

    # Resolve all paths relative to the repository root so the pipeline can be
    # launched from any working directory.
    for key, value in cfg["paths"].items():
        if key.endswith(("_dir", "_workbook")):
            cfg["paths"][key] = str((ROOT / value).resolve())
    return cfg


def get_seed(cfg: dict[str, Any]) -> int:
    """Return the global random seed (used by numpy, sklearn, generators)."""
    return int(cfg["project"]["seed"])
