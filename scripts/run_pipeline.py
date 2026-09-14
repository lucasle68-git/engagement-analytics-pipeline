"""One command to reproduce the whole analysis: executes notebooks 01-06 in order.

Usage
-----
    make pipeline                          # execute all six, then export HTML
    python scripts/run_pipeline.py         # identical
    python scripts/run_pipeline.py --no-html
    python scripts/run_pipeline.py --html-only   # re-export HTML, execute nothing
    python scripts/run_pipeline.py --only 04 05  # execute a subset, in order

What it does
------------
1. Pre-flight: imports the package and loads config, then checks the raw workbook
   is where config says it is. Failing here costs a second; failing halfway
   through notebook 04 costs several minutes.
2. Executes the six notebooks IN ORDER, in place, so each is saved with its
   outputs. Order is not cosmetic: 04 writes the synthetic panel that 05 and 06
   read from disk.
3. Exports each notebook to outputs/notebooks_html/ so the analysis can be read
   in a browser without Jupyter.
4. Writes outputs/run_manifest.json including timestamp, seed, package versions,
   per-notebook runtime, and how many figures/tables now exist. This is the
   provenance record; it is regenerated on every run so it cannot go stale.

Halts at the first notebook that raises, printing the error from the failing
cell. A half-executed pipeline leaves outputs from two different runs side by
side, which is worse than stopping.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NOTEBOOKS = [
    "01_data_loading_quality.ipynb",
    "02_phase1_diagnostic.ipynb",
    "03_phase1_multivariate.ipynb",
    "04_synthetic_generation_validation.ipynb",
    "05_phase2_modelling.ipynb",
    "06_phase3_synthesis.ipynb",
]

HTML_DIR = ROOT / "outputs" / "notebooks_html"
CELL_TIMEOUT = 1800  # seconds; notebook 05 is the slowest at ~1-2 min

PACKAGES = ["pandas", "numpy", "scipy", "scikit-learn", "matplotlib",
            "seaborn", "statsmodels", "shap"]


# --------------------------------------------------------------------- helpers
def _nbconvert(args: list[str]) -> subprocess.CompletedProcess:
    """Run nbconvert, tolerating either invocation style.

    `python -m nbconvert` works wherever nbconvert is installed; the `jupyter
    nbconvert` entry point is the fallback for environments where only the
    console script is on PATH.
    """
    proc = subprocess.run([sys.executable, "-m", "nbconvert", *args],
                          cwd=ROOT, capture_output=True, text=True)
    if proc.returncode != 0 and "No module named" in (proc.stderr or ""):
        proc = subprocess.run(["jupyter", "nbconvert", *args],
                              cwd=ROOT, capture_output=True, text=True)
    return proc


def _tail(text: str, n: int = 25) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def preflight() -> dict | None:
    """Check the things that make the whole run fail, before spending minutes."""
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from engagement.config import load_config
    except ImportError as exc:
        print(f"  ✗ cannot import the engagement package ({exc}).\n"
              "    Run:  pip install -r requirements.txt && pip install -e .")
        return None

    cfg = load_config()
    workbook = Path(cfg["paths"]["raw_workbook"])
    if not workbook.exists():
        print(f"  ✗ survey workbook not found at {workbook}\n"
              "    The client's file is not distributed with this repository.\n"
              "    Run `make demo-data` (or python scripts/make_demo_workbook.py) to write the\n"
              "    demo stand-in, then re-run this command.")
        return None

    missing = [nb for nb in NOTEBOOKS if not (ROOT / "notebooks" / nb).exists()]
    if missing:
        print(f"  ✗ missing notebooks: {missing}")
        return None

    print(f"  ✓ package imports · workbook found · {len(NOTEBOOKS)} notebooks present")
    print(f"  ✓ seed = {cfg['project']['seed']} (config/config.yaml)")
    return cfg


def execute(notebooks: list[str]) -> dict[str, float] | None:
    """Execute each notebook in place; return {notebook: seconds} or None on failure."""
    timings: dict[str, float] = {}
    for i, nb in enumerate(notebooks, start=1):
        print(f"[{i}/{len(notebooks)}] executing {nb} ...", flush=True)
        started = time.time()
        proc = _nbconvert([
            "--to", "notebook", "--execute", "--inplace",
            f"--ExecutePreprocessor.timeout={CELL_TIMEOUT}",
            f"notebooks/{nb}",
        ])
        elapsed = time.time() - started
        if proc.returncode != 0:
            print(f"\n  ✗ {nb} failed after {elapsed:.0f}s. Error from the failing cell:\n")
            print(_tail(proc.stderr))
            print("\n  Pipeline halted — earlier notebooks have been saved with their "
                  "outputs, later ones are untouched.")
            return None
        timings[nb] = round(elapsed, 1)
        print(f"        done in {elapsed:.0f}s")
    return timings


def export_html(notebooks: list[str]) -> int:
    """Export executed notebooks to outputs/notebooks_html/ (no re-execution)."""
    HTML_DIR.mkdir(parents=True, exist_ok=True)
    written = 0
    for nb in notebooks:
        proc = _nbconvert(["--to", "html", "--output-dir", str(HTML_DIR),
                           f"notebooks/{nb}"])
        if proc.returncode != 0:
            print(f"  ! HTML export failed for {nb} (analysis itself is unaffected)")
            print(_tail(proc.stderr, 10))
        else:
            written += 1
    print(f"  ✓ {written}/{len(notebooks)} notebooks exported to outputs/notebooks_html/")
    return written


def write_manifest(cfg: dict, timings: dict[str, float]) -> Path:
    """Provenance record, regenerated every run so it cannot describe an old state."""
    packages = {}
    for pkg in PACKAGES:
        try:
            packages[pkg] = version(pkg)
        except PackageNotFoundError:
            packages[pkg] = "not installed"

    figures = sorted(p.name for p in (ROOT / "outputs" / "figures").glob("*.png"))
    tables = sorted(p.name for p in (ROOT / "outputs" / "tables").glob("*.csv"))

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": cfg["project"]["seed"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "notebooks_executed": timings or "none (--html-only)",
        "outputs": {"figures": len(figures), "tables": len(tables),
                    "figure_files": figures, "table_files": tables},
    }
    path = ROOT / "outputs" / "run_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path


# ------------------------------------------------------------------------ main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-html", action="store_true",
                        help="execute notebooks but skip the HTML export")
    parser.add_argument("--html-only", action="store_true",
                        help="re-export HTML from already-executed notebooks")
    parser.add_argument("--only", nargs="+", metavar="NN",
                        help="run a subset by number, e.g. --only 04 05")
    args = parser.parse_args()

    selected = NOTEBOOKS
    if args.only:
        wanted = {n.zfill(2) for n in args.only}
        selected = [nb for nb in NOTEBOOKS if nb[:2] in wanted]
        if not selected:
            print(f"  ✗ no notebooks match {sorted(wanted)}")
            return 1

    print("Pre-flight checks ...")
    cfg = preflight()
    if cfg is None:
        return 1

    timings: dict[str, float] = {}
    if not args.html_only:
        print(f"\nExecuting {len(selected)} notebook(s) in order "
              f"(04 must precede 05 and 06) ...")
        result = execute(selected)
        if result is None:
            return 1
        timings = result

    if not args.no_html:
        print("\nExporting HTML ...")
        export_html(selected)

    path = write_manifest(cfg, timings)
    total = sum(timings.values()) if timings else 0.0
    print(f"\n✓ Done{f' in {total:.0f}s' if total else ''}. "
          f"Figures → outputs/figures/ · tables → outputs/tables/ · "
          f"HTML → outputs/notebooks_html/")
    print(f"  Provenance: {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())