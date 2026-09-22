# PYTHON can be overridden, e.g.  make pipeline PYTHON=python3.12
PYTHON ?= python3

.PHONY: help install app demo-data test pipeline all clean

help:
	@echo "make install    install the dependencies and the engagement package"
	@echo "make app        open the stakeholder report in your browser (Streamlit)"
	@echo "make demo-data  write the demo survey workbook to data/raw/ (already committed)"
	@echo "make test       run the pytest suite (6 tests on the critical transforms)"
	@echo "make pipeline   execute notebooks 01-06 in order + export HTML + write manifest"
	@echo "make all        test, then pipeline"
	@echo "make clean      delete generated outputs (figures, tables, processed data)"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

# Opens the stakeholder report at http://localhost:8501 (stop with Ctrl+C).
# The app only reads outputs/, so the pipeline must have run once; the
# committed outputs already satisfy that on a fresh clone.
app:
	@test -f outputs/tables/tab03_theme_ranking.csv || { echo "No pipeline outputs yet. Run: make pipeline"; exit 1; }
	$(PYTHON) -m streamlit run app/Home.py

# --force so re-running is not blocked by the committed copy. It refuses to
# overwrite without it, so a real client workbook placed here is never lost.
demo-data:
	$(PYTHON) scripts/make_demo_workbook.py --force

# `python -m pytest` rather than the bare `pytest` console script: it guarantees
# the tests run under the same interpreter the package was installed into.
test:
	$(PYTHON) -m pytest -v

pipeline:
	$(PYTHON) scripts/run_pipeline.py

all: test pipeline

clean:
	rm -rf outputs/figures/* outputs/tables/* outputs/notebooks_html/* data/processed/*
	touch outputs/figures/.gitkeep outputs/tables/.gitkeep data/processed/.gitkeep
