"""AccessFintech employee-engagement analytics pipeline (MGT5496P).

Modules
-------
config     : loads/validates config.yaml — the single source of truth
io         : workbook loading, structural validation, tidy reshaping, exports
quality    : delta reconstruction, scale harmonisation, structure checks (ILO1)
phase1     : diagnostic analysis of the real 2024 aggregated data (RO1, RO3a)
synthetic  : calibrated synthetic-data generator + validators (Phase 2 input)
models     : driver analysis, risk models, evaluation, SHAP (RO2, RO3b)
personas   : segmentation / clustering of the synthetic workforce
viz        : styled, captioned figures — auto-stamps Real vs SYNTHETIC badges
"""

__version__ = "0.1.0"
