"""Scientific evaluation suite: daily / monthly / JJAS-seasonal, spatial.

Submodules:
  aggregate       daily -> monthly / JJAS aggregation, climatology, tercile categories
  deterministic   MBE, MAE, PBIAS, RMSE, SD ratio, CV ratio
  distributions   Q-Q / ECDF / PDF data, Q10/Q50/Q90/Q95 bias, wet-day frequency bias
  spells          wet/dry spell length distributions
  skill           ACC, Spearman anomaly correlation, RMSESS vs climatology, interannual variability ratio
  probabilistic   CRPS/CRPSS, RPSS, Brier Score/BSS, ROC area/skill
  calibration     reliability diagrams, spread-skill ratio
  spatial         spatial pattern correlation, spatial RMSE, spatial correlation over time

All metrics compare a "model" DataArray (raw or corrected ECMWF, with a
`realization` dim for ensemble ones) against a "ref" DataArray (CHIRPS, no
realization dim). Deterministic metrics use the ensemble mean unless noted.
"""
