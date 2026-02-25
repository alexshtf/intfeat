# Conclusions

Sorted by XGBoost **gain** importance (normalized), trained on rows `[0, 400000)`:

Top 10 overall:

- `I6` (0.0806)
- `C12` (0.0728)
- `C26` (0.0658)
- `I11` (0.0603)
- `C21` (0.0599)
- `C4` (0.0595)
- `C7` (0.0557)
- `C10` (0.0539)
- `C24` (0.0532)
- `C16` (0.0517)

Integer columns only (`I*`), ranked by gain:

- `I6` (0.0806)
- `I11` (0.0603)
- `I13` (0.0240)
- `I7` (0.0213)
- `I1` (0.0094)
- `I3` (0.0083)
- `I5` (0.0066)
- `I8` (0.0061)
- `I9` (0.0043)
- `I4` (0.0034)
- `I12` (0.0015)
- `I10` (0.0015)
- `I2` (0.0015)

Artifacts:

- Raw report: `journal/2026-02-25_xgb_feature_importance_400k/importance.json`
- Runner script: `experiments/criteo_fwfm/analysis/xgb_feature_importance.py`

