# Conclusions

This Optuna sweep completed (100 trials) and produced a tuned SL configuration.

Final metrics (from the run summary):

- Best trial val logloss: `0.47254733012305855`
- Final retrain val logloss: `0.47254733012305855`
- Final test logloss: `0.46713404881787735`

Best parameters:

- `lr = 0.0012940077533808388`
- `weight_decay = 1.6965892356085282e-08`
- `sl_alpha = 0.3581980808609457`
- `sl_beta = 0.575839137288588`
- `sl_center = 0.028450174282538376`

Notes:

- This run includes per-trial recomputation of the SL basis (by refitting preprocessing each trial), which makes it substantially slower than tuning only `lr`/`weight_decay`.

Comparison:

- B-spline best-so-far val logloss (separate run, stopped early): `0.47180640675774904`
- Current SL best val logloss gap: `+0.00074092336530951`
