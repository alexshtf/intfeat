# Conclusions

This run completed the `sl_optuna_u_exp_valley_400k_60t` study to `100` Optuna trials (continuing the existing SQLite DB).

## Final Result (Best Trial)

- Variant: `sl_integer_basis` (`u_exp_valley`)
- Best trial: `35`
- Best params:
  - `lr=0.0017712641824756783`
  - `weight_decay=1.5946227851910192e-06`
  - `sl_u0=0.12714125920232416`
  - `sl_left_slope=0.10599526233947935`
  - `sl_right_slope=3.508157381375444`
- Best/Final validation logloss: `0.4717266761299534`
- Best/Final test logloss: `0.46657057111844014`

## Comparison vs Splines

Spline baseline on the same `400k` contiguous split had validation logloss `~0.4718064`.

- SL `u_exp_valley`: `0.4717267`
- Delta: `-7.97e-05` (SL slightly better)

This is not the “large enough margin” improvement we’re looking for; treat this as **parity / tiny edge** rather than a decisive win.

## Artifacts

- Summary JSON: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to100/results.json`
- Study DB (shared): `journal/2026-02-10_sl_optuna_u_exp_valley_400k_60t/study.sqlite3`
- Full log: `journal/2026-02-11_sl_optuna_u_exp_valley_continue_400k_to100/run.log`
