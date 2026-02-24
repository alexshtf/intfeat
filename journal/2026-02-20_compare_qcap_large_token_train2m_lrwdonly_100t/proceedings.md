# Proceedings

## Code References

- Runner: `experiments/criteo_fwfm/optuna_head_contiguous_resumable.py`
- Configs:
  - All-spline: `experiments/criteo_fwfm/config/model_bspline_quantile_large_token.yaml`
  - Hybrid (fixed u-right barrier): `experiments/criteo_fwfm/config/model_hybrid_bspline_sl_i5_quantile_large_token_u_right_inverse_square_fixed.yaml`
  - Hybrid (fixed inverse-square): `experiments/criteo_fwfm/config/model_hybrid_bspline_sl_i5_quantile_large_token_inverse_square_fixed.yaml`

## Commands (Reproducible)

This experiment is intentionally launched in detached mode so it survives shell/Codex shutdown.

Start:

```bash
cd /home/alex/git/intfeat
setsid -f /bin/bash -lc 'bash journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/run_detached.sh'
```

Monitoring:

- Wrapper PID: `journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/pid.txt`
- Per-run logs + sqlite + PID files under:
  - `journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/runs/bspline/`
  - `journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/runs/hybrid_u_right/`
  - `journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/runs/hybrid_inverse_square/`

## Status

Launched: 2026-02-20T19:38:20+02:00

- Wrapper PID: 2065683 (see `journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/pid.txt`)

### Stage 1: All-Spline (Completed)

- Run dir: `journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/runs/bspline/`
- Optuna: 100 trials (tune `lr`, `weight_decay`)
- Best trial val logloss: `0.4513791662` (trial 58)
- Final retrain: val `0.4513791662`, test `0.4555181726`
- Best params:
  - `lr=0.0008986722195700843`
  - `weight_decay=7.111241352134466e-07`

### Stage 2: Hybrid I5 SL (Fixed u-right barrier) (In Progress)

- Run dir: `journal/2026-02-20_compare_qcap_large_token_train2m_lrwdonly_100t/runs/hybrid_u_right/`
- Python PID: 2199108 (see `.../runs/hybrid_u_right/pid.txt`)
- Log: `.../runs/hybrid_u_right/run.log`
