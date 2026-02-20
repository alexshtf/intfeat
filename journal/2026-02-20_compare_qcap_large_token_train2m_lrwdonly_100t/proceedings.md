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

IN PROGRESS

