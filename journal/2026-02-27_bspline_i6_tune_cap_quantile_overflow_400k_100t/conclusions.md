# Conclusions

## Result

This "fair baseline" variant (tune spline cap/overflow for `I6` only) essentially matches the
hybrid `I6=SL` result, and is marginally better on validation.

- Tuned `I6` spline (this experiment):
  - Best/final val logloss: `0.4721077319`
  - Final test logloss: `0.4673236534`
  - Best params:
    - `I6.cap_quantile`: `0.9910423378`
    - `I6.positive_overflow`: `clip_to_cap`
    - `lr`: `1.8144063409e-3`
    - `weight_decay`: `1.1298766673e-7`

## Comparison

Reference runs (same 400k/400k/400k split, same global settings, except for what is tuned):

- All-spline baseline (global settings; no per-column I6 tuning):
  - Val: `0.4731016777`
  - Test: `0.4678396553`
- Hybrid `I6=SL` (tuned SL params for I6; rest splines):
  - Val: `0.4721500809`
  - Test: `0.4673242588`

Differences:

- Tuned `I6` spline vs hybrid `I6=SL`:
  - Val improves by `~4.23e-05` (better)
  - Test improves by `~6.05e-07` (effectively identical)
- Tuned `I6` spline vs all-spline baseline:
  - Val improves by `~9.94e-04`
  - Test improves by `~5.16e-04`

## Takeaway

The previous improvement from switching `I6` to SL does not survive a "fairer" spline baseline.
Once splines are allowed a comparable per-column knob for `I6` (cap quantile + overflow handling),
the spline model reaches the same quality as `I6=SL`.

This suggests that, for this setup, most of the gain came from better `I6` tail handling (and/or
optimizer settings), not from the SL basis being intrinsically better than B-splines.
