# Objective

Run a **hybrid integer encoding** experiment on Criteo (contiguous 400k/400k/400k split):

- All integer columns except `I11`: **B-splines** (same setup as the best 400k b-spline benchmark: `knots=10`).
- `I11`: **discrete Sturm-Liouville (SL)** basis with **its own tuned conductance + tuned potential**.

We tune (Optuna / TPE):

- Optimizer: `lr`, `weight_decay`
- `I11` conductance (u-space "valley" family): `u0`, `left_slope`, `right_slope`
- `I11` potential family `u_right_inverse_square`: `kappa`, `eps` (this is the *V* before multiplying by `w`)

Goal: test whether targeted SL modeling for the most interaction-heavy numeric feature (`I11`) can improve over the all-b-spline model on the same split.

