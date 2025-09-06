# Try CVXPY-based implementation for CDF smoothing (two variants):
#   (1) LS + Laplacian penalty on the CDF with monotonicity and endpoints.
#   (2) L1 on second differences (TV on pdf) with an L-infinity (KS) tube around ECDF.
#
# We'll run on a smaller grid if needed for speed, but attempt max_val=600 first.
# If cvxpy is unavailable, we'll raise a clear message.

import numpy as np
import matplotlib.pyplot as plt

# Attempt to import cvxpy
try:
    import cvxpy as cp

    has_cvxpy = True
except Exception as e:
    has_cvxpy = False
    err = str(e)

max_val = 1000  # use 600 bins to keep the QPs light
xs = np.arange(max_val)
ns = [100, 500, 5000]


# ---------- Synthetic distributions (reuse a subset for speed) ----------
def normalize(w):
    w = np.asarray(w, float)
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def bimodal_powerlaw(alpha=1.0, beta=1.0):
    ws_1 = (alpha + np.abs(xs - max_val / 3)) ** (-beta)
    ws_2 = (alpha + np.abs(xs - 2 * max_val / 3)) ** (-beta)
    return normalize((ws_1 + ws_2) / 2.0)


def trimodal_powerlaw(alpha=1.0, beta=1.2, centers=(0.2, 0.5, 0.8)):
    cs = [int(c * max_val) for c in centers]
    w = sum((alpha + np.abs(xs - c)) ** (-beta) for c in cs) / len(cs)
    return normalize(w)


def zipf_right(alpha=1.0, beta=1.1):
    return normalize((alpha + xs) ** (-beta))


def plateau_mid(baseline=1.0, plateau=5.0, left=0.35, right=0.65):
    L, R = int(left * max_val), int(right * max_val)
    w = np.full(max_val, baseline, float)
    w[L:R] = plateau
    return normalize(w)


strategies = [
    ("Bimodal power-law", bimodal_powerlaw()),
    ("Trimodal", trimodal_powerlaw()),
    ("Zipf right", zipf_right()),
    ("Mid plateau", plateau_mid()),
]

rng = np.random.default_rng(12345)


def sample_counts(p, n):
    return np.bincount(rng.choice(np.arange(max_val), size=n, p=p), minlength=max_val)


# ------------- Discrete operators -------------
D1 = np.eye(max_val) - np.eye(
    max_val, k=-1
)  # forward diff (y[i]-y[i-1]); D1[0] has 1 at 0
D1[0, :] = 0.0
D1[0, 0] = 1.0
D2 = D1 @ D1  # second difference


# helper to build ECDF vector on grid from counts
def ecdf_from_counts(counts):
    p = counts / max(1, counts.sum())
    return np.cumsum(p)


# ------------- Variant (1): LS + Laplacian penalty -------------
def fit_cdf_ls_lap(F_ecdf, lam=10.00):
    # Variables: y (cdf)
    y = cp.Variable(max_val)
    obj = cp.norm(y - F_ecdf, 1) / np.sqrt(max_val) + lam * cp.norm(D2 @ y)
    cons = [
        y[0] == 0.0,
        y[-1] == 1.0,
        cp.diff(y) >= 0,
    ]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve()
    print(prob.status)
    return y.value


# ------------- Variant (2): TV on pdf + KS tube -------------
def fit_cdf_tv_ks(F_ecdf, eps_ks=0.03, lam_tv=10.0):
    # Min ||D2 y||_1 subject to |y - F|_inf <= eps, monotone, endpoints
    y = cp.Variable(max_val)
    obj = cp.norm(F_ecdf - y, "inf") + lam_tv * cp.norm(D2 @ y)
    cons = [
        y[0] == 0.0,
        y[-1] == 1.0,
        cp.diff(y) >= 0,
    ]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve()
    return y.value


# ------------- Run and plot -------------
if not has_cvxpy:
    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.axis("off")
    ax.text(
        0.02,
        0.6,
        "CVXPY is not available in this environment.\n"
        "Please install cvxpy (and OSQP) to run the convex CDF smoothers.\n"
        f"Import error:\n{err}",
        fontsize=12,
    )
    plt.show()
else:
    # choose parameters (light tuning)
    lam_ls = 20.0  # Laplacian strength for LS
    eps_ks = 0.03  # KS tube half-width
    lam_tv = 2.0  # TV weight (objective is lam_tv * ||D2 y||_1)

    N = len(strategies)
    fig, axes = plt.subplots(
        len(ns), N, figsize=(3.7 * N, 8.8), sharex=True, sharey=False
    )

    for col, (name, p_true) in enumerate(strategies):
        F_true = np.cumsum(p_true)

        for row, n in enumerate(ns):
            counts = sample_counts(p_true, n)
            F_emp = ecdf_from_counts(counts)

            # Fit both variants
            y_ls = fit_cdf_ls_lap(F_emp, lam=lam_ls)
            y_tv = fit_cdf_tv_ks(F_emp, eps_ks=eps_ks, lam_tv=lam_tv)

            # PDFs (first differences) for sanity (not plotted)
            f_ls = np.diff(np.r_[0.0, y_ls])
            f_tv = np.diff(np.r_[0.0, y_tv])

            # KS metrics
            ks_emp = float(np.max(np.abs(F_true - F_emp)))
            ks_ls = float(np.max(np.abs(F_true - y_ls)))
            ks_tv = float(np.max(np.abs(F_true - y_tv)))

            ax = axes[row, col]
            # ax.plot(xs, F_true, label="Truth CDF")
            # ax.plot(xs, F_emp, label="Empirical CDF")
            # ax.plot(xs, y_ls, label="LS+Lap CDF")
            # ax.plot(xs, y_tv, label="TV+KS CDF")
            ax.plot(xs, p_true, label="Truth PDF")
            ax.plot(xs, f_ls, label="Least squares")
            ax.plot(xs, f_tv, label="TV + KS")

            ax.set_title(name if row == 0 else "")
            if col == 0:
                ax.set_ylabel(f"n = {n}")
            if row == len(ns) - 1:
                ax.set_xlabel("bin index")
            if col == N - 1 and row == 0:
                ax.legend(loc="lower right", fontsize=8, frameon=False)
            ax.text(
                0.02,
                0.06,
                f"KS(emp)={ks_emp:.3f}\nKS(LS)={ks_ls:.3f}\nKS(TV)={ks_tv:.3f}",
                transform=ax.transAxes,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.75),
            )

    plt.tight_layout()
    plt.show()
