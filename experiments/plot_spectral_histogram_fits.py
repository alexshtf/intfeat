# Re-execute after state reset.

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import linalg
from scipy.stats import gaussian_kde

max_val = 1000
K_modes = 10
shaping_strength = 1
xs = np.arange(max_val)


def normalize(w):
    w = np.asarray(w, float)
    w = np.maximum(w, 0)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def weights_bimodal_powerlaw(alpha=1.0, beta=1.0):
    ws_1 = (alpha + np.abs(xs - max_val / 3)) ** (-beta)
    ws_2 = (alpha + np.abs(xs - 2 * max_val / 3)) ** (-beta)
    ws = (ws_1 + ws_2) / 2.0
    return normalize(ws)


def weights_trimodal_powerlaw(alpha=1.0, beta=1.2, centers=(0.2, 0.5, 0.8)):
    cs = [int(c * max_val) for c in centers]
    w = sum((alpha + np.abs(xs - c)) ** (-beta) for c in cs) / len(cs)
    return normalize(w)


def weights_zipf_right(alpha=1.0, beta=1.1):
    w = (alpha + xs) ** (-beta)
    return normalize(w)


def weights_plateau_mid(baseline=1.0, plateau=5.0, left=0.35, right=0.65):
    L, R = int(left * max_val), int(right * max_val)
    w = np.full(max_val, baseline, float)
    w[L:R] = plateau
    return normalize(w)


def weights_rippled_powerlaw(center=0.6, alpha=1.0, beta=1.2, amp=0.35, period=50):
    c = int(center * max_val)
    base = (alpha + np.abs(xs - c)) ** (-beta)
    ripple = 1.0 + amp * np.cos(2 * np.pi * xs / period)
    w = base * np.maximum(ripple, 0.05)
    return normalize(w)


def weights_spiky_mixture(
    spikes=(50, 500, 900), spike_mass=0.12, sigma=120.0, center=700
):
    gauss = np.exp(-0.5 * ((xs - center) / sigma) ** 2)
    w = (1.0 - spike_mass) * gauss
    for s in spikes:
        if 0 <= s < max_val:
            w[s] += spike_mass / len(spikes)
    return normalize(w)


def weights_u_shaped(alpha=1.0, beta=1.2):
    dist_to_edge = np.minimum(xs, max_val - 1 - xs)
    w = (alpha + dist_to_edge) ** (-beta)
    return normalize(w)


strategies = [
    ("Bimodal power-law (original)", weights_bimodal_powerlaw(alpha=1.0, beta=1.0)),
    ("Trimodal power-law", weights_trimodal_powerlaw()),
    ("Zipf-like (right-tail)", weights_zipf_right()),
    ("Mid plateau", weights_plateau_mid()),
    ("Rippled power-law", weights_rippled_powerlaw()),
    ("Spiky mixture", weights_spiky_mixture()),
    ("U-shaped heavy tails", weights_u_shaped()),
]


def laplacian_project_corrected(
    empirical_hist, num_coefs=K_modes, shaping_strength=shaping_strength
):
    base = 2 * np.ones(max_val)
    base[0] = base[-1] = 1.0
    diag = base - shaping_strength * empirical_hist
    off_diag = -np.ones(max_val - 1)
    eigvals, eigvecs = scipy.linalg.eigh_tridiagonal(
        diag, off_diag, select="i", select_range=(0, num_coefs)
    )
    coefficients = eigvecs.T @ empirical_hist
    apx_hist = eigvecs @ coefficients
    apx_hist = np.maximum(apx_hist, 0)
    s = apx_hist.sum()
    if s > 0:
        apx_hist = apx_hist / s
    return apx_hist


def kde_curve(samples, grid_x=xs):
    kde = gaussian_kde(samples.astype(float), bw_method="scott")
    return kde(grid_x)


rng = np.random.default_rng(22)


def sample_from_weights(w, n, rng):
    return rng.choice(np.arange(max_val), size=n, p=w)


N = len(strategies)
fig, axes = plt.subplots(4, N, figsize=(3.4 * N, 11.0), sharex=True)

for col, (name, w_true) in enumerate(strategies):
    ax = axes[0, col]
    ax.plot(xs, w_true, lw=1.5)
    ax.set_title(name, fontsize=11)
    if col == 0:
        ax.set_ylabel("True pmf")

    s100 = sample_from_weights(w_true, 100, rng)
    counts100 = np.bincount(s100, minlength=max_val)
    p100 = counts100 / counts100.sum()
    p100_hat = laplacian_project_corrected(p100, num_coefs=K_modes)
    ax = axes[1, col]
    ax.bar(xs, p100, width=1.0, color="#f6a04d", alpha=0.55, label="Empirical (100)")
    ax.plot(xs, p100_hat, color="orange", lw=1.2, label=f"Spectrum K={K_modes}")
    ax.plot(xs, kde_curve(s100), color="#1f77b4", lw=1.2, label="KDE")
    if col == 0:
        ax.set_ylabel("n = 100")
    if col == N - 1:
        ax.legend(loc="upper right", fontsize=8, frameon=False)

    s500 = sample_from_weights(w_true, 1000, rng)
    counts500 = np.bincount(s500, minlength=max_val)
    p500 = counts500 / counts500.sum()
    p500_hat = laplacian_project_corrected(p500, num_coefs=K_modes)
    ax = axes[2, col]
    ax.bar(xs, p500, width=1.0, color="#f6a04d", alpha=0.55, label="Empirical (500)")
    ax.plot(xs, p500_hat, color="orange", lw=1.2, label=f"Spectrum K={K_modes}")
    ax.plot(xs, kde_curve(s500), color="#1f77b4", lw=1.2, label="KDE")
    if col == 0:
        ax.set_ylabel("n = 500")

    s5000 = sample_from_weights(w_true, 20000, rng)
    counts5000 = np.bincount(s5000, minlength=max_val)
    p5000 = counts5000 / counts5000.sum()
    p5000_hat = laplacian_project_corrected(p5000, num_coefs=K_modes)
    ax = axes[3, col]
    ax.bar(xs, p5000, width=1.0, color="#f6a04d", alpha=0.55, label="Empirical (5000)")
    ax.plot(xs, p5000_hat, color="orange", lw=1.2, label=f"Spectrum K={K_modes}")
    ax.plot(xs, kde_curve(s5000), color="#1f77b4", lw=1.2, label="KDE")
    if col == 0:
        ax.set_ylabel("n = 5000")

for ax in axes[-1, :]:
    ax.set_xlabel("bin index")

plt.tight_layout()
plt.show()
