Here’s a compact, self-contained tutorial you can hand to advanced undergrads. It introduces the problem, gives a clear algorithm, shows how to implement it in a few lines, and explains why it works.

---

# 1) The problem: “smooth” histograms for bounded discrete data

We observe samples $x_1,\ldots,x_n \in \{0,1,\dots,m-1\}$ and want a probability mass function (pmf) $q\in\Delta^{m-1}$ that:

* looks **smooth** (no spiky bin-to-bin noise),
* **preserves modes/peaks** suggested by the data,
* is fast to compute and easy to interpret.

Why do we care?

* **Visualization:** smooth plots that capture structure without distracting noise.
* **Quantiles/CDFs:** smoother pmfs yield stable CDFs, hence better quantile estimates.
* **Downstream stats:** entropy, divergences, calibration, anomaly scores, etc., are less volatile when the pmf isn’t jagged.
* **Small-n regime:** with many bins but few samples per bin, the empirical histogram is high-variance; smoothing pays off.

---

# 2) Spectral smoothing with a data-dependent operator

## 2.1 Idea in one line

Build a **data-dependent low-pass basis** from the first $K$ eigenvectors of a simple tridiagonal matrix and **project the empirical histogram onto that basis**.

## 2.2 The operator

Let $p$ be the empirical pmf (counts normalized to sum 1). Define the 1-D path-graph Laplacian $L\in\mathbb{R}^{m\times m}$ with **Neumann boundaries**:

* diagonal: $[1,2,2,\dots,2,1]$
* off-diagonals: $-1$ next to the diagonal.

Form the **data-dependent** operator

$$
H \;=\; L \;-\; \lambda\,\mathrm{diag}(p)\qquad (\lambda>0,\ \text{often } \lambda=1).
$$

* $L$ penalizes **wiggles** (adjacent differences); small-eigenvalue eigenvectors are smooth.
* $-\mathrm{diag}(p)$ **rewards** putting mass where $p$ is large; small-eigenvalue eigenvectors get **attracted to peaks**.

Compute the first $K$ eigenvectors $U=[u_1,\ldots,u_K]$ of $H$ (columns orthonormal).
Smooth by **orthogonal projection**:

$$
\hat q \;=\; U\,U^\top p .
$$

This is $O(Km)$ to build (using tridiagonal eigensolvers) and $O(Km)$ to project—very fast.

## 2.3 Minimal implementation (NumPy/SciPy)

```python
import numpy as np
from scipy.linalg import eigh_tridiagonal

def laplacian_neumann(m: int):
    diag = 2.0 * np.ones(m)
    diag[0] = diag[-1] = 1.0
    off  = -np.ones(m-1)
    return diag, off  # tridiagonal representation

def spectral_basis(p, K=10, lam=1.0):
    p = np.asarray(p, float)
    m  = p.size
    diagL, off = laplacian_neumann(m)
    diagH = diagL - lam * p
    w, U = eigh_tridiagonal(diagH, off, select='i', select_range=(0, K-1))
    return U  # m x K, columns orthonormal

def spectral_project(p, K=10, lam=1.0):
    U = spectral_basis(p, K, lam)
    return U @ (U.T @ p)  # Euclidean projection onto span(U)
```

## 2.4 Why it works (intuition only)

* **Smoothness:** The Laplacian $L$ assigns high “energy” to oscillatory vectors. Eigenvectors with small eigenvalues vary slowly across bins—like low-frequency sinusoids on a line.
* **Peak attraction:** Subtracting $\lambda\mathrm{diag}(p)$ lowers the energy where $p$ is big. That **pulls** the low-energy eigenvectors **toward the data’s peaks**, so the low-frequency basis is automatically **shaped by the histogram**.
* **Projection:** Expressing $p$ in this basis and dropping high-index modes removes small-scale noise while keeping the broad features (peaks/shoulders) the data support.

You can think of it as **“smoothness − reward” minimization**: small-eigenvalue eigenvectors strike the best trade-off between being smooth and sitting where the data are.

---

# 3) Getting back to a valid pmf: three normalization options

The projected vector $\hat q = U U^\top p$ can be slightly negative and need not sum to 1. Here are three simple, principled normalizations, each with a one-liner implementation.

## (A) Naïve clip-and-renormalize

Set negatives to 0 and divide by the sum:

```python
def clip_and_renorm(q):
    q = np.maximum(q, 0.0)
    s = q.sum()
    return q / s if s > 0 else np.full_like(q, 1.0/q.size)
```

**Pros:** one line. **Cons:** ignores geometry; can over-sharpen when many small negatives are clipped.

## (B) Bregman (mirror) projections

* **KL / “I-projection” (entropy mirror):**
  $\phi(z)=\sum z_i\log z_i$.
  Minimizing $B_\phi(p,\hat q)$ over the simplex yields:

  $$
  \tilde q_i = \frac{\hat q_i}{\sum_{j:\,\hat q_j>0}\hat q_j}\ \text{for }\hat q_i>0,\quad \tilde q_i=0\ \text{otherwise.}
  $$

  In words: **normalize the positive part**.

* **Burg / Itakura–Saito mirror:**
  $\phi(z)=\sum -\ln z_i$.
  Closed form

  $$
  \tilde q_i \;=\; \frac{r_i}{1+\lambda r_i},\quad r=\hat q,\quad
  \text{with }\sum_i \frac{r_i}{1+\lambda r_i}=1
  $$

  (solve for $\lambda$ by a scalar root-find; robust and fast).

```python
from scipy.optimize import brentq

def kl_simplex_projection(r):
    mask = r > 0
    out = np.zeros_like(r)
    s = r[mask].sum()
    out[mask] = r[mask] / (s if s>0 else 1.0)
    return out

def burg_simplex_projection(r, eps=1e-12):
    r = np.maximum(r, eps); rmax=r.max()
    def g(lam): return (r/(1+lam*r)).sum() - 1
    g0 = g(0.0)
    if abs(g0) < 1e-16:
        lam = 0.0
    elif g0 > 0:
        lo, hi = 0.0, 1.0
        while g(hi) > 0: hi *= 2
        lam = brentq(g, lo, hi)
    else:
        lam = brentq(g, -1.0/rmax + 1e-16, 0.0)
    q = r/(1+lam*r); q = np.maximum(q, 0.0); q /= q.sum()
    return q
```

**When to use which?**
KL favors **mass preservation** where $\hat q$ is already positive; Burg penalizes tiny probabilities more strongly and tends to **spread** mass a bit, which can help with heavy tails.

## (C) Euclidean projection onto the simplex (no mirror)

There’s a closed-form $O(m\log m)$ projection onto $\Delta$ (Duchi et al.). It’s geometry-agnostic but often a good default.

```python
def euclid_simplex_projection(v):
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    rho = np.nonzero(u - cssv / (np.arange(len(u))+1) > 0)[0]
    theta = 0.0 if len(rho)==0 else cssv[rho[-1]]/(rho[-1]+1.0)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / (s if s>0 else 1.0)
```

> **Advanced (optional):** You can project **onto the intersection** $\Delta \cap \mathrm{span}(U)$ (rather than onto $\Delta$ only) by alternating projections (Dykstra). This keeps you in the chosen spectral subspace *and* on the simplex.

---

# 4) Worked examples (what to run)

Below is a small driver that (i) builds the basis from $H=L-\mathrm{diag}(p)$, (ii) projects, (iii) normalizes by your favorite method, and (iv) plots. You can plug in any of the three normalizers.

```python
import matplotlib.pyplot as plt

def spectral_smooth(p_emp, K=10, lam=1.0, normalizer="euclid"):
    q_hat = spectral_project(p_emp, K=K, lam=lam)
    if normalizer == "clip":
        return clip_and_renorm(q_hat)
    elif normalizer == "kl":
        return kl_simplex_projection(q_hat)
    elif normalizer == "burg":
        return burg_simplex_projection(q_hat)
    else:
        return euclid_simplex_projection(q_hat)

# Demo on synthetic data (choose your generator)
def sample_counts(p_true, n, rng):
    m = p_true.size
    xs = rng.choice(np.arange(m), size=n, p=p_true)
    return np.bincount(xs, minlength=m)

# Example usage (single panel)
rng = np.random.default_rng(0)
m = 500
# true pmf with two heavy-tailed peaks
xs = np.arange(m)
p1 = (1+np.abs(xs - m//3))**(-1.0)
p2 = (1+np.abs(xs - 2*m//3))**(-1.0)
p_true = (p1+p2)/np.sum(p1+p2)

n = 500
counts = sample_counts(p_true, n, rng)
p_emp = counts / counts.sum()
q_smooth = spectral_smooth(p_emp, K=10, lam=1.0, normalizer="burg")

plt.plot(p_true, label="true")
plt.bar(xs, p_emp, alpha=0.4, label="empirical")
plt.plot(q_smooth, label="spectral (K=10, Burg)")
plt.legend(); plt.show()
```

> In our experiments (you’ve seen the grids), this method **beats the raw empirical histogram for small/medium $n$** and gracefully tracks the truth as $n$ grows (when you increase $K$ or select modes adaptively).

**Choosing $K$ and $\lambda$:**

* Keep $K$ modest (e.g., 5–15) for very sparse histograms; increase with $n$.
* Simple adaptive rule: keep eigenmodes with $|u_j^\top p|$ above a **universal threshold**
  $|u_j^\top p| \ge \hat\sigma_j \sqrt{2\log K_{\max}}$ with
  $\hat\sigma_j^2 \approx \frac{1}{n}\big(\sum_i u_{ij}^2 p_i - (u_j^\top p)^2\big)$.
* $\lambda=1$ works well; you can tune $\lambda$ on a validation split if desired.

---

# 5) Bonus: adding a rank-one data term $L - \alpha\,\mathrm{diag}(p) - \beta\,p p^\top$

## Intuition

* $-\alpha\,\mathrm{diag}(p)$: **local reward**—attracts basis functions to bins where data mass is large (peak attraction).
* $-\beta\,p p^\top$: **global alignment**—encourages the basis to correlate with the overall pattern of $p$ (not just local spikes). This rank-one term can help when you want the first mode to align strongly with the coarse shape of the histogram.

## Operator and algorithm

Form

$$
H_{\alpha,\beta} \;=\; L - \alpha\,\mathrm{diag}(p) - \beta\,p p^\top .
$$

Compute the first $K$ eigenvectors $U$ of $H_{\alpha,\beta}$ (again tridiagonal + rank-1, still cheap in practice for moderate $m$), then project $p$ and normalize exactly as before.

```python
def spectral_basis_rank1(p, K=10, alpha=1.0, beta=0.2):
    m = p.size
    diagL, off = laplacian_neumann(m)
    diagH = diagL - alpha * p
    # eigenpairs of tridiagonal + rank-one: for simplicity, use dense multiply via matvec
    # (for teaching code; for large m you’d use a Lanczos with a matvec)
    L = np.diag(diagL) + np.diag(off,1) + np.diag(off,-1)
    H = L - alpha*np.diag(p) - beta*np.outer(p, p)
    w, V = np.linalg.eigh(H)
    idx = np.argsort(w)[:K]
    return V[:, idx]

def spectral_project_rank1(p, K=10, alpha=1.0, beta=0.2, normalizer="euclid"):
    U = spectral_basis_rank1(p, K, alpha, beta)
    q_hat = U @ (U.T @ p)
    if normalizer == "burg":   return burg_simplex_projection(q_hat)
    if normalizer == "kl":     return kl_simplex_projection(q_hat)
    if normalizer == "clip":   return clip_and_renorm(q_hat)
    return euclid_simplex_projection(q_hat)
```

## Where $pp^\top$ helps

When the true density has a **dominant global trend** (e.g., heavy right tail) plus **local structure**, $-\beta pp^\top$ can make the **first mode** closely follow that coarse trend, so fewer modes are needed to reconstruct the shape. If $\beta$ is too large, you can over-bias toward $p$ and lose detail—treat it as a small, optional nudge.

---

# 6) Practical guidance

* **When this shines:** many bins, few samples per bin (sparse histograms), or heavy-tailed pmfs.
* **As $n$ grows:** the empirical histogram becomes competitive; let $K$ **increase with $n$** (or use thresholded mode selection) so bias doesn’t dominate.
* **Boundaries matter:** use **Neumann ends** $[1,2,\ldots,2,1]$ in $L$ to avoid artificial damping at the edges.
* **Diagnostics:** compare CDFs (KS), TV distance, and held-out log-likelihood to quantify improvements.
* **Complexity:** tridiagonal eigensolver for first $K$ modes is roughly $O(Km)$; projection is $O(Km)$; normalizations are $O(m\log m)$ (Euclidean) or $O(m)$ (KL/Burg with a 1-D root).

---

This pipeline—**compute a data-dependent low-pass basis, project once, normalize**—is just a few lines of linear algebra, yet it captures the essential structure of the histogram while suppressing sampling noise. It’s fast, interpretable, and easy to drop into visualization and inference workflows.
