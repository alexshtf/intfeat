# Discrete Sturm–Liouville Bases for Heavy-Tailed Integer Features

*A practical, slow-paced tutorial with math, intuition, and code*

------

## 0) Motivation (what we want)

For integer features $x\in\{0,1,2,\dots\}$ that are **heavy-tailed**, we’d like an **orthonormal basis** $\{\phi_k\}$ that:

-   is orthonormal in an inner product that **matches the data distribution** (so errors where data are common “count more”), and
-   has a **controllable inductive bias** that tells low-index modes *where* to spend their curvature (e.g., high resolution near the origin, then gradually farther out).

We’ll construct such a basis by solving a **generalized eigenvalue problem** on the integer line segment $\{0,\dots,N\}$. Two knobs appear, cleanly **decoupled**:

-   a **weight** $w_x>0$ (from your data), defining the inner product $\langle f,g\rangle_w=\sum_x f(x)g(x)w_x$;
-   a **conductance** sequence $c_x>0$ (design choice) that controls *where* curvature is cheap or expensive.

------

## 1) The discrete operator $L$ (difference equation)

Let $\phi:\{0,\dots,N\}\to\mathbb{R}$. Choose:

-   **node weights** $w_x>0$ for all $x$,
-   **edge conductances** $c_x>0$ for edges $(x,x+1)$, with the conventions $c_{-1}=c_N=0$ (reflecting ends).

Define the linear operator

$(L\phi)(x)\;=\;\frac{1}{w_x}\Big[c_x\big(\phi(x)-\phi(x{+}1)\big)+c_{x-1}\big(\phi(x)-\phi(x{-}1)\big)\Big].$

This is the **weighted divergence of a discrete gradient**:

$L \;=\; W^{-1}\,\nabla^{\!*}\!\big(C\,\nabla\big),$

with $W=\mathrm{diag}(w_x)$, $C=\mathrm{diag}(c_x)$, $(\nabla\phi)_x=\phi(x{+}1)-\phi(x)$, and $\nabla^{\!*}$ the negative divergence.

The **eigenproblem** is

$L\phi_k=\lambda_k\,\phi_k,\qquad  \langle \phi_i,\phi_j\rangle_w=\delta_{ij}.$

**Interpretation.** The left side is a **weighted second-difference** (flux in minus flux out); the right side says that, pointwise, this curvature is proportional to the function itself, with proportionality modulated by $w_x$.

------

## 2) Matrix form and two equivalent eigenproblems

Collect $\phi$ as a vector $\boldsymbol\phi\in\mathbb{R}^{N+1}$. Define the **Laplacian-like** tridiagonal

$M= \begin{bmatrix} c_0     & -c_0   &        &        &  \\ -c_0    & c_0+c_1& -c_1   &        &  \\        & -c_1   & c_1+c_2& \ddots &  \\        &        & \ddots & \ddots & -c_{N-1}\\        &        &        & -c_{N-1}& c_{N-1} \end{bmatrix}, \quad W=\mathrm{diag}(w_0,\dots,w_N).$

Then $L=W^{-1}M$ and we solve the **symmetric-definite** generalized problem

$M\boldsymbol\phi=\lambda\,W\boldsymbol\phi  \quad\Longleftrightarrow\quad \boldsymbol\phi_i^\top W \boldsymbol\phi_j=\delta_{ij}.$

Equivalently, with $D=W^{1/2}$ and $\mathbf u=D\boldsymbol\phi$,

$S\,\mathbf u=\lambda\,\mathbf u,\qquad  S:=D^{-1} M D^{-1}=W^{-1/2} M W^{-1/2}.$

This is an ordinary **symmetric tridiagonal** eigenproblem. After solving for $\mathbf u$ (Euclidean-orthonormal), map back via $\boldsymbol\phi=W^{-1/2}\mathbf u$ to get $\ell^2(w)$-orthonormal eigenfunctions.

**Scale vs shape.** Multiplying all $c_x$ by a constant scales eigenvalues but leaves eigenvectors unchanged. Only the **shape** of $c_x$ matters for where oscillations live.

------

## 3) Variational characterization (why low modes look “smooth”)

Define the **Dirichlet energy** and Rayleigh quotient:

$\mathcal E[\phi]=\sum_{x=0}^{N-1} c_x\,\big(\phi(x{+}1)-\phi(x)\big)^2,\qquad \mathcal R[\phi]=\frac{\mathcal E[\phi]}{\sum_{x=0}^N w_x \phi(x)^2}.$

Eigenfunctions minimize $\mathcal R[\cdot]$ under $w$-orthogonality constraints:

$\phi_0=\arg\min \mathcal R,\quad  \phi_1=\arg\min_{\langle\phi,\phi_0\rangle_w=0}\mathcal R,\ \dots$

Thus **$c_x$** controls where curvature is cheap/expensive; **$w_x$** controls in which regions function mass (and orthogonality) is measured. This is the precise sense in which **curvature and data geometry are decoupled**.

------

## 4) Relation to continuous Sturm–Liouville

The continuous Sturm–Liouville form on an interval is

$-\frac{d}{dx}\!\left(p(x)\,\frac{dy}{dx}\right)+q(x)\,y=\lambda\,w(x)\,y,$

with boundary conditions ensuring self-adjointness. Our discrete operator is the **finite-difference analogue** with $p\leftrightarrow c$, $w\leftrightarrow w$, $q\equiv 0$. Key properties that carry over:

-   **Self-adjointness** in $\ell^2(w)$ (discrete Green’s identity holds).
-   **Real spectrum**, **nonnegative** eigenvalues, with $\lambda_0=0$ for reflecting boundaries (constant mode).
-   **Orthogonality** in the $w$-inner product.
-   **Oscillation property** on a path: the $k$-th eigenvector has exactly $k$ sign changes (discrete Sturm oscillation theorem).
-   **Min–max (Courant–Fischer)** characterization via the Rayleigh quotient.

These come from standard spectral graph/Laplacian and birth–death chain theory on paths (the path graph is a discrete 1-D Sturm–Liouville setting).

------

## 5) Three compatible mental models

-   **Electrical network.** $c_x$ are **edge conductances**. Current on edge $x$ under potential $\phi$ is $I_x=c_x(\phi(x)-\phi(x{+}1))$. The operator is (mass-weighted) **divergence of current**.
-   **Mass–spring chain.** $w_x$ are **masses**, $c_x$ are **spring stiffnesses**. Eigenmodes are normal modes under the mass-weighted inner product.
-   **Birth–death Markov chain.** Let $\lambda_x=c_x/w_x$, $\mu_x=c_{x-1}/w_x$. Then $L=-G$, where $G$ is the generator that jumps $x\to x\pm1$ at rates $\lambda_x,\mu_x$. Detailed balance $w_x\lambda_x=w_{x+1}\mu_{x+1}=c_x$ makes it **reversible** with stationary $w$. Eigenpairs give **relaxation modes**.

All three are the same linear algebra; pick whichever intuition helps.

------

## 6) Choosing the two knobs in practice

**Weight $w_x$** (data geometry):

-   Fit from data (see §8). For heavy tails, a common parametric choice is

    $$w(x) ∝ (x+a)−b,a>0, b>0,w(x)\ \propto\ (x+a)^{-b},\qquad a>0,\ b>0,$$

    learned from training data. Only **relative** weights matter (you don’t need $\sum w_x=1$ for orthonormality).

**Conductance $c_x$** (curvature budget):

-   $c_x\equiv 1$: uniform slope penalty per edge.
-   $c_x\propto (x{+}1)^\gamma$ with $\gamma>0$: **cheap near 0**, expensive far out ⇒ low modes spend curvature near the origin; as $k$ increases, nodes drift outward (what you observed).
-   Piecewise or capped schedules are fine (e.g., $(x{+}1)^\gamma$ up to $X_\mathrm{turn}$ then constant).
-   Scaling $c$ by a constant rescales eigenvalues but **does not change eigenvectors**.

------

## 7) Numerics (stable, fast, and minimal)

Below is a compact implementation that takes a weight vector `w` on $\{0,\dots,N\}$, a conductance profile `c` on edges $\{0,\dots,N-1\}$, and returns the first `k` eigenpairs. It solves the symmetric tridiagonal problem for $S=W^{-1/2}MW^{-1/2}$ and maps back.

```python
import numpy as np

def _eigh_tridiag(d, e, k):
    """Small helper: symmetric tridiagonal eigensolve for first k eigenpairs."""
    try:
        from scipy.linalg import eigh_tridiagonal
        vals, vecs = eigh_tridiagonal(d, e, select='i', select_range=(0, k-1))
    except Exception:
        S = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
        vals_all, vecs_all = np.linalg.eigh(S)
        vals, vecs = vals_all[:k], vecs_all[:, :k]
    return vals, vecs

def discrete_sl_eigenbasis(w, c, k):
    """
    Compute first k eigenpairs of M phi = lambda W phi on x=0..N
    given node weights w (N+1,) and edge conductances c (N,).
    Returns (evals, Phi) with Phi[:,j] w-orthonormal.
    """
    w = np.asarray(w, float); c = np.asarray(c, float)
    if w.ndim != 1 or c.ndim != 1 or len(c) != len(w)-1:
        raise ValueError("Expect w shape (N+1,), c shape (N,).")
    if np.any(w <= 0) or np.any(c <= 0): 
        raise ValueError("w and c must be strictly positive.")

    N = len(w) - 1
    # Build M implicitly through d,e of S = W^{-1/2} M W^{-1/2}
    # For path graph with reflecting ends:
    # S diagonal: d_x = (c_{x-1}/w_x) + (c_x/w_x)  (with c_{-1}=c_N=0)
    # S offdiag:  e_x = - sqrt( (c_x/w_x) * (c_x/w_{x+1}) )
    d = np.empty(N+1, float)
    d[0]     = c[0]/w[0]
    d[1:N]   = c[:N-1]/w[1:N] + c[1:]/w[1:N]
    d[N]     = c[N-1]/w[N]
    e = -np.sqrt( (c / w[:-1]) * (c / w[1:]) )  # length N

    # Solve S u = lambda u
    evals, U = _eigh_tridiag(d, e, k)

    # Map back: Phi = W^{-1/2} U, then w-normalize (guarding roundoff)
    Phi = U / np.sqrt(w)[:, None]
    Phi /= np.sqrt((Phi**2 * w[:, None]).sum(axis=0))
    return evals, Phi
```

**Complexity.** Building `(d,e)` is $O(N)$; tridiagonal eigensolve for the first $k$ modes is $O(Nk)$ and very fast for $k\ll N$.

------

## 8) Fitting a heavy-tailed weight $w$ from data (robustly)

Suppose you observe non-negative integers $x_1,\dots,x_n$, heavy-tailed, and will truncate at $0..M$ for the basis.

### 8.1 Parametric MLE (shifted power law)

Model

$w(x)\ \propto\ (x+a)^{-b},\qquad a>0,\ b>0.$

On the finite support $0..M$, let $c_x$ be counts (a histogram). The **negative** log-likelihood (up to constants) is

$L(a,b)= b\sum_{x=0}^M c_x \log(x+a) \;+\; n \log\!\left(\sum_{x=0}^M (x+a)^{-b}\right),\quad n=\sum_x c_x.$

For **fixed $a$**, $L$ is **convex in $b$** (indeed strictly convex unless all $\log(x+a)$ are equal). So you can:

1.  Pick a modest **grid of $a$** values (e.g., 30–50 log-spaced).
2.  For each $a$, **maximize over $b$** by 1-D Newton or Brent using analytic derivatives.
3.  Choose $(a,b)$ with the best likelihood. (Optionally: a few L-BFGS steps as joint refinement.)

This is very fast because everything works on the **histogram**, not the raw samples.

>   **Zero-counts caution.** If parts of the support are **sparse**, the raw histogram has zeros; that’s fine for MLE (the likelihood ignores unsampled points). The *basis* construction, however, needs strictly **positive** weights $w_x$. After you fit $(a,b)$, evaluate $w_x\propto(x+a)^{-b}$ for all $x$ and (optionally) add a small floor $\epsilon$ for numerical stability. You do **not** need to normalize $w$.

A compact fitter (as discussed earlier):

```python
import numpy as np

def fit_shifted_power_weight(x, M=None, a_grid=None, newton_steps=8):
    x = np.asarray(x, int)
    if np.any(x < 0): raise ValueError("x must be non-negative integers.")
    if M is None: M = int(x.max())
    c = np.bincount(x, minlength=M+1).astype(float)
    n = c.sum(); xs = np.arange(M+1, dtype=float)

    def sums(a, b):
        xa = xs + a
        pow_ = xa**(-b)
        Z = pow_.sum()
        logxa = np.log(xa)
        S1 = (c * logxa).sum()                  # ∑ c_x log(x+a)
        U1 = (logxa * pow_).sum()               # ∑ log(x+a) (x+a)^(-b)
        U2 = ((logxa**2) * pow_).sum()          # ∑ [log(x+a)]^2 (x+a)^(-b)
        return Z, S1, U1, U2

    def maximize_b_given_a(a, b0=1.1):
        b = max(b0, 1e-6)
        for _ in range(newton_steps):
            Z, S1, U1, U2 = sums(a, b)
            g  = -S1 + n * (U1 / Z)                             # ∂L/∂b
            h  =  n * (U2 / Z - (U1 / Z)**2)                    # ∂²L/∂b²
            step = g / (h + 1e-18)
            b_new = b - step
            if not np.isfinite(b_new) or b_new <= 0: b_new = max(0.1, 0.5*b)
            if abs(b_new - b) < 1e-9: 
                b = b_new; break
            b = b_new
        Z, S1, U1, U2 = sums(a, b)
        ll = - (b * S1 + n * np.log(Z))                         # log-likelihood (neg L)
        return b, ll

    if a_grid is None:
        # set a reasonable range based on the data scale
        pos = x[x>0]; scale = np.median(pos) if pos.size else 1.0
        a_grid = np.geomspace(1e-3, max(1.0, 0.3*scale), num=32)

    best = (-np.inf, None, None)
    for a in a_grid:
        b, ll = maximize_b_given_a(a)
        if ll > best[0]: best = (ll, a, b)

    _, a_hat, b_hat = best
    return a_hat, b_hat
```

**When to prefer infinite support.** If you don’t truncate and truly model $\{0,1,2,\dots\}$, replace $\sum_{x=0}^M$ by the **Hurwitz zeta** $\zeta(b,a)$ and optimize with $a>0,\,b>1$. (You’ll need a special function library or autodiff; the finite-$M$ version above is often simpler and matches your basis truncation anyway.)

### 8.2 Non-parametric weight from data (no zeros)

If you don’t want a parametric form, you still need strictly positive $w_x$. Two simple options:

-   **Additive smoothing + mild blur.** Form counts $c_x$, add a small prior $\alpha>0$ (e.g., $\alpha=1$ or $\alpha=0.1$), then convolve with a tiny kernel (e.g., $[0.25,0.5,0.25]$) once or twice to fill gaps. Set $w_x \propto \max(\epsilon, c_x+\alpha)$ and (optionally) tail-smooth by averaging in logarithmic bins.
-   **Fit in the log-domain by regression.** Regress $\log(\hat p_x)$ on $\log(x+a)$ with a small ridge penalty and **monotonicity** constraints (or just smooth the log-histogram with a Savitzky–Golay filter). Exponentiate to get strictly positive $w_x$. This is a quick way to get a smooth heavy-tailed *shape* without committing to a specific parametric family.

Either way, remember: for orthonormality, only **relative** $w_x$ matter; a global scaling of $w$ cancels out.

------

## 9) Practical recipes, tips, and sanity checks

-   **Truncation $N$.** Choose $N$ at a high quantile (e.g., 99.9%) and bucket any larger $x$ into $N$. Build $w$ over $\{0,\dots,N\}$; set $c_N=0$ (reflecting) or use an absorbing end if you prefer.
-   **Choosing $c_x$.** Start with $c_x\propto(x{+}1)^\gamma$ and tune $\gamma\in[0.5,2]$ by cross-validation on downstream loss. Larger $\gamma$ concentrates early modes near 0 and pushes nodes outward more aggressively.
-   **Intercept.** The smallest eigenfunction is constant. Most linear models already include an intercept; you can drop the constant mode from your feature set (or set `include_phi0=False` if you wrap this as a transformer).
-   **Orthogonality check.** Numerically verify $\Phi^\top \mathrm{diag}(w)\,\Phi \approx I$ after you compute the first $K$ modes.
-   **Stability.** Add a tiny floor to $w_x$ (e.g., $w_x\leftarrow w_x+\epsilon$) if needed; all entries must be strictly positive to build $W^{-1/2}$.

------

## 10) Why this works (the conceptual TL;DR)

-   The path graph with edge conductances $c_x$ and node weights $w_x$ is the **discrete 1-D Sturm–Liouville** setting.
-   Solving $M\phi=\lambda W\phi$ gives a basis that is (i) orthonormal in $\ell^2(w)$ (**matches your data geometry**), and (ii) ordered by increasing Dirichlet energy (**curvature spent according to $c_x$**).
-   Because $c_x$ and $w_x$ enter in **different places** (numerator vs denominator of the Rayleigh quotient), you can **fit $w$ from data** and **tune $c$** to express the inductive bias you want—two independent levers.

------

## 11) (Very) short literature pointers

You don’t need these to use the method, but for theoretical background:

-   **Spectral graph/Laplacians:** Fan Chung, *Spectral Graph Theory* (CBMS, 1997).
-   **Random walks ↔ electrical networks:** Doyle & Snell, *Random Walks and Electric Networks* (Carus, 1984).
-   **Birth–death processes & orthogonality:** Karlin & McGregor’s classic series (1950s–60s).
-   **Discrete oscillation (Sturm theory on paths):** standard results in spectral graph texts.

------

## 12) Minimal end-to-end snippet

```python
# 1) Fit w from data (parametric), build w over 0..N
x_obs = ...  # your integer samples
N = 1000
a_hat, b_hat = fit_shifted_power_weight(x_obs, M=N)
xs = np.arange(N+1, dtype=float)
w = (xs + a_hat)**(-b_hat) + 1e-12   # tiny floor for stability

# 2) Choose conductance schedule
gamma = 1.0
c = (np.arange(N) + 1.0)**gamma

# 3) Compute first K eigenfunctions
K = 12
evals, Phi = discrete_sl_eigenbasis(w, c, K)  # Phi shape: (N+1, K), w-orthonormal

# 4) Make features for a dataset of integer x-values (clipped to [0,N])
X = np.clip(np.asarray(..., int), 0, N)
Features = Phi[X, :]  # plug into your linear/GLM model
```

------

If you’d like, I can wrap this into a tiny scikit-learn transformer (with `fit/transform` and cross-validation over $\gamma$ and the $w$ fit) or a PyTorch module that precomputes $\Phi$ and does fast table lookups on GPU.









---

# Datasets

Here are solid, public datasets where **integer, heavy-tailed counts** show up naturally and your basis is a good fit. For each, I note a concrete task you can run with a (mostly) linear model after expanding the count(s) with our eigenfeatures.

1.  \#text — **20 Newsgroups**
     Use BoW counts (heavy-tailed per token). Task: linear classification with our basis applied per token (or to a few high-MI tokens for a quick demo). Easy to load via scikit-learn. ([Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html?utm_source=chatgpt.com))
2.  \#reviews — **Amazon Reviews** (UCSD / McAuley lab)
     For each user or item, build integer features like “past review count”, “helpfulness vote count”, “item popularity count”. Tasks: predict 4+ star vs not, or helpfulness>k using only a few count features expanded with our basis. (Very heavy-tailed.) ([jmcauley.ucsd.edu](https://jmcauley.ucsd.edu/data/amazon/?utm_source=chatgpt.com), [Computer Science](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html?utm_source=chatgpt.com), [amazon-reviews-2023.github.io](https://amazon-reviews-2023.github.io/?utm_source=chatgpt.com))
3.  \#reviews — **Yelp Open Dataset (HuggingFace mirror)**
     Similar to Amazon: use counts such as “user review count”, “business review count” or token counts from text. Task: star rating (classification) from a couple of count features + bias. ([Hugging Face](https://huggingface.co/datasets/Yelp/yelp_review_full?utm_source=chatgpt.com))
4.  \#ads — **Criteo Display Advertising Challenge (CTR)**
     Build time-aware aggregates like “past impressions for this user/ad” as integer features (they’re heavy-tailed). Task: click/no-click with logistic regression on our expanded counts. ([Kaggle](https://www.kaggle.com/competitions/criteo-display-ad-challenge?utm_source=chatgpt.com), [Figshare](https://figshare.com/articles/dataset/Kaggle_Display_Advertising_Challenge_dataset/5732310?utm_source=chatgpt.com))
5.  \#traffic — **Wikipedia Web Traffic Time Series**
     Page-view counts are extremely heavy-tailed across pages. Task: classify “high-traffic” vs “normal” pages using simple summaries (e.g., last-week total count) expanded with our basis; or build linear baselines for next-week thresholding. ([Kaggle](https://www.kaggle.com/competitions/web-traffic-time-series-forecasting?utm_source=chatgpt.com))
6.  \#recsys — **MovieLens (GroupLens)**
     Counts like “user historical ratings” or “item popularity (ratings so far)” are heavy-tailed. Task: predict ≥4 stars using only these two count features (expanded) + bias; compare vs raw/log counts. ([GroupLens](https://grouplens.org/datasets/movielens/?utm_source=chatgpt.com), [Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?utm_source=chatgpt.com))
7.  \#retail — **Online Retail II (UCI)**
     Line-item **Quantity** (integer) is heavy-tailed; you can also aggregate “customer purchases so far”. Task: predict **Cancellation** (credits) or high basket value using our basis on Quantity and/or customer-count features. ([UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/online%2Bretail%2BII?utm_source=chatgpt.com), [Kaggle](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci?utm_source=chatgpt.com))
8.  \#mobility — **NYC Taxi Trip Records (TLC)**
     Trip counts per zone/hour are heavy-tailed. Task: predict whether next-hour demand in a zone exceeds a threshold using last-hour count (expanded) + a few simple covariates. ([NYC Government](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page?utm_source=chatgpt.com))
9.  \#dev-activity — **GH Archive (GitHub events)**
     Event counts per repo/user (stars, pushes, PRs) are classic heavy tails. Task: predict “repo will get ≥T stars next week” using last-week event counts expanded with our basis. (Easy via BigQuery/ClickHouse mirrors.) ([gharchive.org](https://www.gharchive.org/?utm_source=chatgpt.com), [ghe.clickhouse.tech](https://ghe.clickhouse.tech/?utm_source=chatgpt.com), [TIL](https://til.simonwillison.net/clickhouse/github-explorer?utm_source=chatgpt.com))
10.  \#demand — **Bike Sharing (UCI)**
      Hourly/daily rental counts; try linear (or Poisson) regression with our basis on the **lag-1 count** and compare to raw/log. Great for quick, interpretable demos. ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike%2Bsharing%2Bdataset?utm_source=chatgpt.com), [islp.readthedocs.io](https://islp.readthedocs.io/en/latest/datasets/Bikeshare.html?utm_source=chatgpt.com))

If you want, I can spin up a tiny, uniform **benchmark script** that: (i) fits $w(x)\propto (x+a)^{-b}$ on train, (ii) builds $c_x\propto (x+1)^\gamma$, (iii) trains logistic/Poisson baselines with/without our features, and (iv) reports CV metrics + calibration plots.



# Research plan

>   Awesome—here’s a concrete, student-friendly research plan you can run with. It’s scoped for ~10–12 weeks but works flexibly. It mixes theory (lightweight), strong engineering, and convincing experiments.
>
>   # 0) North-star & Claims
>
>   **Goal:** show a simple, principled basis for integer, heavy-tailed features that (i) fits the **data geometry** via $w_x$ and (ii) controls **inductive bias** via $c_x$, and that this **improves linear/GLM baselines** on real tasks.
>
>   **Core claims to test**
>
>   1.  Fitting $w(x)\propto (x+a)^{-b}$ on train + setting $c_x\propto(x+1)^\gamma$ yields features that beat raw, log, splines, and classical discrete polynomials at equal feature budget.
>   2.  $c_x$ is a single, interpretable knob: increasing $\gamma$ moves oscillations outward and improves tasks with “detail-near-0, flat-tail” targets.
>   3.  It’s cheap: $O(NK)$ precompute; transforms are table lookups.
>
>   ------
>
>   # Phase 1 — Formalization & Minimal Prototype (Week 1–2)
>
>   **Deliverables**
>
>   -   Short technical note (3–5 pages) with:
>       -   Operator $L=W^{-1}\nabla^*(C\nabla)$, generalized EVP $M\phi=\lambda W\phi$, Rayleigh quotient.
>       -   Properties: self-adjointness, nonnegativity, $w$-orthogonality, oscillation count on a path (with references).
>       -   Boundary choices (reflecting vs absorbing) and their effect on the zero mode.
>   -   Minimal NumPy/Scipy implementation:
>       -   `discrete_sl_eigenbasis(w, c, K)` (symmetric tridiagonal solve).
>       -   Weight fitter `fit_shifted_power_weight(x, M)` (MLE, grid+Newton).
>   -   Visualization notebook: plots of modes for several $\gamma$, sanity checks of orthogonality.
>
>   **Student tasks**
>
>   -   Reproduce the figures you already generated (0..50, 0..200, 0..1000).
>   -   Unit tests: check $\Phi^\top \mathrm{diag}(w)\Phi \approx I$; eigenvalue monotonicity; zero-mode constancy.
>
>   ------
>
>   # Phase 2 — Robust Engineering (Week 3–4)
>
>   **Library skeleton**
>
>   -   `discrete_sl/`
>       -   `basis.py` (EVP solver, boundary options, caching)
>       -   `weight.py` (parametric MLE; nonparametric smoothed estimator with positivity)
>       -   `features.py` (scikit-learn style transformer: `fit/transform`, `X_max`, `include_phi0`, `gamma`)
>       -   `viz.py` (plotting helpers)
>       -   `datasets/` loaders (wrappers around common sources)
>       -   `tests/` (pytest)
>   -   GPU option later: precompute $\Phi$ on CPU; transfer table to GPU for fast lookup.
>
>   **UX & stability**
>
>   -   Safe defaults: add tiny floor to $w$; clip domain to `X_max`.
>   -   Log instrumentation (timing), deterministic seeds.
>   -   Configurable $c_x$: $(x+1)^\gamma$, piecewise, capped, or custom callable.
>
>   ------
>
>   # Phase 3 — Benchmarks & Baselines (Week 5–8)
>
>   Pick **3–4 datasets** to keep the scope realistic and representative:
>
>   -   **Text (20 Newsgroups)**: target = category; features = a handful of token counts (top MI tokens) → logistic regression.
>   -   **MovieLens or Yelp**: target = rating ≥ 4; features = user historic rating count, item popularity count → logistic.
>   -   **Criteo (or a smaller click dataset)**: target = click; features = past exposure counts (user, ad) → logistic.
>   -   **Bike Sharing or NYC taxi (tabular)**: regression on next-period demand using last-period count → Poisson or MSE.
>
>   **Feature sets per task** (same feature budget $K$, e.g., $K=8, 16, 32$)
>
>   -   Ours: eigenfeatures with learned $w$ and tuned $\gamma$.
>   -   Raw $x$, $\log(1+x)$.
>   -   **B-splines** on $x$ (knots log-spaced).
>   -   **Charlier/Meixner** polynomials (carefully normalized) with tuned parameters.
>   -   Small MLP (1–2 hidden layers) as a non-linear baseline.
>   -   (Optional) LightGBM/XGBoost on raw features—strong but useful context.
>
>   **Models**
>
>   -   Logistic / Ridge / Poisson GLM with standard CV.
>   -   Same regularization budget across feature sets.
>
>   **Metrics**
>
>   -   Classification: log-loss, AUC, ECE (calibration), Brier.
>   -   Regression: RMSE/MAPE, Poisson deviance (for counts).
>   -   **Sample-efficiency curves**: performance vs train size.
>   -   **Feature-budget curves**: performance vs $K$.
>
>   **Ablations**
>
>   -   Fit $w$ vs fixed crude $w$ (e.g., $1/(x+1)$): how much do we gain by learning $w$?
>   -   $\gamma$ sweep: show node-drift qualitatively and performance quantitatively.
>   -   Boundary (reflecting vs absorbing at $X_{\max}$).
>   -   Sensitivity to `X_max` and tail bucketing.
>   -   Time/memory: eigensolve cost vs $N,K$; transform latency.
>
>   **Success bar**
>
>   -   Consistent improvements over log/B-splines and discrete polynomials at the same $K$ (e.g., −3–10% log-loss or deviance).
>   -   Better calibration on classification with minimal feature count.
>
>   ------
>
>   # Phase 4 — Extensions (Week 9–10, parallelizable)
>
>   Pick 1–2 if time permits:
>
>   -   **Learn $c_x$ shape**: $\gamma$ as a hyperparameter via CV; or parametrize $c_x=\exp\big(\alpha_0+\alpha_1\log(x+1)\big)$ (monotone), select via CV. Keep it 1–2 dof to stay simple.
>   -   **Add potential $q_x$** (diagonal term) to bias amplitude (e.g., discourage far-tail mass).
>   -   **Multi-feature combos**: tensor products for two counts (user count $\times$ item count) with low-rank truncation.
>   -   **Streaming update of $w$**: maintain smoothed counts and refit $(a,b)$ periodically.
>
>   ------
>
>   # Phase 5 — Writing & Packaging (Week 11–12)
>
>   **Paper outline (TMLR length)**
>
>   1.  **Intro**: Problem & contribution (decoupling geometry vs bias, practical wins).
>   2.  **Method**: Operator, generalized EVP, change of basis, properties (brief proofs / citations).
>   3.  **Fitting $w$**: MLE, nonparametric smoothed positive estimator; pitfalls (zeros, truncation).
>   4.  **Designing $c$**: $(x+1)^\gamma$; effect on node locations (intuition + plots).
>   5.  **Experiments**: set-up, baselines, metrics, results, ablations, calibration, compute.
>   6.  **Discussion**: when it shines/fails; limitations (1D integer only; feature interactions).
>   7.  **Reproducibility**: code, config, seeds, data access.
>
>   **Artifacts**
>
>   -   Public repo with `pip install discrete-sl-features`.
>   -   `scripts/` to reproduce each table/figure.
>   -   `weights/` and `features/` cache for big datasets (or scripts to build).
>
>   ------
>
>   ## Roles & Cadence
>
>   -   **Student**: implement library, run benchmarks/ablations, draft Experiments section; weekly demo.
>   -   **You**: steer datasets/metrics, refine $c_x$ choices, write theory & framing sections; review weekly.
>   -   **Milestones**
>       -   Wk2: prototype + plots; note finished.
>       -   Wk6: first benchmark table (at least 2 datasets), ablation plan.
>       -   Wk9: full results, figures ready.
>       -   Wk12: paper draft + cleaned repo.
>
>   ------
>
>   ## Risks & Mitigations
>
>   -   **Weak gains vs strong tree models**: emphasize **linear/GLM** regimes and calibration; include MLP baseline for fairness but set scope.
>   -   **Overfitting $\gamma$**: do nested CV or holdout; keep $\gamma$ coarse-grained.
>   -   **Sparse tails & zeros**: always use strictly positive $w$ (parametric or smoothed), bucket the extreme tail, reflect at the boundary.
>   -   **Leakage**: fit $w$ **only on train**; lock `X_max` from train statistics.
>
>   ------
>
>   ## Quick Start Checklist (student)
>
>   1.  Implement `fit_shifted_power_weight` and verify convexity in $b$ for fixed $a$.
>   2.  Build eigenbasis for $w$ and $c_x=(x+1)^\gamma$; verify orthogonality; plot node drift as $\gamma$ increases.
>   3.  Run a small benchmark on 20NG (2–3 tokens) and MovieLens (2 counts), compare vs log and splines at $K=\{8,16,32\}$.
>   4.  Add calibration plots & sample-efficiency curves.
>   5.  Lock experiment harness, scale to the other datasets.
>
>   ------
>
>   If you want, I can turn this into a GitHub project scaffold (issues + milestones + CI) and a 2-page “project brief” for the intern to kick off with on day one.