# Eigen / Variational Facts (Quick Reference)

## Rayleigh quotient (standard symmetric case)

Let `A` be real symmetric. The Rayleigh quotient is

`R_A(x) = (x^T A x) / (x^T x)` for `x != 0`.

- `min_x R_A(x) = lambda_min(A)` and `max_x R_A(x) = lambda_max(A)`.
- Any stationary point of `R_A` is an eigenvector of `A`.

Reference: Wikipedia, "Min-max theorem".  
https://en.wikipedia.org/wiki/Min-max_theorem

## Courant–Fischer / min-max principle

Let `A` be real symmetric with eigenvalues `lambda_1 <= ... <= lambda_n`. Then

- `lambda_k = min_{dim(S)=k} max_{x in S, x!=0} R_A(x)`
- `lambda_k = max_{dim(S)=n-k+1} min_{x in S, x!=0} R_A(x)`

This is the cleanest way to reason about how changing an operator changes its low-frequency eigenmodes.

Reference: Wikipedia, "Courant minimax principle".  
https://en.wikipedia.org/wiki/Courant_minimax_principle

## Generalized eigenproblems (SPD mass matrix)

For symmetric `A` and symmetric positive-definite `B`, the generalized eigenproblem

`A v = lambda B v`

has real eigenvalues and a `B`-orthonormal eigenbasis: you can choose eigenvectors `{v_k}` such that

`v_i^T B v_j = delta_ij`.

The generalized Rayleigh quotient is

`R_{A,B}(x) = (x^T A x) / (x^T B x)`.

The eigenvalues satisfy the same Courant–Fischer min-max characterization but with `R_{A,B}`.

Reference: SJSU lecture notes (Rayleigh quotient + generalized eigenproblem).  
https://www.sjsu.edu/faculty/guangliang.chen/Math253S20/lec10gen_rayleigh.pdf

## Why “first K eigenvectors” are a good approximation space (energy vs error)

This repo’s discrete Sturm–Liouville construction is a generalized eigenproblem of the form

`L phi_k = lambda_k W phi_k`,

where

- `L` is a (weighted) graph Laplacian on a 1D grid (conductances),
- `W` is a diagonal matrix with positive node weights.

Assume the eigenvectors `{phi_k}` are `W`-orthonormal and eigenvalues are ordered
`0 = lambda_0 <= lambda_1 <= ...`.

Expand any vector/function `f` as `f = sum_k a_k phi_k`, where `a_k = <f, phi_k>_W`.

Then the Laplacian energy and weighted `L2(W)` norm diagonalize:

- `||f||_W^2 = f^T W f = sum_k a_k^2`
- `E(f) = f^T L f = sum_k lambda_k a_k^2`

Projection onto the first `K` eigenvectors keeps the components with smallest energy-per-norm ratio.
This yields a simple bound:

- If `f` is orthogonal (in `W`) to the first `K` modes, then `||f||_W^2 <= E(f) / lambda_{K+1}`.

Interpretation:

- Larger `lambda_{K+1}` means a sharper “spectral gap” after `K`, i.e., a better low-dimensional approximation space for functions of bounded energy.

The point of the conductance schedule is to choose what counts as “energy” (where variation is expensive), while `W` chooses what counts as “error” (where approximation should care).

## Perturbation quick facts (stability)

These are useful when thinking about how the SL basis changes when the estimated `w` (pmf) changes.

- **Weyl (eigenvalues)**: for Hermitian `A` and perturbation `E`, eigenvalues of `A` and `A+E` move by at most `||E||_2` (spectral norm) when paired in sorted order.
  - One accessible statement of this inequality (and extensions to generalized eigenvalues) is in this open-access survey-style paper:
    https://journalofinequalitiesandapplications.springeropen.com/articles/10.1186/s13660-018-1749-0
- **Davis–Kahan / sin-theta (eigenspaces)**: eigenvectors/subspaces are stable when there is a spectral gap; the subspace angle is controlled by `||E||` divided by the eigengap.
  - Classic reference: Davis & Kahan (1970), SIAM J. Numer. Anal., DOI `10.1137/0707001`.
    (One index page with DOI + link to the SIAM PDF: https://cir.nii.ac.jp/crid/1363107370704474880)
