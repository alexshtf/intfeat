# Symmetric Tridiagonals / Jacobi Matrices (Quick Facts)

Discrete 1D Sturm–Liouville operators are (generalized) eigenproblems for *symmetric tridiagonal* matrices.

## Jacobi matrix (tridiagonal, positive off-diagonals)

A (real) Jacobi matrix is a tridiagonal matrix with positive off-diagonal entries.

Standard consequences (under mild positivity conditions):

- All eigenvalues are real and *simple* (no multiplicities).
- Eigenvectors exhibit a 1D “oscillation” structure (increasing numbers of sign changes as eigenvalues increase).

Reference: Encyclopedia of Mathematics, "Jacobi matrix".  
https://encyclopediaofmath.org/wiki/Jacobi_matrix

## Interlacing (why truncation behaves well)

Eigenvalues of principal submatrices of a real symmetric matrix interlace the eigenvalues of the full matrix.
In 1D problems, this is one way to reason about how changing the cutoff `N` affects the low-frequency eigenvalues.

Reference: Wikipedia, "Cauchy interlacing theorem".  
https://en.wikipedia.org/wiki/Cauchy_interlacing_theorem

