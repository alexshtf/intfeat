# Graph Laplacians and Discrete Sturm–Liouville (1D)

This repo’s core eigenbasis is a *generalized eigenproblem on a weighted path graph*.

## Weighted 1D Laplacian from conductances

Let nodes be `0,1,...,N-1` with edge conductances `c_i > 0` on edges `(i, i+1)` for `i=0..N-2`.

Define the (weighted) graph Laplacian `L` for the path graph. As a quadratic form:

`f^T L f = sum_{i=0}^{N-2} c_i (f_{i+1} - f_i)^2`.

So:

- Large `c_i` makes variation across edge `(i,i+1)` expensive.
- Small `c_i` makes variation cheap, so low-energy modes can “spend oscillations” there.

Reference: Wikipedia, "Laplacian matrix" (graph Laplacian + quadratic form).  
https://en.wikipedia.org/wiki/Laplacian_matrix

## Node weights (“mass matrix”) from the data distribution

Let `w_i > 0` be node weights (typically a pmf over observed integer values). Define

`W = diag(w_0, ..., w_{N-1})`.

The weighted inner product is `<f,g>_W = f^T W g = sum_i w_i f_i g_i`.

This is the mechanism for “distribution adaptation”: errors in regions with larger `w_i` count more in `L2(W)`.

## Discrete Sturm–Liouville = generalized eigenproblem

The basis functions are the eigenvectors of

`L phi = lambda W phi`.

Equivalently, if `W` is SPD, this is the standard symmetric eigenproblem

`W^{-1/2} L W^{-1/2} v = lambda v`, where `phi = W^{-1/2} v`.

This is exactly what the code does in `intfeat/strum_liouville.py`:

- build `L` from conductances `c`,
- form `W^{-1/2} L W^{-1/2}`,
- take its first `K` eigenvectors,
- map back to a `W`-orthonormal basis `phi`.

## What “two knobs” means (and the coupling that remains)

From the generalized Rayleigh quotient:

`lambda(phi) = (phi^T L phi) / (phi^T W phi)`

you can read off two roles:

- `L` (via `c`) defines what it means to be “smooth” (where variation costs energy).
- `W` (via `w`) defines what it means to be “small” / “important” (where approximation error is measured).

However, the eigenvectors are solutions to a *coupled* optimization problem: changing either `c` or `w` changes the eigenvectors themselves, not just their ordering.
So the knobs are “conceptually separable” (energy vs inner product), but they are not independent in the literal sense.

## Uniform case: discrete cosine/sine structure

For the unweighted/uniform-conductance path graph, the Laplacian eigenvectors reduce to discrete sine/cosine patterns (the “Fourier modes” of the line).

A friendly pointer into that world (with references to Strang’s DCT paper):

https://web.ece.ucdavis.edu/faculty/richardson/research/graph_harmonics.html

