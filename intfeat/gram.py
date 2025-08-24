import numpy as np


def powerlaw_weight(alpha, beta, n):
    """Compute the power-law weights.

    Parameters
    ----------
    alpha : float
        The alpha parameter for the power-law.
    beta : float
        The beta parameter for the power-law.
    n : int
        The number of elements.

    Returns
    -------
    np.ndarray
        A numpy array of shape (n,) containing the power-law weights.
    """
    xs = np.arange(n).astype(float)
    return np.reciprocal((xs + alpha) ** beta)


def gram_matrix(vander, weights):
    """Compute the Gram matrix.

    Parameters
    ----------
    vander : np.ndarray
        The Vandermonde matrix.
    weights : np.ndarray
        The weights to apply.

    Returns
    -------
    np.ndarray
        The Gram matrix.
    """
    weighted_vander = vander * weights[:, np.newaxis]
    return weighted_vander.T @ weighted_vander


def powerlaw_gram_matrix(vander, alpha, beta):
    """Compute the power-law weighted Gram matrix.

    Parameters
    ----------
    vander : np.ndarray
        The Vandermonde matrix.
    alpha : float
        The alpha parameter for the power-law.
    beta : float
        The beta parameter for the power-law.

    Returns
    -------
    np.ndarray
        The power-law weighted Gram matrix.
    """
    n, _ = vander.shape
    weights = powerlaw_weight(alpha, beta, n)
    return gram_matrix(vander, weights)
