import matplotlib.pyplot as plt
from .strum_liouville import StrumLiouvilleBasis
import math
import numpy as np


def plot_sl_basis(basis: StrumLiouvilleBasis):
    num_funcs = basis.num_funcs
    xs = np.arange(basis.max_val)
    vecs = basis(xs)

    rows = math.floor(math.sqrt(num_funcs))
    cols = math.ceil(num_funcs / rows)
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 3 * rows),
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for i in range(num_funcs):
        ax = axs[i // cols, i % cols]
        ax.plot(xs, vecs[:, i])
        ax.plot(xs[:10], vecs[:10, i], marker="o", markersize=3, linestyle="None")
        ax.set_xscale("asinh", linear_width=10)
        ax.set_title(rf"$\phi_{{{i + 1}}}$")
    return fig
