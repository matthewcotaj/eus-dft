"""Shared plot style so every figure in the repo looks the same."""

import matplotlib

matplotlib.use("Agg")   # draw to files, no window needed
import matplotlib.pyplot as plt

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#d9d8d4"

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "lines.linewidth": 2,
    "lines.markersize": 6,
    "legend.frameon": False,
})
