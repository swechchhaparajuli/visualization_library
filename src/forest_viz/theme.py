"""A recessive matplotlib theme for forest-loss charts.

Thin hairline grid, no top/right spines, muted ticks, sans-serif type.
Call :func:`apply_theme` before plotting, or pass ``dark=True`` for the
dark surface.
"""

from __future__ import annotations

import matplotlib as mpl

from forest_viz import palette


def apply_theme(dark: bool = False) -> None:
    """Set global matplotlib rcParams for the forest-loss look."""
    c = palette.chrome(dark=dark)
    mpl.rcParams.update(
        {
            "figure.facecolor": c["surface"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["surface"],
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
            "font.size": 11,
            "text.color": c["text"],
            "axes.edgecolor": c["baseline"],
            "axes.labelcolor": c["text_secondary"],
            "axes.titlecolor": c["text"],
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": c["grid"],
            "grid.linewidth": 0.8,
            "xtick.color": c["muted"],
            "ytick.color": c["muted"],
            "xtick.labelcolor": c["text_secondary"],
            "ytick.labelcolor": c["text_secondary"],
            "legend.frameon": False,
            "legend.fontsize": 9.5,
            "figure.dpi": 110,
        }
    )
