"""A recessive matplotlib theme for forest-loss charts.

Thin hairline grid, no top/right spines, muted ticks, sans-serif type.
Call :func:`apply_theme` before plotting, or pass ``dark=True`` for the
dark surface.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager as _fm

from forest_viz import palette

# Register the bundled Helvetica-equivalent (Nimbus Sans) once.
_FONT_DIR = Path(__file__).with_name("assets") / "fonts"
_FONT_FAMILY = "sans-serif"
for _f in sorted(_FONT_DIR.glob("*.otf")):
    try:
        _fm.fontManager.addfont(str(_f))
        _FONT_FAMILY = _fm.FontProperties(fname=str(_f)).get_name()  # "Nimbus Sans"
    except Exception:
        pass


def apply_theme(dark: bool = False) -> None:
    """Set global matplotlib rcParams for the forest-loss look."""
    c = palette.chrome(dark=dark)
    mpl.rcParams.update(
        {
            "figure.facecolor": c["surface"],
            "axes.facecolor": c["surface"],
            "savefig.facecolor": c["surface"],
            # Helvetica; the bundled Nimbus Sans is a metric-identical clone
            # used when Helvetica itself isn't installed.
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", _FONT_FAMILY, "Nimbus Sans", "Arial",
                                 "Liberation Sans", "DejaVu Sans"],
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
