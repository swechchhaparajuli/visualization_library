"""Colors for forest-loss charts.

The categorical hues are the first seven slots of a colorblind-safe,
fixed-order palette (validated for CVD separation and normal-vision
contrast in both light and dark modes). Each driver is bound to a fixed
color so identity never depends on rank -- a filter that reorders the
drivers must not repaint them.

Drivers are ordered by their global contribution to primary-forest loss,
so the largest band uses the leading (most distinguishable) slot.
"""

from __future__ import annotations

# Canonical driver order (largest global contribution first).
DRIVER_ORDER = [
    "Permanent agriculture",
    "Shifting cultivation",
    "Wildfire",
    "Logging",
    "Other natural disturbances",
    "Hard commodities",
    "Settlements & Infrastructure",
]

# Categorical slots 1-7, fixed order, validated palette.
_SLOTS_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
_SLOTS_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]

DRIVER_COLORS = dict(zip(DRIVER_ORDER, _SLOTS_LIGHT))
DRIVER_COLORS_DARK = dict(zip(DRIVER_ORDER, _SLOTS_DARK))

# Single-hue accent for one-series charts (blue slot 1).
ACCENT = "#2a78d6"
ACCENT_DARK = "#3987e5"

# Chart chrome / ink.
LIGHT = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
}
DARK = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "baseline": "#383835",
}


def driver_colors(dark: bool = False) -> dict[str, str]:
    """Return the driver -> hex color map for the chosen mode."""
    return dict(DRIVER_COLORS_DARK if dark else DRIVER_COLORS)


def chrome(dark: bool = False) -> dict[str, str]:
    """Return the chart chrome/ink colors for the chosen mode."""
    return dict(DARK if dark else LIGHT)
