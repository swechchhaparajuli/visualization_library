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

# Secondary (non-primary) drivers use a fixed pastel-green shade, consistent
# across every country. Order matches DRIVER_ORDER.
_SLOTS_LIGHT = ["#e3f3d8", "#c9e8bd", "#addca3", "#8fd08b", "#6fc276", "#4fb265", "#2f9f57"]
_SLOTS_DARK = ["#b7d9a6", "#9ccb8a", "#7fbd75", "#61ad63", "#469a55", "#2f8547", "#1f6f39"]

# Themed "biome" color used ONLY for a country's primary driver (behind its
# icons), echoing the reference infographic's land-use palette.
_THEME_LIGHT = ["#e4b23e", "#b3c94f", "#d9663a", "#3f8f6e", "#6f93a6", "#8a8f99", "#b5563f"]
_THEME_DARK = ["#c9992f", "#97ad3f", "#bf5330", "#2f7d5e", "#5a7d8f", "#74797f", "#9c4634"]

DRIVER_COLORS = dict(zip(DRIVER_ORDER, _SLOTS_LIGHT))
DRIVER_COLORS_DARK = dict(zip(DRIVER_ORDER, _SLOTS_DARK))

DRIVER_THEME = dict(zip(DRIVER_ORDER, _THEME_LIGHT))
DRIVER_THEME_DARK = dict(zip(DRIVER_ORDER, _THEME_DARK))

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
    """Return the driver -> pastel-green (secondary) color map."""
    return dict(DRIVER_COLORS_DARK if dark else DRIVER_COLORS)


def driver_theme(dark: bool = False) -> dict[str, str]:
    """Return the driver -> themed (primary) color map."""
    return dict(DRIVER_THEME_DARK if dark else DRIVER_THEME)


def chrome(dark: bool = False) -> dict[str, str]:
    """Return the chart chrome/ink colors for the chosen mode."""
    return dict(DARK if dark else LIGHT)
