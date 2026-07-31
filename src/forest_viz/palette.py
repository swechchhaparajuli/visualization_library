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

def _hex(c):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4))


def _to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(max(0, min(255, int(v))) for v in rgb))


def _lighten(c, t):
    r, g, b = _hex(c)
    return _to_hex((r + (255 - r) * t, g + (255 - g) * t, b + (255 - b) * t))


def _darken(c, f):
    r, g, b = _hex(c)
    return _to_hex((r * f, g * f, b * f))


# User-supplied categorical palette (warm, muted). Assigned to drivers in
# DRIVER_ORDER; this is the identity color used across every graph.
_PALETTE = ["#a9a06c", "#474a29", "#bd6b45", "#c99c81", "#7397ac", "#47799a", "#aebdc2"]
_NEUTRAL = "#c7c2b3"  # the spare greige swatch

# Full-strength "theme" colors (primary driver / most charts).
_THEME_LIGHT = list(_PALETTE)
_THEME_DARK = list(_PALETTE)
# Muted (secondary) tints of the same palette, for the map's non-primary zones.
_SLOTS_LIGHT = [_lighten(c, 0.55) for c in _PALETTE]
_SLOTS_DARK = [_darken(c, 0.7) for c in _PALETTE]

DRIVER_COLORS = dict(zip(DRIVER_ORDER, _SLOTS_LIGHT))
DRIVER_COLORS_DARK = dict(zip(DRIVER_ORDER, _SLOTS_DARK))

DRIVER_THEME = dict(zip(DRIVER_ORDER, _THEME_LIGHT))
DRIVER_THEME_DARK = dict(zip(DRIVER_ORDER, _THEME_DARK))

# Single-hue accent for one-series charts (steel blue from the palette).
ACCENT = "#47799a"
ACCENT_DARK = _lighten("#47799a", 0.2)

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
