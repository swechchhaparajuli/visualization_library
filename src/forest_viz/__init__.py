"""forest_viz: visualize Global Forest Watch tree-cover loss and its drivers.

    from forest_viz import data, plots, theme

    tcl = data.load_tree_cover_loss("global.xlsx")
    drivers = data.load_drivers("global.xlsx", primary=True)

    theme.apply_theme()
    plots.plot_loss_trend(tcl)
    plots.plot_driver_composition(drivers)
"""

from forest_viz import data, maps, palette, plots, theme
from forest_viz.maps import (
    load_country_geometry,
    plot_driver_pie,
    plot_drivers_over_time_3d,
    plot_top_countries_map,
)
from forest_viz.plots import (
    plot_driver_composition,
    plot_drivers_over_time,
    plot_loss_trend,
    plot_top_countries,
    plot_top_countries_drivers,
)
from forest_viz.theme import apply_theme

__all__ = [
    "data",
    "maps",
    "palette",
    "plots",
    "theme",
    "apply_theme",
    "plot_loss_trend",
    "plot_top_countries",
    "plot_top_countries_drivers",
    "plot_drivers_over_time",
    "plot_driver_composition",
    "plot_top_countries_map",
    "plot_driver_pie",
    "plot_drivers_over_time_3d",
    "load_country_geometry",
]
__version__ = "0.1.0"
