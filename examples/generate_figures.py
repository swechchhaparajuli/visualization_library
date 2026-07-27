"""Generate the forest_viz example figures from the GFW workbook.

Usage:
    python examples/generate_figures.py path/to/global.xlsx

Writes PNGs to reports/figures/ and small tidy CSVs to data/processed/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

from forest_viz import data, maps, plots, theme

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"
DATA_DIR = ROOT / "data" / "processed"
GEOJSON = ROOT / "data" / "geo" / "ne_110m_admin_0_countries.geojson"


def main(workbook: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tcl = data.load_tree_cover_loss(workbook)
    drivers = data.load_drivers(workbook, primary=True)
    all_drivers = data.load_drivers(workbook, primary=False)  # for the total-loss breakdown

    # Vendor compact tidy CSVs so the charts are reproducible without the 18MB xlsx.
    data.loss_by_year(tcl).to_csv(DATA_DIR / "global_loss_by_year.csv", index=False)
    data.loss_by_country(tcl).head(30).to_csv(DATA_DIR / "loss_by_country_top30.csv", index=False)
    drivers.groupby(["driver", "year"], as_index=False)["tc_loss_ha"].sum().to_csv(
        DATA_DIR / "global_primary_drivers_by_year.csv", index=False
    )
    data.top_countries_drivers(all_drivers, n=15, year_range=(2001, 2024)).to_csv(
        DATA_DIR / "top15_countries_by_driver.csv"
    )

    theme.apply_theme()
    figs = {
        "tree_cover_loss_trend.png": plots.plot_loss_trend(tcl),
        "top_countries_loss.png": plots.plot_top_countries(tcl, n=15, year_range=(2001, 2024)),
        "top_countries_by_driver.png": plots.plot_top_countries_drivers(
            all_drivers, n=15, year_range=(2001, 2024)
        ),
        "primary_drivers_over_time.png": maps.plot_drivers_over_time_3d(drivers),
        "primary_driver_pie.png": maps.plot_driver_pie(drivers, year_range=(2002, 2024)),
    }
    if GEOJSON.exists():
        figs["top_countries_map.png"] = maps.plot_top_countries_map(
            all_drivers, str(GEOJSON), n=15, year_range=(2001, 2024)
        )
    else:
        print("skip map: missing", GEOJSON)
    for name, ax in figs.items():
        ax.figure.tight_layout()
        ax.figure.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
        plt.close(ax.figure)
        print("wrote", FIG_DIR / name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/generate_figures.py path/to/global.xlsx")
    main(sys.argv[1])
