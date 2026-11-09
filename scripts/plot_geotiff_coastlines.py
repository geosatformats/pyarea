"""Plot a GeoTIFF from area_to_geotiff.py with coastlines and a lat/lon grid, to check the navigation.

    python scripts/plot_geotiff_coastlines.py IN.tif OUT.png [VMIN VMAX]

VMIN and VMAX set the grey stretch (default: 2nd and 98th percentile). The coastlines are drawn by cartopy from
Natural Earth (downloaded on first use); the lat/lon grid works offline. Needs rasterio, cartopy and matplotlib.
"""
import sys

import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402


def geos_crs(proj4):
    """A cartopy Geostationary CRS from the proj4 string area_to_geotiff.py writes."""
    params = dict(item.lstrip("+").split("=", 1) for item in proj4.split() if "=" in item)
    a = float(params["a"])
    b = float(params["b"]) if "b" in params else a * (1.0 - 1.0 / float(params["rf"]))  # rasterio gives a and rf
    globe = ccrs.Globe(semimajor_axis=a, semiminor_axis=b, ellipse=None)
    return ccrs.Geostationary(central_longitude=float(params["lon_0"]), satellite_height=float(params["h"]),
                              globe=globe)


def plot(tif, out, vmin=None, vmax=None):
    with rasterio.open(tif) as src:
        data = src.read(1)
        t = src.transform
        extent = (t.c, t.c + t.a * src.width, t.f + t.e * src.height, t.f)  # left, right, bottom, top
        crs = geos_crs(src.crs.to_proj4() if "+proj" not in str(src.crs) else str(src.crs))
    data = np.ma.masked_equal(data, 0)
    lo, hi = np.percentile(data.compressed(), [2, 98])
    fig = plt.figure(figsize=(12, 12))
    ax = plt.axes(projection=crs)
    ax.imshow(data, origin="upper", extent=extent, transform=crs, cmap="gray_r",
              vmin=lo if vmin is None else vmin, vmax=hi if vmax is None else vmax)
    ax.gridlines(draw_labels=False, linewidth=0.4, color="cyan", alpha=0.7,
                 xlocs=np.arange(-180, 181, 10), ylocs=np.arange(-80, 81, 10))
    try:  # fetch first (cartopy downloads lazily, while drawing), so a download problem is caught here
        shapereader.natural_earth(resolution="50m", category="physical", name="coastline")
        ax.coastlines(resolution="50m", color="red", linewidth=0.7)
    except Exception as exc:  # no network: the grid still shows
        print(f"coastlines not drawn ({type(exc).__name__}: {exc})")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    plot(sys.argv[1], sys.argv[2], *(float(v) for v in sys.argv[3:5]))
