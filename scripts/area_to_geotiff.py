"""Calibrate a single-band GVAR AREA file and write it as a GeoTIFF on a geostationary grid.

    python scripts/area_to_geotiff.py FILE.area OUT.tif [UNITS] [RESOLUTION_M] [--proj4]

UNITS is TEMP (default, K), RAD, ALB (visible, %), BRIT or RAW. RESOLUTION_M is the output pixel size in metres
(default 2000). Pixels outside the image are 0 and marked as nodata. Needs rasterio (which Satpy installs).

The geostationary output grid is navigated with the HRIT navigation (``pyarea.navigation.hrit_navigation``) by
default. ``--proj4`` uses ``pyarea.navigation.proj4_navigation`` (PROJ) instead, so the two can be compared on a file.
"""
import sys

import numpy as np
import rasterio
from rasterio.transform import Affine

from pyarea.file import AreaFile
from pyarea.navigation.area_navigation import reproject_area2geos


class _Calibrated:
    """What reproject_area2geos needs from a file: .navigation, .data and .close()."""

    def __init__(self, navigation, data):
        self.navigation = navigation
        self.data = data[None]

    def close(self):  # the helper closes what it is given; the AreaFile stays open
        pass


def area_to_geotiff(area, out_path, units="TEMP", resolution=2000, use_proj_navigation=False):
    """Write ``area.read_calibrated(units)`` to ``out_path`` as a GeoTIFF in a geostationary projection.

    ``use_proj_navigation`` picks PROJ (proj4_navigation) rather than the HRIT formulas for the output grid.
    """
    data = area.read_calibrated(units)
    ssp = area.navigation.subpoint[1]
    grid, (geotransform, proj4) = reproject_area2geos(
        _Calibrated(area.navigation, data), out_resolution=resolution, out_ssp=ssp,
        use_proj_navigation=use_proj_navigation, return_gdalinfo=True)
    # a GDAL GeoTransform is (x0, dx, 0, y0, 0, dy); Affine.from_gdal takes it as it is
    with rasterio.open(out_path, "w", driver="GTiff", height=grid.shape[0], width=grid.shape[1], count=1,
                       dtype="float32", crs=proj4, transform=Affine.from_gdal(*geotransform), nodata=0,
                       compress="deflate") as dst:
        dst.write(grid.astype(np.float32), 1)
    return grid, geotransform, proj4


def main(argv):
    use_proj_navigation = "--proj4" in argv
    argv = [a for a in argv if a != "--proj4"]
    area = AreaFile(argv[0])
    units = argv[2] if len(argv) > 2 else "TEMP"
    resolution = int(argv[3]) if len(argv) > 3 else 2000
    grid, geotransform, proj4 = area_to_geotiff(area, argv[1], units, resolution, use_proj_navigation)
    filled = grid != 0
    print(f"wrote {argv[1]} ({'proj4' if use_proj_navigation else 'HRIT'} navigation): "
          f"{grid.shape[1]} x {grid.shape[0]} pixels at {resolution} m, {filled.mean():.1%} filled")
    print(f"values {np.nanmin(grid[filled]):.2f} to {np.nanmax(grid[filled]):.2f}")
    print(f"GeoTransform {geotransform}\n{proj4}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
