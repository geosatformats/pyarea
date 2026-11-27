# Pyarea

Pyarea is a Python package that reads data in the SSEC McIDAS AREA format: the
files of McIDAS-X, the images that SSEC's McFETCH and ADDE servers send, and the
pre-GVAR GOES archive.

The navigation is a port of the `edu.wisc.ssec.mcidas` Java package: 
it mostly mimics its structure and logic, with a few exceptions 
imposed by practical aspects of the Python language. Some additional, 
more efficient methods were added that work on numpy arrays 
(`to_latlon`, `to_lin_ele`).

## Installation

```
pip install .
```

Python 3.10 or later; numpy, numba and pyproj are needed.

## Usage

An `AreaFile` can be made from a file name or from the bytes of a file. 
It is lightweight and initializes lazily: the data are only read on request, 
and come back as a 3D numpy array (use `squeeze` for a 2D one).

```python
from pyarea import AreaFile

area = AreaFile("path/to/file.AREA")
print(area)                     # the directory
data = area.read_data()         # the pixels of a single band file, shape (lines, elements)
```

The pixels are 1, 2 or 4 bytes wide (4-byte data are signed, as ADDE sends radiances). 
`area.data` holds all the bands, shape `(bands, lines, elements)`.

### Navigation

`area.navigation` gives the navigation object of the file's nav block 
(`None` if the file has none). Lines and elements are those of the AREA 
file, counted from 0, and the navigation applies the file's start and resolution 
itself:

```python
nav = area.navigation
lat, lon = nav.to_latlon(line, elem)      # numpy arrays of any shape; NaN where the Earth is not seen
line, elem = nav.to_lin_ele(lat, lon)     # east longitudes are positive
```

The Java style methods (`toLatLon`, `toLinEle`, taking lists) are there as well. 
`area.get_lonlats()` earth-locates the whole image with the file's own navigation, 
whatever its type (or only the given `lines=` and `elems=`), and returns `(lon, lat)`:

```python
import numpy as np

lon, lat = area.get_lonlats()                       # arrays of shape (lines, elements), NaN off the Earth
lon, lat = area.get_lonlats(lines=np.arange(0, area.lines, 4), elems=np.arange(0, area.elements, 4))
```

Currently implemented navigations:

| Nav type | Class        | Satellites                                                          |
|----------|--------------|---------------------------------------------------------------------|
| `GVAR`   | `GVARNav`    | GOES-8 and later (GVAR)                                             |
| `GOES`   | `GOESNav`    | SMS and GOES-1 to 7 (pre-GVAR)                                      |
| `GMSX`   | `GMSXNav`    | GMS-5 stretched VISSR, and the Fengyun satellites that use it       |
| `RECT`   | `RECTNav`    | rectilinear (lat/lon) grids                                         |
| `ABIN`   | `ABINNav`    | GOES-R ABI fixed grid (GOES-16 and later)                           |
| `GEOS`   | `GEOSNav`    | Himawari, COMS-1 (LRIT/HRIT layout)                                 |
| `LALO`   | `LALONav`    | Kalpana-1 and others: tables of latitude and longitude in the aux block |
| `MSAT`   | `MSATNav`    | Meteosat first generation (Meteosat-1 to 7)                         |
| `MSGT`   | `MSGTNav`    | Meteosat second generation (Meteosat-8 to 11)                       |

Any other nav type raises `NotImplementedError` and names the ones that are supported.
All of these have `to_latlon` and `to_lin_ele` for numpy arrays, which `get_lonlats` uses.
The reverse direction of `LALONav` follows VisAD's `Gridded2DSet.valueToGrid`, and that of `MSATNav` 
only inverts `to_latlon` for files whose third nav offset is 1250 (as in the Java code).

### Files from McFETCH

McFETCH, like ADDE, calibrates on the server, so the pixels of its files are already in the units 
asked for: the directory's calibration type (`area.cal_type`) says which (`TEMP`, `ALB`, `RAD`, `BRIT` or `RAW`). 
`TEMP`, `ALB` and `RAD` are stored as integers times the directory's scaling (`area.directory.calibration_scale_factor`):
a `TEMP` file with a scaling of 100 holds 29315 for 293.15 K. 

```python
area = AreaFile("mcfetch_band04.area")
print(area.cal_type, area.directory.calibration_scale_factor)   # TEMP 100
kelvin = area.read_calibrated("TEMP")      # a file already in TEMP: its pixels divided by the scaling, float32
lon, lat = area.get_lonlats()
```

For a file that already holds the requested type (any source type, GVAR included), `read_calibrated` returns its
pixels divided by the scaling, and asking for another type raises a `ValueError`. Only raw GVAR files go through
`GVARCalibrator`. `read_ir_temperature_a` likewise returns the stored temperatures of a TEMP file.
`area.directory.check(file_size)` lists problems it finds in a directory (an unknown sensor, an unsupported 
pixel width, block offsets outside the file).

### Calibration

```python
temperature = area.read_calibrated("TEMP")   # GVAR imagery: 'RAW', 'RAD', 'ALB', 'TEMP' or 'BRIT'
reflectance = area.read_vis("unnormalized_reflectance")   # a pre-GVAR visible image
```

The calibration is **not** a port of the Java package, and comes in two parts:

* `GVARCalibrator` (GOES-8 and later, GVAR) is a vectorized port of VisAD's 
  `CalibratorGvar`.
* `GOESCalibrator` (SMS and GOES-1 to 7) is new code, as the Java package does 
  not read these data. It covers Mode A (VISR), Mode AA (VAS, GOES-4 to 6) and 
  Mode AAA (VAS, GOES-6 and 7). The visible channel only has a relative reflectance, 
  there is no calibration to albedo for these satellites.

## Scripts

The `scripts` folder has command line tools to check a navigation by looking at coastlines on the image. They 
come in pairs, because the plotting needs matplotlib and cartopy, which can live in a different environment 
from pyarea (and numba). Run them from the root of the repository, so that `pyarea` is found (or install it, 
or set `PYTHONPATH=.`). The coastlines are downloaded by cartopy on first use (Natural Earth).

### Any AREA file with a navigation, McFETCH files in particular

1. In the pyarea environment, `mcfetch_to_npz.py` earth-locates the file and saves the image with its latitudes 
   and longitudes to an `.npz`:

   ```
   python scripts/mcfetch_to_npz.py FILE.area [OUT.npz] [--step N] [--calibrate TEMP|ALB|RAD|BRIT]
   ```

   It prints the directory, the navigation type and words, and the latitude and longitude of the corners and the 
   centre of the image, which is the first thing to look at when a navigation seems wrong. `OUT.npz` defaults to 
   `FILE_lonlat.npz`. A file from McFETCH is saved as it is (TEMP in kelvin, and so on); `--calibrate` is only for 
   raw data, such as a GVAR file. `--step N` keeps every Nth line and element, by default 1 unless the image is 
   very large.

2. In the environment with matplotlib and cartopy (pyarea is not needed), `plot_lonlat_coastlines.py` draws it, 
   in grey without enhancement, with a lat/lon grid, coastlines and a colour bar:

   ```
   python scripts/plot_lonlat_coastlines.py FILE_lonlat.npz OUT.png [--celsius] [--light] [--vmin V --vmax V]
   ```

   `--celsius` shows kelvin as degrees Celsius, `--light` gives a white background instead of a black one, and 
   `--vmin`/`--vmax` set the stretch (the 2nd and 98th percentiles by default).

   To look at a small area, such as one stretch of coast at the resolution of the image, save the `.npz` with `--step 1` and crop:

   ```
   python scripts/plot_lonlat_coastlines.py FILE_lonlat.npz crop.png --extent -66.6 -66.0 17.9 18.2 --res 10m
   ```

   `--extent W E S N` is in degrees east of Greenwich, and `--res` is the coastline resolution (`110m`, `50m` or `10m`).

If the coastlines do not match the coast: a constant shift of the same size everywhere is an offset convention 
(a pixel or half a pixel), while a shift that grows towards the edge of the image is a scale factor or the 
longitude of the subpoint. The corner report of the first step helps to tell which.

### Calibrated GOES (GVAR) files to a GeoTIFF

* `area_to_geotiff.py FILE.area OUT.tif [UNITS] [RESOLUTION_M] [--proj4]` (pyarea environment, needs rasterio) calibrates a 
  single-band GVAR file and writes it on a geostationary grid. 
* `plot_geotiff_coastlines.py IN.tif OUT.png [VMIN VMAX]` (needs rasterio, cartopy and matplotlib) draws that GeoTIFF with 
  coastlines.
* `check_vis_encoding.py FILE.AREA_VIS` tells whether the counts of a pre-GVAR visible image are linear or square-root encoded.

Each script has its usage in the docstring at the top.

## License

Copyright 2017 Ioan Ferencik, Solargis s.r.0 & Kin Largadas

This library is a Python port of `edu.wisc.ssec.mcidas`. Copyright 1998 - 2023
by Tom Whittaker, Tommy Jasmin, Tom Rink, Don Murray, James Kelly, Bill Hibbard,
Dave Glowacki, Curtis Rueden and others.

The original work is licensed under the GNU Library General Public License,
version 2.0 or later. As permitted in the "or later" clause of the license,
this port and all its derived modifications are distributed under the terms of
the GNU Lesser General Public License, version 3.0 or later.

This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License as published by the Free
Software Foundation; either version 3.0 of the License, or (at your option)
any later version. You may obtain a copy of the License at:

    https://www.gnu.org/licenses/lgpl-3.0.html

This library is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
