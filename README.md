# Pyarea

Pyarea is a Python package that reads data in the SSEC McIDAS AREA format.

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

The pixels are 1, 2 or 4 bytes wide. `area.data` holds all the bands, shape 
`(bands, lines, elements)`.

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
| `GMSX`   | `GMSXNav`    | GMS-5 and FY satellites                                             |
| `RECT`   | `RECTNav`    | rectilinear (lat/lon) grids                                         |
| `ABIN`   | `ABINNav`    | GOES-R ABI fixed grid (GOES-16 and later)                           |
| `GEOS`   | `GEOSNav`    | Himawari, COMS-1 (LRIT/HRIT layout)                                 |
| `LALO`   | `LALONav`    | Kalpana-1 and others: tables of latitude and longitude in the aux block |
| `MSAT`   | `MSATNav`    | Meteosat first generation (Meteosat-1 to 7)                         |
| `MSGT`   | `MSGTNav`    | Meteosat second generation (Meteosat-8 to 11)                       |

Any other nav type raises `NotImplementedError` and names the ones that are supported.
All of these have `to_latlon` and `to_lin_ele` for numpy arrays, which `get_lonlats` uses.

### Calibration

SSEC's McFETCH can give a calibrated area file (with the option "UNIT") like ADDE. 
The directory's calibration type (`area.cal_type`) tells what the file has 
(`TEMP`, `ALB`, `RAD`, `BRIT` or `RAW`). 

```python
area = AreaFile("mcfetch_band04.area")
print(area.cal_type, area.directory.calibration_scale_factor)   # TEMP 100
kelvin = area.read_calibrated("TEMP")      # a file already in TEMP: its pixels divided by the scaling, float32
lon, lat = area.get_lonlats()
```

For a file that already holds the requested unit, `read_calibrated` returns its pixels 
divided by the scaling,  and asking for another type raises a `ValueError`. 
`area.directory.check(file_size)` lists problems it finds in a directory (an unknown sensor, 
an unsupported pixel width, block offsets outside the file).

Area files that were ordered from NOAA CLASS only come in `RAW`, and needs to be calibrated: 

```python
temperature = area.read_calibrated("TEMP")   # GVAR imagery: 'RAW', 'RAD', 'ALB', 'TEMP' or 'BRIT'
reflectance = area.read_vis("unnormalized_reflectance")   # a pre-GVAR visible image
```

Calibrations currently implemented:

* `GVARCalibrator` (GOES-8 and later, GVAR), a vectorized port of VisAD's 
  `CalibratorGvar`.
* `GOESCalibrator` (SMS and GOES-1 to 7), new code that covers Mode A (VISR), 
  Mode AA (VAS, GOES-4 to 6) and Mode AAA (VAS, GOES-6 and 7). The visible channel 
  only has a relative reflectance; there is no calibration to albedo for these 
  satellites.

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
