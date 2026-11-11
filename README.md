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
data = area.read_data()         # counts, shape (bands, lines, elements)
```

### Navigation

`area.navigation` gives the navigation object of the file's nav block 
(`None` if the file has none). Lines and elements are those of the AREA 
file (1-based), and the navigation applies the file's start and resolution 
itself:

```python
nav = area.navigation
lat, lon = nav.to_latlon(line, elem)      # numpy arrays of any shape; NaN where the Earth is not seen
line, elem = nav.to_lin_ele(lat, lon)     # east longitudes are positive
```

The Java style methods (`toLatLon`, `toLinEle`, taking lists) are there as well. 
`area.get_lonlats()` earth-locates the whole image of a pre-GVAR GOES file.

Currently implemented navigations:

| Nav type | Class        | Satellites                                      |
|----------|--------------|-------------------------------------------------|
| `GVAR`   | `GVARNav`    | GOES-8 and later                                |
| `GOES`   | `GOESNav`    | SMS and GOES-1 to 7 (pre-GVAR)                  |
| `GMSX`   | `GMSXNav`    | GMS-5 stretched VISSR                           |
| `RECT`   | `RECTNav`    | rectilinear (lat/lon) grids                     |


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
