# Pyarea

Pyarea is a Python package to read data from the GOES VISSR, VAS and/or
GVAR instruments onboard GOES satellites in McIDAS AREA format.

A python port of the edu.wisc.ssec.mcidas Java package, it mostly mimics
its structure and logic with a few exceptions imposed by practical aspects
of Python language. Some additional more efficient functions were added
(toLineEle2, toLatLon2) to perform navigation and work with numpy arrays.

## Documentation

The AreaFile can be instantiated from a file or a binary string:



The AreaFile is lightweight, and features lazy initialization (reads the
data only upon request). The data returns a 3D numpy array; use squeeze
To get a 2D version.

## License

Copyright 2017 Ioan Ferencik, Solargis s.r.0 & Kin Largadas

A derivative of edu.wisc.ssec.mcidas. Copyright 1998 - 2023 by Tom Whittaker, 
Tommy Jasmin, Tom Rink, Don Murray, James Kelly, Bill Hibbard, Dave Glowacki, 
Curtis Rueden and others. Licensed under the GNU Library General Public License.

This library is free software; you can redistribute it and/or modify it under 
the terms of the GNU Lesser General Public License as published by the Free 
Software Foundation; either version 3 of the License, or (at your option) any
later version. You may obtain a copy of the License at:

    https://www.gnu.org/licenses/lgpl-3.0.en.html

This library is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.