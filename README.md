# Pyarea

Pyarea is a Python package that can read data in the SSEC McIDAS AREA format.

A port of the edu.wisc.ssec.mcidas Java package, it mostly mimics its 
structure and logic with a few exceptions imposed by practical aspects 
of Python language. Some additional more efficient functions were added
(toLineEle2, toLatLon2) to perform navigation and work with numpy arrays.

It currently supports reading old GOES AREA files (GOESNav and GVARNav), 
although GMSXNav and RECTNav is also ported.

## Documentation

The AreaFile can be instantiated from a file or a binary string:



The AreaFile is lightweight, and features lazy initialization (reads the
data only upon request). The data returns a 3D numpy array; use squeeze
to get a 2D version.

## License

Copyright 2017 Ioan Ferencik, Solargis s.r.0 & Kin Largadas

This library is a Python port of edu.wisc.ssec.mcidas. Copyright 1998 - 2023 
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