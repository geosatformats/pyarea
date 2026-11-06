"""Pre-GVAR (SMS/GOES-1..7 VISSR/VAS) McIDAS AREA support.

Requires ``numba`` (``pip install pyarea[pregvar]``).
"""
from pyarea.pregvar.area_file import PregvarAreaFile
from pyarea.nav import GOESNav

__all__ = ["GOESNav", "PregvarAreaFile"]
