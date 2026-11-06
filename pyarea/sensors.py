"""The AREA sensor source numbers (word 3 of the directory) and what they stand for.

The numbers and names are those of ``AreaDirectory.java`` in VisAD/McIDAS-V; see also Appendix A of the
`SSEC AREA format <https://www.ssec.wisc.edu/mcidas/doc/misc_doc/area2.html>`_ document. The names are
parsed once into a platform and an instrument/band, so the rest of the package never has to look at text.
"""

import re
from typing import NamedTuple

SENSOR_NAMES = {
    0: "Derived Data",
    1: "Test Patterns",
    2: "Graphics",
    3: "Miscellaneous",
    4: "PDUS Meteosat visible",
    5: "PDUS Meteosat infrared",
    6: "PDUS Meteosat water vapor",
    7: "Radar",
    8: "Miscellaneous Aircraft Data",
    9: "Raw Meteosat",
    10: "Composite image",
    11: "Topography image",
    12: "GMS visible",
    13: "GMS infrared",
    14: "ATS 6 visible",
    15: "ATS 6 infrared",
    16: "SMS-1 visible",
    17: "SMS-1 infrared",
    18: "SMS-2 visible",
    19: "SMS-2 infrared",
    20: "GOES-1 visible",
    21: "GOES-1 infrared",
    22: "GOES-2 visible",
    23: "GOES-2 infrared",
    24: "GOES-3 visible",
    25: "GOES-3 infrared",
    26: "GOES-4 visible (VAS)",
    27: "GOES-4 infrared and water vapor (VAS)",
    28: "GOES-5 visible (VAS)",
    29: "GOES-5 infrared and water vapor (VAS)",
    30: "GOES-6 visible",
    31: "GOES-6 infrared",
    32: "GOES-7 visible, Block 1 supplemental data",
    33: "GOES-7 infrared",
    34: "FY-2B",
    35: "FY-2C",
    36: "FY-2D",
    37: "FY-2E",
    38: "FY-2F",
    39: "FY-2G",
    40: "FY-2H",
    41: "TIROS-N (POES)",
    42: "NOAA-6",
    43: "NOAA-7",
    44: "NOAA-8",
    45: "NOAA-9",
    46: "Venus",
    47: "Voyager 1",
    48: "Voyager 2",
    49: "Galileo",
    50: "Hubble space telescope",
    51: "MSG-1",
    52: "MSG-2",
    53: "MSG-3",
    54: "Meteosat-3",
    55: "Meteosat-4",
    56: "Meteosat-5",
    57: "Meteosat-6",
    58: "Meteosat-7",
    60: "NOAA-10",
    61: "NOAA-11",
    62: "NOAA-12",
    63: "NOAA-13",
    64: "NOAA-14",
    65: "NOAA-15",
    66: "NOAA-16",
    67: "NOAA-17",
    68: "NOAA-18",
    69: "NOAA-19",
    70: "GOES 8 imager",
    71: "GOES 8 sounder",
    72: "GOES 9 imager",
    73: "GOES 9 sounder",
    74: "GOES 10 imager",
    75: "GOES 10 sounder",
    76: "GOES 11 imager",
    77: "GOES 11 sounder",
    78: "GOES 12 imager",
    79: "GOES 12 sounder",
    80: "ERBE",
    82: "GMS-4",
    83: "GMS-5",
    84: "MTSAT-1R",
    85: "MTSAT-2",
    86: "Himawari-8",
    87: "DMSP F-8",
    88: "DMSP F-9",
    89: "DMSP F-10",
    90: "DMSP F-11",
    91: "DMSP F-12",
    92: "DMSP F-13",
    93: "DMSP F-14",
    94: "DMSP F-15",
    95: "FY-1B",
    96: "FY-1C",
    97: "FY-1D",
    101: "TERRA L1B",
    102: "TERRA CLD",
    103: "TERRA GEO",
    104: "TERRA-AER",
    106: "TERRA TOP",
    107: "TERRA ATM",
    108: "TERRA GUE",
    109: "TERRA RET",
    111: "AQUA L1B",
    112: "AQUA CLD",
    113: "AQUA GEO",
    114: "AQUA AER",
    116: "AQUA TOP",
    117: "AQUA ATM",
    118: "AQUA GUE",
    119: "AQUA RET",
    160: "TERRA NDVI",
    161: "TERRA CREF",
    170: "AQUA NDVI",
    171: "AQUA CREF",
    180: "GOES 13 imager",
    181: "GOES 13 sounder",
    182: "GOES 14 imager",
    183: "GOES 14 sounder",
    184: "GOES 15 imager",
    185: "GOES 15 sounder",
    186: "GOES 16 imager",
    187: "GOES 16 sounder",
    195: "DMSP F-16",
    196: "DMSP F-17",
    200: "AIRS L1B",
    210: "AMSR-E L1B",
    211: "AMSR-E RAIN",
    230: "Kalpana-1",
    231: "INSAT-3D Imager",
    232: "INSAT-3D Sounder",
    240: "MetOp-A",
    241: "MetOp-B",
    242: "MetOp-C",
    250: "COMS-1",
    261: "Landsat 1",
    262: "Landsat 2",
    263: "Landsat 3",
    264: "Landsat 4",
    265: "Landsat 5",
    266: "Landsat 6",
    267: "Landsat 7",
    268: "Landsat 8",
    300: "Suomi NPP-VIIRS",
    354: "METEOSAT-11",
    400: "South Pole Composites",
    401: "North Pole Composites",
    410: "Megha-Tropiques",
}

class Sensor(NamedTuple):
    """One sensor source.

    Attributes:
        name: The full name, e.g. ``GOES-3 infrared``.
        platform: The satellite or product, e.g. ``GOES-3``; ``GOES 8`` style names are written ``GOES-8``.
        instrument: What follows the platform in the name (``infrared``, ``imager``, ``sounder``...), or None.
        band: ``VIS`` or ``IR`` for visible and infrared (including water vapor) sensors, otherwise None.
    """

    name: str
    platform: str
    instrument: str | None
    band: str | None


_NAME = re.compile(r"^(?P<platform>.+?) (?P<instrument>visible|infrared|water vapor|imager|sounder)\b(?P<rest>.*)$",
                   re.IGNORECASE)


def _parse(name):
    match = _NAME.match(name)
    if match is None:
        return Sensor(name, name, None, None)
    platform = re.sub(r"^GOES (\d+)$", r"GOES-\1", match["platform"])
    instrument = (match["instrument"] + match["rest"]).strip()
    kind = match["instrument"].lower()
    band = "VIS" if kind == "visible" else "IR" if kind in ("infrared", "water vapor") else None
    return Sensor(name, platform, instrument, band)


SENSORS = {number: _parse(name) for number, name in SENSOR_NAMES.items()}


def get_sensor(number):
    """The :class:`Sensor` for a sensor source number, or None if the number is not in the table."""
    return SENSORS.get(number)
