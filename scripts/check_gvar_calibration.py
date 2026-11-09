"""Calibrate a single-band GVAR AREA file and print values to compare with McIDAS-V.

    python scripts/check_gvar_calibration.py FILE.area [LINE ELEM] [LINE ELEM] ...
    python scripts/check_gvar_calibration.py FILE.area --latlon LAT LON [LAT LON] ...   (east longitude positive)

LINE and ELEM are 0-based indices into the array (row, column); McIDAS-V shows image coordinates, so convert
with  row = (image_line - directory line_ul) / line_res  (and the same for columns) or just pick pixels by index
and read the same pixels in McIDAS-V's probe.
"""
import sys

import numpy as np

from pyarea.calibration import calibrated_unit, cal_type_from_name
from pyarea.file import AreaFile


def main(argv):
    area = AreaFile(argv[0])
    d = area.directory
    print(f"sensor {d.sensor_source_number} ({d.sensor_name}), source type {area.source_type}, "
          f"bands {d.bands}, {area.lines} lines x {area.elements} elements, byte order {d.byte_order!r}")
    rest = argv[1:]
    pixels = []
    if rest and rest[0] == "--latlon":
        nums = [float(x) for x in rest[1:]]
        for lat, lon in zip(nums[0::2], nums[1::2]):
            elem, line = area.navigation.toLinEle([[lat], [lon]])  # image coordinates, [[elements], [lines]]
            # the navigation already accounts for the file's start and resolution; its line/element are 1-based
            row = float(line[0]) - 1
            col = float(elem[0]) - 1
            print(f"lat {lat} lon {lon} -> line {float(line[0]):.1f} elem {float(elem[0]):.1f} "
                  f"-> row {row:.1f} col {col:.1f} (array is {area.lines} x {area.elements})")
            pixels.append((int(round(row)), int(round(col))))
    else:
        pixels = [(int(a), int(b)) for a, b in zip(rest[0::2], rest[1::2])]
    if not pixels:
        pixels = [(area.lines // 2, area.elements // 2)]
    raw = area.read_data()
    band = d.bands[0]
    visible = band == (1 if d.sensor_source_number % 2 == 0 else 19)
    kinds = ("ALB", "BRIT") if visible else ("RAD", "TEMP", "BRIT")
    out = {k: area.read_calibrated(k) for k in kinds}
    for k, arr in out.items():
        unit = calibrated_unit(cal_type_from_name(k)) or ""
        print(f"{k:5s} {unit:>14s}: min {np.nanmin(arr):.4f}  max {np.nanmax(arr):.4f}  "
              f"mean {np.nanmean(arr):.4f}  NaN {np.isnan(arr).sum()}")
    print("\nrow  col   raw  " + "  ".join(f"{k:>10s}" for k in kinds))
    for r, c in pixels:
        print(f"{r:4d} {c:4d} {raw[r, c]:5d}  " + "  ".join(f"{out[k][r, c]:10.4f}" for k in kinds))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
