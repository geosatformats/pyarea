"""Test whether the raw counts of a pre-GVAR VISSR visible (VIS) AREA file are linear or square-root encoded in albedo.

    python scripts/check_vis_encoding.py FILE.AREA_VIS [--dark-box L0 L1 E0 E1] [--bright-box L0 L1 E0 E1]

The counts of these files are 6-bit values shifted left by two (0 to 252). If they are linear in albedo, clear ocean
(albedo about 0.065) sits near count 16 and thick cloud (about 0.8) near 200. If they are square-root encoded, which
gives dark scenes more of the few levels, the same ocean sits near 64 and the cloud near 225.

The script takes the mean count of a dark and of a bright patch and reports

* the counts themselves, to compare with the two predictions above,
* the exponent ``n`` in ``albedo ~ count ** n`` that the two patches imply for the albedos you assume: n near 1 means
  linear, n near 2 means square-root encoded counts (their squares are linear in albedo).

The boxes are 0-based ``line_start line_stop element_start element_stop`` in the array (as in numpy slicing). Without
them the darkest and brightest 2% of the pixels on the Earth are used, which is only a rough guide: pick the boxes by
eye over open ocean and an unsaturated cloud top when you can. Only a file whose navigation works is masked to the
Earth; otherwise zero counts (space) are left out.
"""
import argparse

import numpy as np

from pyarea.file import AreaFile

FULL_SCALE = 252.0  # 6-bit counts shifted left by two


def earth_mask(area, counts):
    """Pixels on the Earth, from the file's navigation; those with a non-zero count if it cannot be used."""
    mask = counts > 0
    try:
        lon, lat = area.get_lonlats()
        mask &= np.isfinite(lat) & np.isfinite(lon)
    except Exception as error:  # a file without usable navigation: fall back on the counts
        print(f"(no usable navigation, {type(error).__name__}: only zero counts are left out)\n")
    return mask


def patch_mean(counts, mask, box=None, fraction=0.02, end="dark"):
    """Mean count of a box, or of the darkest / brightest ``fraction`` of the masked pixels."""
    if box is not None:
        l0, l1, e0, e1 = box
        patch = counts[l0:l1, e0:e1]
        patch = patch[mask[l0:l1, e0:e1]]
        if patch.size == 0:
            raise SystemExit(f"The box {box} has no pixels on the Earth.")
        return float(patch.mean())
    values = np.sort(counts[mask].astype(np.float64))
    n = max(1, int(values.size * fraction))
    return float(values[:n].mean() if end == "dark" else values[-n:].mean())


def implied_exponent(dark_count, bright_count, dark_albedo, bright_albedo):
    """n in ``albedo ~ count ** n``: 1 is linear counts, 2 is counts that are the square root of albedo."""
    return float(np.log(bright_albedo / dark_albedo) / np.log(bright_count / dark_count))


def predicted_counts(albedo, n, full_scale_albedo=1.0):
    """Count of a pixel of the given albedo if albedo = full_scale_albedo * (count / 252) ** n."""
    return FULL_SCALE * (albedo / full_scale_albedo) ** (1.0 / n)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("area_vis_file")
    parser.add_argument("--dark-box", nargs=4, type=int, metavar=("L0", "L1", "E0", "E1"))
    parser.add_argument("--bright-box", nargs=4, type=int, metavar=("L0", "L1", "E0", "E1"))
    parser.add_argument("--dark-albedo", type=float, default=0.065, help="assumed albedo of the dark patch (ocean)")
    parser.add_argument("--bright-albedo", type=float, default=0.8, help="assumed albedo of the bright patch (cloud)")
    args = parser.parse_args()

    area = AreaFile(args.area_vis_file)
    counts = area.read_vis()
    print(f"{args.area_vis_file}: {counts.shape[0]} lines x {counts.shape[1]} elements, max count {counts.max():.0f}")
    steps = np.unique(counts[counts > 0])
    print(f"distinct non-zero counts: {steps.size} (a 6-bit channel shifted by two has at most 63)\n")

    mask = earth_mask(area, counts)
    print(f"{mask.mean():.0%} of the pixels are used\n")
    pct = np.percentile(counts[mask], [1, 5, 25, 50, 75, 95, 99])
    print("count percentiles (1, 5, 25, 50, 75, 95, 99):", " ".join(f"{v:.0f}" for v in pct), "\n")

    dark = patch_mean(counts, mask, args.dark_box, end="dark")
    bright = patch_mean(counts, mask, args.bright_box, end="bright")
    how = lambda box, text: "box" if box else text  # noqa: E731
    print(f"dark patch   ({how(args.dark_box, 'darkest 2%')}): mean count {dark:6.1f}")
    print(f"bright patch ({how(args.bright_box, 'brightest 2%')}): mean count {bright:6.1f}\n")
    if dark <= 0 or bright <= dark:
        raise SystemExit("The dark patch has to be darker than the bright one, and above zero; choose other patches.")

    print(f"For albedos {args.dark_albedo} (dark) and {args.bright_albedo} (bright), with 252 = albedo 1.0:")
    for n, name in ((1.0, "linear counts (n=1)"), (2.0, "square-root counts (n=2)")):
        print(f"  {name:26s} expects counts {predicted_counts(args.dark_albedo, n):5.1f} and "
              f"{predicted_counts(args.bright_albedo, n):5.1f}")
    n = implied_exponent(dark, bright, args.dark_albedo, args.bright_albedo)
    print(f"\nObserved counts {dark:.1f} and {bright:.1f} imply albedo ~ count ** {n:.2f}")
    print("(near 1: linear; near 2: square-root encoded; the result depends on the albedos you assume, and a "
          "bright patch near 252 may be saturated)")


if __name__ == "__main__":
    main()
