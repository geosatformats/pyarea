"""MSATNav, the navigation of Meteosat first generation AREA files (nav type MSAT), against MSATnav.java."""
import struct

import numpy as np
import pytest

from pyarea.file import AreaFile
from pyarea.nav import AreaNav, MSATNav
from tests.fake_area_files import GOES7_IR, FakeArea, mode_a_data, write_area_file

R, NOMORB, FLATTENING = 6378.155, 42164.0, 1.0 / 297.0
RP = R / (1.0 + FLATTENING)
DELTA = 18.0 / 2500.0
SUBLON_EAST = 63.0  # Meteosat-5 at 63 E; the file has it as -63 degrees, west positive: -630000 packed
CENTRE_LINE, CENTRE_ELEM = 2500.0, 2501.0  # the pixel where both scan angles are 0 for ioff = (0, 0, 1250)


def msat_block(order=">", ioff=(0, 0, 1250), sublon_packed=-630000):
    words = [struct.unpack(f"{order}i", b"MSAT")[0], 0, 0, *ioff, sublon_packed]
    return tuple(words + [0] * 633)


def make_nav(order=">", **kwargs):
    nav = AreaNav(navblock=msat_block(order, **kwargs), byte_order=order, int_format="i")
    nav.setImageStart(0, 0)
    nav.setRes(1, 1)
    nav.setMag(1, 1)
    nav.setStart(0, 0)
    return nav


def ray_latlon(line, elem, ioff=(0, 0, 1250)):
    """Latitude and longitude of the pixel from the geometry alone: the line of sight from the satellite on the
    equator meets the ellipsoid (flattening 1/297) at the near root. The pixel is turned into the scan angles."""
    x = np.radians((1250.5 - elem / 2.0) * DELTA)  # east-west
    y = np.radians((ioff[2] - (line / 2.0 + ioff[1] - ioff[0])) * DELTA)  # north-south
    d = np.array([-np.cos(x), -np.sin(x), np.tan(y)])  # from the satellite (NOMORB, 0, 0) towards the Earth
    s = np.array([NOMORB, 0.0, 0.0])
    # (X^2 + Y^2) / R^2 + Z^2 / RP^2 = 1 along s + t d
    scale = np.array([1 / R ** 2, 1 / R ** 2, 1 / RP ** 2])
    a, b, c = (d * d * scale).sum(), 2 * (s * d * scale).sum(), (s * s * scale).sum() - 1
    disc = b * b - 4 * a * c
    if disc < 0:
        return np.nan, np.nan
    point = s + d * (-b - np.sqrt(disc)) / (2 * a)
    lat = np.degrees(np.arctan2(point[2] * R ** 2 / RP ** 2, np.hypot(point[0], point[1])))
    return lat, np.degrees(np.arctan2(point[1], point[0])) + SUBLON_EAST


def test_the_block_is_read_in_either_byte_order():
    for order in (">", "<"):
        nav = make_nav(order, ioff=(3, 7, 1250))
        assert type(nav) is MSATNav
        assert nav.ioff == (3, 7, 1250)
        assert nav.sublon == pytest.approx(-63.0)


def test_the_subpoint_is_a_packed_integer():
    assert make_nav(sublon_packed=-123000).sublon == pytest.approx(-12.5)  # 12 degrees 30 minutes
    assert make_nav(sublon_packed=753045).sublon == pytest.approx(75 + 30 / 60 + 45 / 3600)


def test_short_blocks_and_other_nav_types_are_rejected():
    with pytest.raises(ValueError, match="7 are needed"):
        AreaNav(navblock=msat_block()[:5], byte_order=">", int_format="i")
    nav = MSATNav.__new__(MSATNav, navblock=msat_block(), byte_order=">", int_format="i")
    nav.navtype = "GVAR"
    with pytest.raises(ValueError, match="Invalid navigation type"):
        nav.__init__(navblock=msat_block())


def test_the_centre_of_the_grid_is_the_subpoint():
    lat, lon = make_nav().to_latlon(CENTRE_LINE, CENTRE_ELEM)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(SUBLON_EAST, abs=1e-9)


def test_to_latlon_is_the_line_of_sight_meeting_the_ellipsoid():
    lines, elems = np.meshgrid(np.linspace(700, 4300, 7), np.linspace(700, 4300, 7), indexing="ij")
    lat, lon = make_nav().to_latlon(lines, elems)
    ok = np.isfinite(lat)
    assert ok.sum() > 30
    reference = np.array([[ray_latlon(a, b) for a, b in zip(row_l, row_e)] for row_l, row_e in zip(lines, elems)])
    np.testing.assert_allclose(lat[ok], reference[..., 0][ok], atol=1e-8)
    np.testing.assert_allclose(lon[ok], reference[..., 1][ok], atol=1e-8)


def test_lines_run_from_north_to_south_and_elements_from_west_to_east():
    nav = make_nav()
    lat, _ = nav.to_latlon(np.array([1500.0, 3500.0]), np.array([CENTRE_ELEM, CENTRE_ELEM]))
    assert lat[0] > lat[1]
    _, lon = nav.to_latlon(np.array([CENTRE_LINE, CENTRE_LINE]), np.array([1500.0, 3500.0]))
    assert lon[0] < lon[1]


def test_the_offsets_move_the_grid():
    """ioff[0] and ioff[1] shift the lines by 2 pixels each, in opposite directions."""
    lines, elems = np.array([1800.0, 2600.0]), np.array([2200.0, 3000.0])
    base = make_nav().to_latlon(lines, elems)
    first = make_nav(ioff=(1, 0, 1250)).to_latlon(lines + 2, elems)
    second = make_nav(ioff=(0, 1, 1250)).to_latlon(lines - 2, elems)
    for shifted in (first, second):
        np.testing.assert_allclose(shifted[0], base[0], atol=1e-12)
        np.testing.assert_allclose(shifted[1], base[1], atol=1e-12)


def test_the_round_trip_gives_back_the_pixel():
    nav = make_nav(ioff=(3, 7, 1250))
    lines, elems = np.meshgrid(np.linspace(600, 4400, 9), np.linspace(600, 4400, 9), indexing="ij")
    lat, lon = nav.to_latlon(lines, elems)
    back_lines, back_elems = nav.to_lin_ele(lat, lon)
    ok = np.isfinite(lat)
    assert ok.sum() > 40
    np.testing.assert_allclose(back_lines[ok], lines[ok], atol=1e-6)
    np.testing.assert_allclose(back_elems[ok], elems[ok], atol=1e-6)


def test_the_reverse_direction_assumes_the_third_offset_is_1250():
    """Java's toLinEle has 5000 lines hard-coded, which is the third offset only when that is 1250."""
    lat, lon = make_nav(ioff=(0, 0, 1250)).to_latlon(2000.0, 2200.0)
    nav = make_nav(ioff=(0, 0, 1300))
    other_lat, other_lon = nav.to_latlon(2000.0, 2200.0)
    line, _ = nav.to_lin_ele(other_lat, other_lon)
    assert line == pytest.approx(2000.0 + 4 * (1250 - 1300), abs=1e-6)  # off by 4 lines for each step of the offset


def test_space_is_nan_and_the_far_side_has_no_pixel():
    nav = make_nav()
    lat, lon = nav.to_latlon(np.array([0.0, 5000.0, np.nan]), np.array([0.0, 5000.0, 100.0]))
    assert np.isnan(lat).all() and np.isnan(lon).all()
    line, elem = nav.to_lin_ele(0.0, SUBLON_EAST + 180.0)
    assert np.isnan(line) and np.isnan(elem)
    assert np.isnan(nav.to_lin_ele(np.nan, 10.0)[0])


def test_the_area_to_image_conversion_is_applied():
    nav = make_nav()
    nav.setRes(2, 2)
    nav.setImageStart(1000, 1200)
    assert nav.to_latlon(10.0, 20.0) == make_nav().to_latlon(1020.0, 1240.0)


def test_the_list_interface_has_elements_first_like_visad():
    nav = make_nav()
    lat, lon = nav.to_latlon(np.array([1800.0]), np.array([2200.0]))
    out = nav.toLatLon([[2200.0], [1800.0]])  # [[elements], [lines]]
    assert out[0][0] == lat[0] and out[1][0] == lon[0]
    line_elem = nav.toLinEle(out)
    assert line_elem[nav.indexLine][0] == pytest.approx(1800.0, abs=1e-6)
    assert line_elem[nav.indexEle][0] == pytest.approx(2200.0, abs=1e-6)
    assert nav.toLatLon2(np.array([1800.0]), np.array([2200.0]))[0][0] == lat[0]


@pytest.mark.parametrize("endian", [">", "<"])
def test_a_file_is_located_through_get_lonlats(tmp_path, endian):
    words = np.array(msat_block(">", ioff=(0, 0, 1250)))
    spec = FakeArea(GOES7_IR, mode_a_data([100]), source_type="MSAT", ul_line=2496, ul_element=2494, line_res=1,
                    elem_res=1, nav_words=words, endian=endian)
    lon, lat = AreaFile(write_area_file(tmp_path / "msat.area", spec)).get_lonlats()
    assert (lat[4, 7], lon[4, 7]) == pytest.approx((0.0, SUBLON_EAST), abs=1e-9)  # image line 2500, element 2501
    assert lat[0, 7] > lat[-1, 7]
