"""MSGTNav, the navigation of Meteosat second generation AREA files (nav type MSGT), against MSGTnav.java."""
import struct

import numpy as np
import pytest

from pyarea.file import AreaFile
from pyarea.nav import AreaNav, MSGTNav
from tests.fake_area_files import GOES7_IR, FakeArea, mode_a_data, write_area_file

R, NOMORB, FLATTENING = 6378.155, 42164.0, 1.0 / 297.0
RP = R / (1.0 + FLATTENING)
# SEVIRI 3 km: HRIT COFF = LOFF = 1856 and CFAC = LFAC = 13642337 (units of 2**-16 degree), subpoint 41.5 E
LOFF = COFF = 18560
FACTOR = round(1e6 * 13642337 / 2 ** 16)  # the words hold 1e6 / (degrees per step)
SUBLON_EAST = 41.5
SUBLON_WORD = -415000  # degrees x 10000, west positive
CENTRE = 5571.0  # image line and element where both scan angles are 0: (11139 - 3 * 1856)


def msgt_block(order=">", loff=LOFF, coff=COFF, lfac=FACTOR, cfac=FACTOR, sublon=SUBLON_WORD):
    words = [struct.unpack(f"{order}i", b"MSGT")[0], loff, coff, lfac, cfac, sublon]
    return tuple(words + [0] * 634)


def make_nav(order=">", **kwargs):
    nav = AreaNav(navblock=msgt_block(order, **kwargs), byte_order=order, int_format="i")
    nav.setImageStart(0, 0)
    nav.setRes(1, 1)
    nav.setMag(1, 1)
    nav.setStart(0, 0)
    return nav


def ray_latlon(line, elem):
    """Latitude and longitude of the pixel from the geometry alone: the line of sight from the satellite on the
    equator meets the ellipsoid at the near root. The 11136 grid is 3 pixels per SEVIRI pixel, counted from the
    north-west corner (lines go south, elements go east)."""
    step = 65536.0 / 13642337.0  # degrees per SEVIRI pixel
    x = np.radians(((11139.0 - elem) / 3.0 - 1856.0) * step)  # to the west
    y = np.radians(((11139.0 - line) / 3.0 - 1856.0) * step)  # to the north
    d = np.array([-np.cos(x), -np.sin(x), np.tan(y)])  # from the satellite at (NOMORB, 0, 0)
    s = np.array([NOMORB, 0.0, 0.0])
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
        nav = make_nav(order)
        assert type(nav) is MSGTNav
        assert (nav.loff, nav.coff, nav.lfac, nav.cfac) == (LOFF, COFF, FACTOR, FACTOR)
        assert nav.sublon == pytest.approx(SUBLON_EAST)
        assert nav.deltax == pytest.approx(65536.0 / 13642337.0, rel=1e-8)


def test_bad_blocks_are_rejected():
    with pytest.raises(ValueError, match="6 are needed"):
        AreaNav(navblock=msgt_block()[:4], byte_order=">", int_format="i")
    with pytest.raises(ValueError, match="cannot be 0"):
        AreaNav(navblock=msgt_block(cfac=0), byte_order=">", int_format="i")
    nav = MSGTNav.__new__(MSGTNav, navblock=msgt_block(), byte_order=">", int_format="i")
    nav.navtype = "GVAR"
    with pytest.raises(ValueError, match="Invalid navigation type"):
        nav.__init__(navblock=msgt_block())


def test_the_centre_of_the_grid_is_the_subpoint():
    lat, lon = make_nav().to_latlon(CENTRE, CENTRE)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(SUBLON_EAST, abs=1e-9)


def test_to_latlon_is_the_line_of_sight_meeting_the_ellipsoid():
    lines, elems = np.meshgrid(np.linspace(1200, 9900, 8), np.linspace(1200, 9900, 8), indexing="ij")
    lat, lon = make_nav().to_latlon(lines, elems)
    ok = np.isfinite(lat)
    assert ok.sum() > 40
    reference = np.array([[ray_latlon(a, b) for a, b in zip(row_l, row_e)] for row_l, row_e in zip(lines, elems)])
    np.testing.assert_allclose(lat[ok], reference[..., 0][ok], atol=1e-8)
    np.testing.assert_allclose(lon[ok], reference[..., 1][ok], atol=1e-8)


def test_lines_run_from_north_to_south_and_elements_from_west_to_east():
    nav = make_nav()
    lat, _ = nav.to_latlon(np.array([3000.0, 8000.0]), np.array([CENTRE, CENTRE]))
    assert lat[0] > 0 > lat[1]
    _, lon = nav.to_latlon(np.array([CENTRE, CENTRE]), np.array([3000.0, 8000.0]))
    assert lon[0] < SUBLON_EAST < lon[1]


def test_three_pixels_of_the_image_are_one_pixel_of_the_scan_grid():
    """At the subpoint a SEVIRI pixel is 3 km, which is 0.027 degrees of latitude; the image pixel is a third of it."""
    nav = make_nav()
    lat3 = nav.to_latlon(np.array([CENTRE - 3.0, CENTRE - 6.0, CENTRE - 1.0]), np.array([CENTRE] * 3))[0]
    assert 0.0265 < lat3[0] < 0.0275
    assert lat3[1] == pytest.approx(2 * lat3[0], rel=1e-3)
    assert lat3[2] == pytest.approx(lat3[0] / 3, rel=1e-3)


def test_the_round_trip_gives_back_the_pixel():
    nav = make_nav()
    lines, elems = np.meshgrid(np.linspace(1500, 9600, 9), np.linspace(1500, 9600, 9), indexing="ij")
    lat, lon = nav.to_latlon(lines, elems)
    back_lines, back_elems = nav.to_lin_ele(lat, lon)
    ok = np.isfinite(lat)
    assert ok.sum() > 60
    np.testing.assert_allclose(back_lines[ok], lines[ok], atol=1e-6)
    np.testing.assert_allclose(back_elems[ok], elems[ok], atol=1e-6)


def test_space_is_nan_and_the_far_side_has_no_pixel():
    nav = make_nav()
    lat, lon = nav.to_latlon(np.array([0.0, 11136.0, np.nan]), np.array([0.0, 11136.0, 100.0]))
    assert np.isnan(lat).all() and np.isnan(lon).all()
    line, elem = nav.to_lin_ele(0.0, SUBLON_EAST + 180.0)
    assert np.isnan(line) and np.isnan(elem)
    assert np.isnan(nav.to_lin_ele(np.nan, 10.0)[0]) and np.isnan(nav.to_lin_ele(10.0, np.nan)[1])


def test_the_area_to_image_conversion_is_applied():
    nav = make_nav()
    nav.setRes(2, 2)
    nav.setImageStart(4000, 4400)
    area = nav.to_latlon(10.0, 20.0)  # image line 4000 + 2 * 10, element 4400 + 2 * 20
    assert np.isfinite(area[0]) and area == make_nav().to_latlon(4020.0, 4440.0)


def test_the_list_interface_has_elements_first_like_visad():
    nav = make_nav()
    lat, lon = nav.to_latlon(np.array([4000.0]), np.array([6000.0]))
    out = nav.toLatLon([[6000.0], [4000.0]])  # [[elements], [lines]]
    assert out[0][0] == lat[0] and out[1][0] == lon[0]
    line_elem = nav.toLinEle(out)
    assert line_elem[nav.indexLine][0] == pytest.approx(4000.0, abs=1e-6)
    assert line_elem[nav.indexEle][0] == pytest.approx(6000.0, abs=1e-6)
    assert nav.toLatLon2(np.array([4000.0]), np.array([6000.0]))[0][0] == lat[0]


@pytest.mark.parametrize("endian", [">", "<"])
def test_a_file_is_located_through_get_lonlats(tmp_path, endian):
    spec = FakeArea(GOES7_IR, mode_a_data([100]), source_type="MSGT", ul_line=5567, ul_element=5564, line_res=1,
                    elem_res=1, nav_words=np.array(msgt_block()), endian=endian)
    lon, lat = AreaFile(write_area_file(tmp_path / "msgt.area", spec)).get_lonlats()
    assert (lat[4, 7], lon[4, 7]) == pytest.approx((0.0, SUBLON_EAST), abs=1e-9)  # image line and element 5571
    assert lat[0, 7] > lat[-1, 7]
