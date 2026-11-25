"""ABINNav, the navigation of GOES-R ABI fixed grid AREA files (nav type ABIN), against the GOES-R PUG formulas."""
import struct

import numpy as np
import pytest

from pyarea.nav import ABINNav, AreaNav

# GOES-East band 13 (2 km): the scan angles of the first line and element and their steps, in radians
Y_TOP, X_LEFT, STEP = 0.151858, -0.151858, 5.6e-05
SUBPOINT = -75.0
H, REQ, RPOL = 42164.16, 6378.137, 6356.7523
MCIDAS_RATIO = 1.006803  # FP in nvxabin.dlm; the PUG has REQ ** 2 / RPOL ** 2 = 1.0067395
PUG_RATIO = REQ ** 2 / RPOL ** 2


def abin_block(order=">", bres=4, loff=Y_TOP, coff=X_LEFT, lfac=-STEP, cfac=STEP, plon=SUBPOINT):
    words = [struct.unpack(f"{order}i", b"ABIN")[0], round(loff * 1e8), round(coff * 1e8), round(lfac * 1e8),
             round(cfac * 1e8), round(plon * 10), bres]
    return tuple(words + [0] * 633)


def make_nav(order=">", **kwargs):
    nav = AreaNav(navblock=abin_block(order, **kwargs), byte_order=order, int_format="i")
    nav.setImageStart(0, 0)
    nav.setRes(1, 1)
    nav.setMag(1, 1)
    nav.setStart(0, 0)
    return nav


def pug_latlon(x, y, lon_0=SUBPOINT, ratio=PUG_RATIO):
    """The fixed grid scan angles x (east-west) and y (north-south) to lat/lon, as in the GOES-R PUG (section 4.2.8).

    ``ratio`` is the (equatorial / polar radius) squared of the formulas.
    """
    a = np.sin(x) ** 2 + np.cos(x) ** 2 * (np.cos(y) ** 2 + ratio * np.sin(y) ** 2)
    b = -2.0 * H * np.cos(x) * np.cos(y)
    c = H ** 2 - REQ ** 2
    with np.errstate(invalid="ignore"):
        rs = (-b - np.sqrt(b ** 2 - 4.0 * a * c)) / (2.0 * a)
    sx = rs * np.cos(x) * np.cos(y)
    sy = -rs * np.sin(x)
    sz = rs * np.cos(x) * np.sin(y)
    lat = np.degrees(np.arctan(ratio * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
    lon = lon_0 - np.degrees(np.arctan(sy / (H - sx)))
    return lat, lon


def test_the_block_is_read_in_either_byte_order():
    for order in (">", "<"):
        nav = make_nav(order)
        assert type(nav) is ABINNav
        assert (nav.loff, nav.coff, nav.lfac, nav.cfac, nav.plon, nav.bres) == (
            pytest.approx(Y_TOP), pytest.approx(X_LEFT), pytest.approx(-STEP), pytest.approx(STEP), SUBPOINT, 4.0)


def test_short_blocks_and_other_nav_types_are_rejected():
    with pytest.raises(ValueError, match="7 are needed"):
        AreaNav(navblock=abin_block()[:5], byte_order=">", int_format="i")
    nav = ABINNav.__new__(ABINNav, navblock=abin_block(), byte_order=">", int_format="i")
    nav.navtype = "GVAR"
    with pytest.raises(ValueError, match="Invalid navigation type"):
        nav.__init__(navblock=abin_block())


def test_the_centre_of_the_grid_is_the_subpoint():
    nav = make_nav(bres=1, loff=0.0, coff=0.0)
    lat, lon = nav.to_latlon(0.0, 0.0)
    assert lat == pytest.approx(0.0, abs=1e-12)
    assert lon == pytest.approx(SUBPOINT, abs=1e-12)


def grid_points(bres):
    """Pixels spread over the disc: the native line and element numbers, and their image coordinates."""
    native_lines = np.array([700, 1500, 2712, 3000, 4200, 2000])
    native_elems = np.array([2712, 4000, 2712, 800, 3300, 1500])
    return native_lines, native_elems, native_lines * bres + (bres - 1) / 2, native_elems * bres + (bres - 1) / 2


@pytest.mark.parametrize("bres", [1, 4])
def test_to_latlon_is_the_pug_algorithm_with_the_mcidas_constant(bres):
    """Written from the PUG equations, with the (equatorial / polar radius) squared McIDAS uses: the same to 1e-9."""
    nav = make_nav(bres=bres)
    native_lines, native_elems, lines, elems = grid_points(bres)
    lat, lon = nav.to_latlon(lines, elems)
    expected_lat, expected_lon = pug_latlon(X_LEFT + native_elems * STEP, Y_TOP - native_lines * STEP,
                                            ratio=MCIDAS_RATIO)
    assert np.isfinite(lat).all()
    np.testing.assert_allclose(lat, expected_lat, atol=1e-9)
    np.testing.assert_allclose(lon, expected_lon, atol=1e-9)


def test_to_latlon_differs_from_the_pug_by_the_radius_ratio_only():
    """McIDAS has 1.006803 where the PUG has 1.0067395, which moves a point by up to about 0.003 degrees (300 m)."""
    nav = make_nav(bres=1)
    native_lines, native_elems, lines, elems = grid_points(1)
    lat, lon = nav.to_latlon(lines, elems)
    pug_lat, pug_lon = pug_latlon(X_LEFT + native_elems * STEP, Y_TOP - native_lines * STEP)
    assert 0 < np.abs(lat - pug_lat).max() < 0.003
    assert np.abs(lon - pug_lon).max() < 0.003


def test_lines_run_from_north_to_south_and_elements_from_west_to_east():
    nav = make_nav(bres=1)
    lat, lon = nav.to_latlon(np.array([1000.0, 3000.0]), np.array([2712.0, 2712.0]))
    assert lat[0] > lat[1]
    lat, lon = nav.to_latlon(np.array([2712.0, 2712.0]), np.array([1000.0, 4000.0]))
    assert lon[0] < lon[1]


@pytest.mark.parametrize("bres", [1, 4])
def test_the_round_trip_gives_back_the_pixel(bres):
    nav = make_nav(bres=bres)
    lines, elems = np.meshgrid(np.linspace(300, 5000, 9) * bres, np.linspace(300, 5000, 9) * bres, indexing="ij")
    lat, lon = nav.to_latlon(lines, elems)
    back_lines, back_elems = nav.to_lin_ele(lat, lon)
    ok = np.isfinite(lat)
    assert ok.sum() > 20
    # as in VisAD the forward formulas use 1.006803 and the inverse the radii (1.0067395): about 0.1 pixel apart
    np.testing.assert_allclose(back_lines[ok], lines[ok], atol=0.15 * bres)
    np.testing.assert_allclose(back_elems[ok], elems[ok], atol=0.15 * bres)


def test_space_is_nan_and_the_far_side_has_no_pixel():
    nav = make_nav(bres=1)
    lat, lon = nav.to_latlon(np.array([0.0, 5423.0, np.nan]), np.array([0.0, 5423.0, 100.0]))  # corners, NaN
    assert np.isnan(lat).all() and np.isnan(lon).all()
    line, elem = nav.to_lin_ele(0.0, SUBPOINT + 180.0)  # the other side of the Earth
    assert np.isnan(line) and np.isnan(elem)


def test_longitudes_wrap_into_minus_180_to_180():
    nav = make_nav(bres=1, plon=179.0)
    lat, lon = nav.to_latlon(np.array([2712.0]), np.array([4000.0]))
    unwrapped = pug_latlon(X_LEFT + 4000 * STEP, Y_TOP - 2712 * STEP, lon_0=179.0, ratio=MCIDAS_RATIO)[1]
    assert unwrapped > 180  # east of the subpoint, past the date line
    assert lon[0] == pytest.approx(unwrapped - 360.0, abs=1e-9)


def test_the_area_to_image_conversion_is_applied():
    nav = make_nav(bres=1)
    nav.setRes(2, 2)
    nav.setImageStart(2000, 2500)
    area_lat, area_lon = nav.to_latlon(10.0, 20.0)  # image line 2000 + 2 * 10, element 2500 + 2 * 20
    lat, lon = make_nav(bres=1).to_latlon(2020.0, 2540.0)
    assert np.isfinite(lat) and (area_lat, area_lon) == (lat, lon)


def test_the_list_interface_has_elements_first_like_visad():
    nav = make_nav(bres=1)
    lat, lon = nav.to_latlon(np.array([2000.0]), np.array([2500.0]))
    out = nav.toLatLon([[2500.0], [2000.0]])  # [[elements], [lines]]
    assert out[0][0] == lat[0] and out[1][0] == lon[0]
    line_elem = nav.toLinEle(out)
    assert line_elem[nav.indexLine][0] == pytest.approx(2000.0, abs=0.15)
    assert line_elem[nav.indexEle][0] == pytest.approx(2500.0, abs=0.15)
