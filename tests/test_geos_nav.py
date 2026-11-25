"""GEOSNav, the navigation of AREA files in the LRIT/HRIT layout (nav type GEOS: Himawari, COMS-1), against CGMS 03."""
import struct

import numpy as np
import pytest

from pyarea.nav import AreaNav, GEOSNav

# Himawari-8 AHI full disk at 2 km: HRIT COFF = LOFF = 2750.5 and CFAC = LFAC = 20466275, subpoint 140.7 E
COFF = LOFF = 2750.5
CFAC = LFAC = 20466275
SUBPOINT = 140.7
X42, RADEQ, RADPOL = 42164.0, 6378.1690, 6356.5838


def geos_block(order=">", bres=1, flag=None, loff=LOFF, coff=COFF, lfac=LFAC, cfac=CFAC, plon=SUBPOINT, words=None):
    block = [struct.unpack(f"{order}i", b"GEOS")[0], round(loff * 10), round(coff * 10), round(lfac * 10),
             round(cfac * 10), round(plon * 10), bres]
    if flag is not None:
        block.append(flag)
    return tuple(block + [0] * (640 - len(block))) if words is None else words


def make_nav(order=">", **kwargs):
    nav = AreaNav(navblock=geos_block(order, **kwargs), byte_order=order, int_format="i")
    nav.setImageStart(0, 0)
    nav.setRes(1, 1)
    nav.setMag(1, 1)
    nav.setStart(0, 0)
    return nav


def cgms_latlon(column, line, bres=1, shift=None):
    """The CGMS 03 section 4.4.4 inverse navigation, written from the specification, with McIDAS' constants."""
    shift = bres - 1 if shift is None else shift
    c, l_ = (column + shift) / bres, (line + shift) / bres
    x = np.radians((c - COFF) * 2.0 ** 16 / CFAC)
    y = np.radians((l_ - LOFF) * 2.0 ** 16 / LFAC)
    with np.errstate(invalid="ignore"):  # NaN for the pixels in space
        sd = np.sqrt((X42 * np.cos(x) * np.cos(y)) ** 2
                     - (np.cos(y) ** 2 + 1.006803 * np.sin(y) ** 2) * 1737121856.0)
    sn = (X42 * np.cos(x) * np.cos(y) - sd) / (np.cos(y) ** 2 + 1.006803 * np.sin(y) ** 2)
    s1, s2, s3 = X42 - sn * np.cos(x) * np.cos(y), sn * np.sin(x) * np.cos(y), -sn * np.sin(y)
    sxy = np.sqrt(s1 ** 2 + s2 ** 2)
    return np.degrees(np.arctan(1.006803 * s3 / sxy)), np.degrees(np.arctan(s2 / s1)) + SUBPOINT


def test_the_block_is_read_in_either_byte_order():
    for order in (">", "<"):
        nav = make_nav(order)
        assert type(nav) is GEOSNav
        assert (nav.loff, nav.coff, nav.lfac, nav.cfac, nav.plon, nav.bres, nav.pixelOffsetFlag) == (
            27505, 27505, 204662750, 204662750, 1407, 1, 0)


def test_the_resolution_defaults_to_one_and_short_blocks_are_rejected():
    nav = AreaNav(navblock=geos_block(words=geos_block()[:6]), byte_order=">", int_format="i")
    assert (nav.bres, nav.pixelOffsetFlag) == (1, 0)  # as in Java, where a missing or non-positive word 7 is 1
    nav = AreaNav(navblock=geos_block(bres=0), byte_order=">", int_format="i")
    assert nav.bres == 1
    with pytest.raises(ValueError, match="6 are needed"):
        AreaNav(navblock=geos_block()[:5], byte_order=">", int_format="i")
    nav = GEOSNav.__new__(GEOSNav, navblock=geos_block(), byte_order=">", int_format="i")
    nav.navtype = "GVAR"
    with pytest.raises(ValueError, match="Invalid navigation type"):
        nav.__init__(navblock=geos_block())


def test_the_centre_of_the_grid_is_the_subpoint():
    lat, lon = make_nav().to_latlon(COFF, COFF)
    assert lat == pytest.approx(0.0, abs=1e-9)
    assert lon == pytest.approx(SUBPOINT, abs=1e-9)


def test_to_latlon_is_the_cgms_algorithm():
    lines, elems = np.meshgrid(np.linspace(300, 2500, 7), np.linspace(300, 2500, 7), indexing="ij")
    lat, lon = make_nav().to_latlon(lines, elems)
    ok = np.isfinite(lat)
    assert ok.sum() > 25
    ref_lat, ref_lon = cgms_latlon(elems, lines)
    np.testing.assert_allclose(lat[ok], ref_lat[ok], atol=1e-9)
    np.testing.assert_allclose(lon[ok], ref_lon[ok], atol=1e-9)


@pytest.mark.parametrize("bres", [1, 2, 4])
@pytest.mark.parametrize("flag", [None, 0, 1])
def test_the_base_resolution_and_the_pixel_offset_flag(bres, flag):
    """Image pixels are in base resolution units; only a flag of exactly 1 selects the half pixel adjustment."""
    nav = make_nav(bres=bres, flag=flag)
    line, elem = 1200.0 * bres + 1.0, 1500.0 * bres + 2.0
    shift = (bres - 1) / 2.0 if flag == 1 else None
    lat, lon = nav.to_latlon(line, elem)
    ref_lat, ref_lon = cgms_latlon(elem, line, bres, shift)
    assert (lat, lon) == (pytest.approx(ref_lat, abs=1e-9), pytest.approx(ref_lon, abs=1e-9))


def test_the_flag_moves_the_pixels_by_half_a_base_pixel_at_most():
    on, off = make_nav(bres=4, flag=1), make_nav(bres=4, flag=0)
    lat1, _ = on.to_latlon(4000.0, 4000.0)
    lat0, _ = off.to_latlon(4000.0, 4000.0)
    step = np.degrees(2.0 ** 16 / CFAC) * 42164.0 / 6378.0 / 4  # about one pixel in degrees of latitude
    assert 0 < abs(lat1 - lat0) < 3 * step


def test_lines_run_from_north_to_south_and_elements_from_west_to_east():
    nav = make_nav()
    lat, _ = nav.to_latlon(np.array([1000.0, 2000.0]), np.array([1400.0, 1400.0]))
    assert lat[0] > lat[1]
    _, lon = nav.to_latlon(np.array([1400.0, 1400.0]), np.array([1000.0, 2000.0]))
    assert lon[0] < lon[1]


@pytest.mark.parametrize("bres", [1, 4])
@pytest.mark.parametrize("flag", [0, 1])
def test_the_round_trip_gives_back_the_pixel(bres, flag):
    nav = make_nav(bres=bres, flag=flag)
    lines, elems = np.meshgrid(np.linspace(300, 2500, 9) * bres, np.linspace(300, 2500, 9) * bres, indexing="ij")
    lat, lon = nav.to_latlon(lines, elems)
    back_lines, back_elems = nav.to_lin_ele(lat, lon)
    ok = np.isfinite(lat)
    assert ok.sum() > 30
    np.testing.assert_allclose(back_lines[ok], lines[ok], atol=1e-3 * bres)
    np.testing.assert_allclose(back_elems[ok], elems[ok], atol=1e-3 * bres)


def test_space_is_nan_and_the_far_side_has_no_pixel():
    nav = make_nav()
    lat, lon = nav.to_latlon(np.array([0.0, 5499.0, np.nan]), np.array([0.0, 5499.0, 100.0]))
    assert np.isnan(lat).all() and np.isnan(lon).all()
    line, elem = nav.to_lin_ele(0.0, SUBPOINT + 180.0)
    assert np.isnan(line) and np.isnan(elem)
    line, elem = nav.to_lin_ele(np.array([np.nan]), np.array([140.0]))
    assert np.isnan(line).all() and np.isnan(elem).all()


def test_longitudes_wrap_into_minus_180_to_180():
    nav = make_nav(plon=179.0)
    _, lon = nav.to_latlon(np.array([COFF]), np.array([COFF + 1000]))
    assert -180 <= lon[0] < 0  # east of the subpoint, past the date line


def test_the_area_to_image_conversion_is_applied():
    nav = make_nav()
    nav.setRes(2, 2)
    nav.setImageStart(1000, 1200)
    area = nav.to_latlon(10.0, 20.0)  # image line 1000 + 2 * 10, element 1200 + 2 * 20
    assert area == make_nav().to_latlon(1020.0, 1240.0)


def test_the_list_interface_has_elements_first_like_visad():
    nav = make_nav()
    lat, lon = nav.to_latlon(np.array([1000.0]), np.array([1400.0]))
    out = nav.toLatLon([[1400.0], [1000.0]])  # [[elements], [lines]]
    assert out[0][0] == lat[0] and out[1][0] == lon[0]
    line_elem = nav.toLinEle(out)
    assert line_elem[nav.indexLine][0] == pytest.approx(1000.0, abs=1e-3)
    assert line_elem[nav.indexEle][0] == pytest.approx(1400.0, abs=1e-3)
    assert nav.toLatLon2(np.array([1000.0]), np.array([1400.0]))[0][0] == lat[0]
