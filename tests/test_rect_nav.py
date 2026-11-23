"""Tests for RECT (rectangular latitude/longitude) navigation."""
import numpy as np
import pytest

from pyarea.nav import AreaNav, RECTNav

RECT = int.from_bytes(b"RECT", "little", signed=True)


def block(row, lat, col, lon, dlat, dlon, west, powers=(4, 4, 4, 4)):
    """A RECT navigation block; ``west`` >= 0 means west-positive longitudes, < 0 east-positive."""
    scale = [10 ** (power or 4) for power in powers]  # a power word of 0 means the values are scaled by 10**4
    return (RECT, row, int(round(lat * scale[0])), col, int(round(lon * scale[1])), int(round(dlat * scale[2])),
            int(round(dlon * scale[3])), 6378388, 81992, 0, west, *powers, 0, 0)


def make(blk):
    return AreaNav(navblock=blk, byte_order="<", int_format="i")


# values recorded from the scalar implementation this one replaced: (block, lines, elements, lats, lons)
RECORDED = [
    (block(101, 40.0, 201, -100.0, 0.25, 0.25, 0), [10.0, 101.0, 150.5, 301.0], [20.0, 201.0, 260.0, 400.0],
     [62.75, 40.0, 27.625, -10.0], [54.75, 100.0, 114.75, 149.75]),
    (block(1, 90.0, 1, -180.0, 0.5, 0.5, 0), [1.0, 181.0, 361.0, 100.0], [1.0, 360.0, 720.0, 5.0],
     [90.0, 0.0, -90.0, 40.5], [180.0, -0.5, 179.5, -178.0]),
    (block(51, 10.0, 301, 70.0, 0.1, 0.1, 5), [51.0, 80.0, 20.0], [301.0, 350.0, 250.0],
     [10.0, 7.1, 13.1], [-70.0, -65.1, -75.1]),
]


@pytest.mark.parametrize(("blk", "lines", "elems", "lats", "lons"), RECORDED)
def test_matches_recorded_values(blk, lines, elems, lats, lons):
    nav = make(blk)
    assert type(nav) is RECTNav
    lat, lon = nav.toLatLon([elems, lines])
    np.testing.assert_allclose(lat, lats, atol=1e-9)
    np.testing.assert_allclose(lon, lons, atol=1e-9)
    elem_back, line_back = nav.toLinEle([lats, lons])
    np.testing.assert_allclose(elem_back, elems, atol=1e-7)
    np.testing.assert_allclose(line_back, lines, atol=1e-7)


def test_longitude_conventions():
    """West-positive blocks store west longitudes as positive; either way the result is east-positive."""
    west = make(block(11, 30.0, 21, 100.0, 0.1, 0.1, 0))   # 100 degrees west
    east = make(block(11, 30.0, 21, -100.0, 0.1, 0.1, -1))  # the same place, east-positive
    for nav in (west, east):
        lat, lon = nav.to_latlon(11.0, 21.0)
        assert float(lat) == pytest.approx(30.0)
        assert float(lon) == pytest.approx(-100.0)
        # with a positive step, columns to the right are further east: 99 degrees west
        assert float(nav.to_latlon(11.0, 31.0)[1]) == pytest.approx(-99.0)


def test_zero_power_words_default_to_four():
    """A power-of-ten word of 0 means 10**4, as in the McIDAS and VisAD implementations."""
    explicit = make(block(11, 30.0, 21, -100.0, 0.1, 0.1, -1))
    defaults = make(block(11, 30.0, 21, -100.0, 0.1, 0.1, -1, powers=(0, 0, 0, 0)))
    for attr in ("zslat", "zslon", "zdlat", "zdlon"):
        assert getattr(defaults, attr) == pytest.approx(getattr(explicit, attr))
    assert defaults.zdlat == pytest.approx(0.1)


def test_other_powers_of_ten():
    """Different powers scale the same way: values stored times 1e3 or 1e5 give the same navigation."""
    ref = make(block(11, 30.0, 21, -100.0, 0.1, 0.1, -1))
    for powers in ((3, 5, 6, 2), (5, 5, 5, 5)):
        other = make(block(11, 30.0, 21, -100.0, 0.1, 0.1, -1, powers=powers))
        np.testing.assert_allclose(other.to_latlon([5.0, 50.0], [7.0, 90.0]), ref.to_latlon([5.0, 50.0], [7.0, 90.0]),
                                   atol=1e-9)


def test_round_trip_and_shapes():
    nav = make(block(101, 40.0, 201, -100.0, 0.25, 0.25, -1))
    lines, elems = np.meshgrid(np.linspace(1, 300, 7), np.linspace(1, 500, 9), indexing="ij")
    lat, lon = nav.to_latlon(lines, elems)
    assert lat.shape == lon.shape == (7, 9)
    back_line, back_elem = nav.to_lin_ele(lat, lon)
    ok = np.isfinite(lat)
    np.testing.assert_allclose(back_line[ok], lines[ok], atol=1e-9)
    np.testing.assert_allclose(back_elem[ok], elems[ok], atol=1e-9)
    # a column of lines against a row of elements broadcasts
    assert nav.to_latlon(np.array([[10.0], [20.0]]), np.array([1.0, 2.0, 3.0]))[0].shape == (2, 3)
    assert np.ndim(nav.to_latlon(10.0, 20.0)[0]) == 0


def test_out_of_range_and_nan():
    nav = make(block(101, 40.0, 201, -100.0, 0.5, 0.5, -1))
    lat, lon = nav.to_latlon([-1000.0, 101.0, np.nan, 101.0, 101.0], [201.0, 201.0, 201.0, np.nan, 5000.0])
    assert np.isnan(lat[[0, 2, 3, 4]]).all() and np.isnan(lon[[0, 2, 3, 4]]).all()  # lat > 90, NaN in, > 180 away
    assert lat[1] == pytest.approx(40.0) and lon[1] == pytest.approx(-100.0)


def test_longitude_wrapping():
    """Longitudes past the date line are wrapped into -180..180, unless asked not to."""
    nav = make(block(11, 0.0, 21, 170.0, 1.0, 1.0, -1))
    assert float(nav.to_latlon(11.0, 41.0)[1]) == pytest.approx(-170.0)   # 190 wrapped
    assert float(nav.to_latlon(11.0, 41.0, lon_circular=False)[1]) == pytest.approx(190.0)
    # and the other way: -170 and 190 are the same column
    assert float(nav.to_lin_ele(0.0, -170.0)[1]) == pytest.approx(41.0)
    assert float(nav.to_lin_ele(0.0, 190.0)[1]) == pytest.approx(41.0)


def test_image_start_and_resolution():
    """Navigation is in image coordinates; the file's start and resolution map file coordinates onto them."""
    nav = make(block(101, 40.0, 201, -100.0, 0.25, 0.25, -1))
    nav.setImageStart(1, 1)
    nav.setRes(2, 4)
    nav.setMag(1, 1)
    nav.setStart(0, 0)
    # file line 50 is image line 1 + 2 * 50 = 101, and file element 50 is image element 1 + 4 * 50 = 201
    lat, lon = nav.to_latlon(50.0, 50.0)
    assert float(lat) == pytest.approx(40.0) and float(lon) == pytest.approx(-100.0)
    line, elem = nav.to_lin_ele(40.0, -100.0)
    assert float(line) == pytest.approx(50.0) and float(elem) == pytest.approx(50.0)


def test_list_interface_returns_arrays_in_the_convention_order():
    nav = make(block(101, 40.0, 201, -100.0, 0.25, 0.25, -1))
    latlon = nav.toLatLon([[201.0, 205.0], [101.0, 105.0]])      # [[elements], [lines]]
    np.testing.assert_allclose(latlon[nav.indexLat], [40.0, 39.0])
    np.testing.assert_allclose(latlon[nav.indexLon], [-100.0, -99.0])
    linele = nav.toLinEle(latlon)
    np.testing.assert_allclose(linele[nav.indexEle], [201.0, 205.0], atol=1e-9)
    np.testing.assert_allclose(linele[nav.indexLine], [101.0, 105.0], atol=1e-9)


def test_str_lists_the_parameters():
    text = str(make(block(101, 40.0, 201, -100.0, 0.25, 0.25, -1)))
    assert "xrow 101" in text and "zdlat 0.250000" in text
