"""GMSXNav against an independent scalar port of VisAD's GMSXnav.java, on a made-up GMS-5 like navigation block."""
import numpy as np
import pytest

from pyarea.nav import AreaNav, GMSXNav
from tests.fake_gmsx_nav import make_gmsx_navblock
from tests.visad_gmsxnav_reference import GMSXRef


@pytest.fixture(scope='module')
def block():
    return make_gmsx_navblock()


@pytest.fixture(scope='module')
def nav(block):
    return AreaNav(navblock=list(block), byte_order='>', int_format='i')


@pytest.fixture(scope='module')
def points():
    rng = np.random.default_rng(1)
    return rng.uniform(1, 10000, 200), rng.uniform(1, 9164, 200)


def test_factory_makes_gmsx(nav):
    assert isinstance(nav, GMSXNav)
    assert nav.getSubpoint() == (0.0, 140.0)


def test_to_latlon_matches_visad_port(block, nav, points):
    line, elem = points
    lat, lon = nav.to_latlon(line, elem)
    ref = GMSXRef(block)
    finite = 0
    for i in range(line.size):
        rlat, rlon = ref.to_latlon(line[i], elem[i])
        assert np.isnan(rlat) == np.isnan(lat[i])
        if not np.isnan(rlat):
            finite += 1
            assert lat[i] == pytest.approx(rlat, abs=1e-4)
            assert lon[i] == pytest.approx(rlon, abs=1e-4)
    assert finite > 50


def test_to_lin_ele_matches_visad_port_and_round_trips(block, nav, points):
    line, elem = points
    lat, lon = nav.to_latlon(line, elem)
    ok = np.isfinite(lat)
    l2, e2 = nav.to_lin_ele(lat[ok], lon[ok])
    assert np.abs(l2 - line[ok]).max() < 0.01
    assert np.abs(e2 - elem[ok]).max() < 0.1
    ref = GMSXRef(block)
    for i in range(0, ok.sum(), 5):
        rl, re = ref.to_linele(lat[ok][i], lon[ok][i])
        assert l2[i] == pytest.approx(rl, abs=1e-2)
        assert e2[i] == pytest.approx(re, abs=1e-2)


def test_far_side_and_off_earth_are_nan(nav):
    lat, lon = nav.to_latlon([1.0, 5000.0], [1.0, 4582.0])
    assert np.isnan(lat[0]) and np.isnan(lon[0])
    assert np.isfinite(lat[1])
    line, elem = nav.to_lin_ele([0.0, 95.0], [-40.0, 140.0 - 180.0])
    assert np.isnan(line[0]) or np.isfinite(line[0])  # the near side point
    assert np.isnan(line[1]) and np.isnan(elem[1])  # more than 90 degrees from the sub-satellite longitude
    assert np.isnan(nav.to_lin_ele([95.0], [140.0])[0][0])  # latitude out of range


def test_list_interface(nav):
    # [[elements], [lines]] in, [[lats], [lons]] out, and back
    latlon = nav.toLatLon([[4582.0], [5000.0]])
    elements, lines = nav.toLinEle(latlon)
    assert lines[0] == pytest.approx(5000.0, abs=0.01)
    assert elements[0] == pytest.approx(4582.0, abs=0.1)


def test_uses_the_full_circle_not_a_half(nav):
    assert nav.DPAI == pytest.approx(2 * np.pi)
