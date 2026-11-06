"""Tests for the pre-GVAR GOES navigation (``pyarea.nav.GOESNav``)."""

import numpy as np
import pytest

import pyarea.nav as nav
from tests.fake_area_files import IR_WINDOW, NADIR_COL, NADIR_ROW, make_nav_words


def _jit_methods():
    """The numba-compiled functions of the navigation module."""
    return {name: obj for name, obj in vars(nav).items() if hasattr(obj, "py_func")}


@pytest.fixture(params=[False, True], autouse=True)
def _disable_jit(request, monkeypatch):
    """Run tests with jit enabled and disabled.

    Reason: Coverage report is only accurate with jit disabled.
    """
    if request.param:
        for name, method in _jit_methods().items():
            monkeypatch.setattr(nav, name, method.py_func)


def _build(words):
    """Build the navigation the way AreaFile.navigation does."""
    gnav = nav.GOESNav(navblock=tuple(int(w) for w in words), byte_order=">", int_format="i")
    gnav.setImageStart(IR_WINDOW["ul_line"], IR_WINDOW["ul_element"])
    gnav.setRes(IR_WINDOW["line_res"], IR_WINDOW["elem_res"])
    gnav.setStart(0, 0)
    gnav.setMag(1, 1)
    return gnav


def _nav_with(changes):
    """Build a GOESNav from the default nav block with some words replaced."""
    words = make_nav_words()
    for index, value in changes.items():
        words[index] = value
    return _build(words)


class TestNavigationInternals:
    """Epoch handling, scan geometry words and degenerate cases of the navigation."""

    @pytest.mark.parametrize(("yymmdd", "expected"), [(920823, 92236), (840301, 84061), (841301, 84001)])
    def test_year_day_conversion(self, yymmdd, expected):
        """Convert YYMMDD to YYDDD, with leap days and an invalid month falling back to January."""
        assert nav._icon1(yymmdd) == expected

    @pytest.mark.parametrize(("year", "days"), [(84, 366), (85, 365), (1084, 366)])
    def test_days_in_year(self, year, days):
        """Count the days of leap and normal years."""
        assert nav._leapyr(year) == days

    @pytest.mark.parametrize(
        ("hours", "expected"),
        [(50.0, (2.0, 2)), (30.0, (6.0, 1)), (12.0, (12.0, 0)), (-5.0, (19.0, -1)), (-30.0, (18.0, -2))])
    def test_day_rollover(self, hours, expected):
        """Wrap hours into a day and report the whole days that were moved."""
        assert nav._day_rollover(hours) == expected

    def test_add_days(self):
        """Move an epoch within a year and back over the start of a leap year."""
        assert nav._add_days(84100, 2) == 84102
        assert nav._add_days(85001, -1) == 84366

    def test_epoch_moves_to_the_previous_day(self):
        """An anomaly longer than the time of day puts the epoch on the day before."""
        assert nav._epoch(92236, 10000, 3.0) == (92235, 220000)

    def test_epoch_moves_to_the_next_day(self):
        """A negative anomaly can carry the epoch over midnight."""
        assert nav._epoch(92236, 230000, -3.0) == (92237, 20000)

    def test_wrong_magic_is_rejected(self):
        """Refuse a nav block that does not start with GOES."""
        words = make_nav_words()
        words[0] = 0
        with pytest.raises(ValueError, match="Not a GOES nav block"):
            _build(words)

    def test_picture_center_line_with_four_decimals(self):
        """A picture center line of a million or more is stored times ten thousand."""
        assert _nav_with({14: 7285 * 10000}).piclin == 7285.0

    def test_spin_rate_in_rpm(self):
        """A spin value below 300 is a rate in rpm, not a period in milliseconds."""
        assert _nav_with({15: 100000}).tmpscl == pytest.approx(_nav_with({}).tmpscl)

    def test_scan_times_default_to_the_picture_time(self):
        """Without scan line times in the nav block, they follow from the picture time."""
        sat_nav = _nav_with({})
        assert sat_nav.scan1 == 1.0
        assert sat_nav.scan2 == 1821.0
        assert sat_nav.time1 == sat_nav.pictim
        assert sat_nav.time2 == pytest.approx(sat_nav.time1 + 1821 * sat_nav.tmpscl)

    def test_scan_times_from_the_nav_block(self):
        """Newer satellites record the first and last scan line and their times."""
        sat_nav = _nav_with({1: 26 * 100000 + 92236, 30: 5, 31: 130200, 34: 1800, 35: 131000})
        assert sat_nav.scan1 == 5.0
        assert sat_nav.scan2 == 1800.0
        assert sat_nav.time1 == pytest.approx(13 + 2 / 60)
        assert sat_nav.time2 == pytest.approx(13 + 10 / 60)

    def test_eccentric_orbit(self):
        """Solve Kepler's equation iteratively for an orbit that is not circular."""
        sat_nav = _nav_with({7: 50000, 9: 120000})
        lat, lon = sat_nav.to_latlon(np.array([NADIR_ROW + 1.0]), np.array([NADIR_COL + 1.0]))
        assert np.isfinite(lat).all()
        assert np.isfinite(lon).all()
        assert all(np.isfinite(sat_nav.get_subpoint()))

    def test_degenerate_surface_point_has_no_position(self):
        """A point on the rotation axis has no longitude."""
        params = _nav_with({})._kernel_params()
        lat, lon = nav._geodetic_latlon((0.0, 0.0, 1.0), 1.0, 0.0, params)
        assert np.isnan(lat)
        assert np.isnan(lon)

    def test_line_of_sight_missing_the_earth(self):
        """Lines of sight beyond the limb have no position."""
        lat, lon = _nav_with({}).to_latlon(np.array([-1000.0]), np.array([NADIR_COL + 1.0]))
        assert np.isnan(lat).all()
        assert np.isnan(lon).all()


class TestPyareaInterface:
    """The AreaNav-style API: [[elements], [lines]] in, [[lats], [lons]] out."""

    @staticmethod
    def _grid():
        offsets = np.arange(-250, 250, 25.0)
        lines, elems = np.meshgrid(offsets + NADIR_ROW + 1, offsets + NADIR_COL + 1, indexing="ij")
        return lines.ravel(), elems.ravel()

    def test_to_latlon_matches_the_list_interface(self):
        """toLatLon is to_latlon with the pyarea argument order."""
        gnav = _build(make_nav_words())
        lines, elems = self._grid()
        lat, lon = gnav.to_latlon(lines, elems)
        list_lat, list_lon = gnav.toLatLon([elems.tolist(), lines.tolist()])
        np.testing.assert_allclose(list_lat, lat)
        np.testing.assert_allclose(list_lon, lon)

    def test_lin_ele_round_trip(self):
        """Going to lat/lon and back lands on the same pixel."""
        gnav = _build(make_nav_words())
        lines, elems = self._grid()
        lat, lon = gnav.to_latlon(lines, elems)
        on_earth = np.isfinite(lat)
        assert on_earth.any()
        elem_back, line_back = gnav.toLinEle([lat, lon])
        np.testing.assert_allclose(line_back[on_earth], lines[on_earth], atol=1e-4)
        np.testing.assert_allclose(elem_back[on_earth], elems[on_earth], atol=1e-4)

    def test_lin_ele_of_the_far_side_has_no_element(self):
        """A point on the other side of the earth is not visible."""
        gnav = _build(make_nav_words())
        _, elem = gnav.to_lin_ele(0.0, 180.0 + gnav.get_subpoint()[1])
        assert np.isnan(elem)

    def test_lin_ele_of_an_impossible_latitude_is_nan(self):
        """Latitudes beyond the poles have no pixel."""
        gnav = _build(make_nav_words())
        line, elem = gnav.to_lin_ele(95.0, 0.0)
        assert np.isnan(line)
        assert np.isnan(elem)

    def test_subpoint(self):
        """getSubpoint reports the sub-satellite latitude and longitude."""
        lat, lon = _build(make_nav_words()).getSubpoint()
        assert lat == pytest.approx(0.0, abs=1e-6)
        assert lon == pytest.approx(-112.1, abs=0.01)
