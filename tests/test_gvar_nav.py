"""Tests for GVAR (GOES-8 onward) navigation, against results recorded from real GOES-8 and GOES-13 files.

The JSON files in tests/data hold a navigation block, the image geometry and the lat/lon and line/element
results at a grid of points. They were produced by scripts/dump_gvar_nav_golden.py from the scalar
implementation that this module's kernels replaced.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pyarea import nav as nav_module
from pyarea.nav import AreaNav

DATA = Path(__file__).parent / "data"
FILES = ["goes8_gvar_nav.json", "goes13_gvar_nav.json"]


@pytest.fixture(params=[False, True], autouse=True)
def _disable_jit(request, monkeypatch):
    """Run every test with the numba kernels compiled and as plain Python (what coverage can see)."""
    if request.param:
        for name, obj in vars(nav_module).items():
            if name.startswith("_gvar_") and hasattr(obj, "py_func"):
                monkeypatch.setattr(nav_module, name, obj.py_func)


def load(name):
    return json.loads((DATA / name).read_text())


def make_nav(rec):
    nav = AreaNav(navblock=tuple(rec["nav_words"]), byte_order=rec["byte_order"], int_format=rec["int_format"])
    nav.setImageStart(rec["line_ul"], rec["element_ul"])
    nav.setRes(rec["line_res"], rec["element_res"])
    nav.setMag(1, 1)
    nav.setStart(0, 0)
    return nav


def nan_list(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


@pytest.fixture(params=FILES)
def case(request):
    rec = load(request.param)
    return rec, make_nav(rec)


def test_is_gvar_nav(case):
    _, nav = case
    assert type(nav).__name__ == "GVARNav"


def test_to_latlon_matches_recorded(case):
    rec, nav = case
    g = rec["to_latlon"]
    lat, lon = nav.toLatLon([g["elements"], g["lines"]])
    np.testing.assert_allclose(np.asarray(lat, dtype=float), nan_list(g["lat"]), rtol=0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(lon, dtype=float), nan_list(g["lon"]), rtol=0, atol=1e-8)


def test_to_linele_matches_recorded(case):
    rec, nav = case
    g = rec["to_linele"]
    elems, lines = nav.toLinEle([g["lat"], g["lon"]])
    np.testing.assert_allclose(np.asarray(elems, dtype=float), nan_list(g["elements"]), rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(lines, dtype=float), nan_list(g["lines"]), rtol=0, atol=1e-6)


def test_subpoint(case):
    rec, nav = case
    np.testing.assert_allclose(nav.getSubpoint(), rec["subpoint"], atol=1e-9)


def test_round_trip(case):
    """Line/element -> lat/lon -> line/element returns the starting point, in file coordinates."""
    rec, nav = case
    lines, elems = np.meshgrid(np.linspace(1, rec["image_lines"], 25), np.linspace(1, rec["image_elements"], 25))
    lat, lon = nav.to_latlon(lines, elems)
    on_earth = np.isfinite(lat)
    assert on_earth.any()
    back_line, back_elem = nav.to_lin_ele(lat, lon)
    # a few hundredths of a pixel at the limb, where the line of sight grazes the surface
    np.testing.assert_allclose(back_line[on_earth], lines[on_earth], atol=0.05)
    np.testing.assert_allclose(back_elem[on_earth], elems[on_earth], atol=0.05)
    # (the forward and inverse transforms are separate approximations, so they agree to ~2e-4 pixels, not exactly)
    mid = on_earth & (np.abs(lat) < 60)
    np.testing.assert_allclose(back_line[mid], lines[mid], atol=1e-3)
    np.testing.assert_allclose(back_elem[mid], elems[mid], atol=1e-3)


def test_shapes_and_scalars(case):
    rec, nav = case
    line, elem = np.full((3, 4), rec["image_lines"] / 2), np.full((3, 4), rec["image_elements"] / 2)
    lat, lon = nav.to_latlon(line, elem)
    assert lat.shape == lon.shape == (3, 4)
    one_lat, one_lon = nav.to_latlon(line[0, 0], elem[0, 0])
    assert np.ndim(one_lat) == 0
    assert float(one_lat) == pytest.approx(lat[0, 0], abs=1e-12)
    # broadcasting a column of lines against a row of elements gives a grid
    grid_lat, _ = nav.to_latlon(np.array([[100.0], [200.0]]), np.array([500.0, 600.0, 700.0]))
    assert grid_lat.shape == (2, 3)


def test_nan_and_off_earth(case):
    rec, nav = case
    lat, lon = nav.to_latlon([np.nan, 1.0, 100.0], [100.0, np.nan, 100.0])
    assert np.isnan(lat[:2]).all() and np.isnan(lon[:2]).all()
    line, elem = nav.to_lin_ele([np.nan, 95.0, 0.0], [0.0, 0.0, np.nan])
    assert np.isnan(line).all() and np.isnan(elem).all()
    # the far side of the Earth is not visible from the satellite
    sub_lat, sub_lon = nav.getSubpoint()
    line, elem = nav.to_lin_ele([sub_lat], [sub_lon + 180.0])
    assert np.isnan(line).all() and np.isnan(elem).all()


def test_two_dimensional_variants_agree_with_the_list_versions(case):
    """toLinEle2 and toLatLon2 (used by pyarea.navigation) give the same answers as the list-based methods."""
    rec, nav = case
    g = rec["to_latlon"]
    lat, lon = nav.toLatLon([g["elements"], g["lines"]])
    lat2, lon2 = nav.toLatLon2(np.array(g["lines"]).reshape(12, 12), np.array(g["elements"]).reshape(12, 12))
    np.testing.assert_array_equal(lat2.ravel(), lat)
    np.testing.assert_array_equal(lon2.ravel(), lon)
    elems, lines = nav.toLinEle([lat, lon])
    lines2, elems2 = nav.toLinEle2(lat2, lon2)
    np.testing.assert_array_equal(lines2.ravel(), lines)
    np.testing.assert_array_equal(elems2.ravel(), elems)
    assert np.isnan(lines2[np.isnan(lat2)]).all()


def test_flipped_lines_round_trip(case):
    rec, nav = case
    nav.isLineFlipped = True
    nav.lineOffset = 10000.0
    lat, lon = nav.to_latlon([[1500.0]], [[1800.0]])
    if np.isnan(lat).all():  # keep the point on the Earth for either file
        lat, lon = nav.to_latlon([[9000.0]], [[1800.0]])
        line_in = 9000.0
    else:
        line_in = 1500.0
    assert np.isfinite(lat).all()
    line, elem = nav.to_lin_ele(lat, lon)
    assert line[0, 0] == pytest.approx(line_in, abs=1e-3)
    assert elem[0, 0] == pytest.approx(1800.0, abs=1e-3)
    # the base class's array helper must agree with the scalar one
    base_line, base_elem = nav.imageCoordToAreaCoord_np(linele=[np.array([2900.0]), np.array([9100.0])])
    scalar = nav.imageCoordToAreaCoord([[9100.0], [2900.0]])
    assert base_line[0] == pytest.approx(scalar[nav.indexLine][0])
    assert base_elem[0] == pytest.approx(scalar[nav.indexEle][0])


def test_full_image_is_fast(case):
    """A whole image's worth of points takes seconds, not minutes, once compiled."""
    import time

    if not hasattr(nav_module._gvar_to_latlon, "py_func"):
        pytest.skip("the pure Python run of the kernels is too slow for a whole image")
    rec, nav = case
    nav.to_latlon(np.array([1.0]), np.array([1.0]))  # compile
    lines, elems = np.meshgrid(np.arange(1, rec["image_lines"] + 1.0), np.arange(1, rec["image_elements"] + 1.0),
                               indexing="ij")
    start = time.perf_counter()
    lat, lon = nav.to_latlon(lines, elems)
    elapsed = time.perf_counter() - start
    assert lat.shape == lines.shape
    assert elapsed < 10, f"{lines.size} points took {elapsed:.1f} s"
