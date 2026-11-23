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
from pyarea.nav import AreaNav, GVARNav

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
    # the base class's array helper must agree with the list one
    base_line, base_elem = nav._image_to_area(np.array([2900.0]), np.array([9100.0]))
    listed = nav.imageCoordToAreaCoord([[9100.0], [2900.0]])
    assert base_line[0] == pytest.approx(listed[nav.indexLine][0])
    assert base_elem[0] == pytest.approx(listed[nav.indexEle][0])


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


# IMC active and IMC off take different paths through the constructor; the second is not exercised by real files here
# so both are compared with an independent port of VisAD's GVARnav.java, with the IMC flag cleared for the second.
@pytest.mark.parametrize("imc_off", [False, True])
@pytest.mark.parametrize("name", ["goes8_gvar_nav.json", "goes13_gvar_nav.json"])
def test_to_latlon_matches_visad_port(name, imc_off):
    from tests.visad_gvarnav_reference import Ref

    rec = load(name)
    words = list(rec["nav_words"])
    if imc_off:
        words[2] &= ~(1 << 7)  # IMCFLG: clear the "IMC active" bit
    nav = make_nav(dict(rec, nav_words=words))
    ref = Ref(words, rec["byte_order"])
    assert nav.imc == ref.imc == (1 if imc_off else 0)
    if imc_off:
        assert nav.ts == pytest.approx(ref.ts, abs=1e-6)  # minutes from the orbit epoch to the image time
    compared = 0
    for line in np.linspace(100, 1500, 6):
        for elem in np.linspace(100, 3000, 6):
            image_line, image_elem = nav._area_to_image(line, elem)
            lat, lon = nav.to_latlon(np.array(line), np.array(elem))
            ref_lat, ref_lon = ref.to_latlon(float(image_line), float(image_elem))
            assert np.isnan(lat) == np.isnan(ref_lat)
            if np.isfinite(lat):
                assert lat == pytest.approx(ref_lat, abs=1e-4)
                assert lon == pytest.approx(ref_lon, abs=1e-4)
                compared += 1
    assert compared > 20


def test_navs_of_different_files_do_not_share_state():
    """Building a second GVARNav must not change the first; the satellite state is per instance."""
    first = make_nav(load("goes8_gvar_nav.json"))
    line, elem = np.array([500.0, 800.0]), np.array([1500.0, 2500.0])
    lat, lon = first.to_latlon(line, elem)
    second = make_nav(load("goes13_gvar_nav.json"))
    again_lat, again_lon = first.to_latlon(line, elem)
    np.testing.assert_array_equal(again_lat, lat)
    np.testing.assert_array_equal(again_lon, lon)
    assert second.xs is not first.xs and second.bt is not first.bt
    assert second.elvmax is not first.elvmax and second.scnmax is not first.scnmax
    assert np.isnan(GVARNav.bt).all()  # the class level defaults were never filled in
    assert GVARNav.elvmax == [0.220896, 0.22089375]


def test_area_nav_codes_are_the_words_of_visads_area_nav():
    """The code constants are the four characters of the first nav word, as in AREAnav.java."""
    java = {"DMSP": 0x444D5250, "GMSX": 0x474D5358, "GOES": 0x474F4553, "GEOS": 0x47454F53, "GVAR": 0x47564152,
            "MOLL": 0x4D4F4C4C, "MSAT": 0x4D534154, "MSGT": 0x4D534754, "MSG": 0x4D534720, "POES": 0x5449524F,
            "RADR": 0x52414452, "RECT": 0x52454354, "PS": 0x50532020, "MERC": 0x4D455243, "TANC": 0x54414E43,
            "SIN": 0x53494E20, "LAMB": 0x4C414D42, "LALO": 0x4C414C4F, "KALP": 0x4B414C50, "ABIS": 0x41424953,
            "ABIN": 1094863182}
    for name, word in java.items():
        assert getattr(AreaNav, name) == word.to_bytes(4, "big").decode("ascii"), name


def test_flipped_line_coordinates_api():
    nav = make_nav(load("goes8_gvar_nav.json"))
    assert nav.isFlippedLineCoordinates() is False
    nav.setFlipLineCoordinates(10000)
    assert nav.isFlippedLineCoordinates() is True and nav.getLineOffset() == 10000.0
    line, elem = nav._image_to_area(np.array([2900.0]), np.array([9100.0]))
    base = make_nav(load("goes8_gvar_nav.json"))._image_to_area(np.array([2900.0]), np.array([9100.0]))
    assert line[0] == pytest.approx(10000.0 - base[0][0]) and elem[0] == base[1][0]


def test_navigations_without_a_subpoint_report_nan():
    lat, lon = AreaNav.getSubpoint(make_nav(load("goes8_gvar_nav.json")))
    assert np.isnan(lat) and np.isnan(lon)
