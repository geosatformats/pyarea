"""LALONav, navigation from tables of latitude and longitude in the aux block (nav type LALO), against LALOnav.java."""
import struct

import numpy as np
import pytest

from pyarea.file import AreaFile
from pyarea.nav import AreaNav, LALONav
from tests.fake_area_files import GOES7_IR, FakeArea, mode_a_data, write_area_file

ROWS, COLS = 5, 6
ULLINE, ULELEM, LATRES, LONRES = 1000, 2000, 8, 4


def make_tables(rows=ROWS, cols=COLS, lon0=70.0):
    """A smooth, slightly sheared grid: latitude falls with the row, longitude grows with the column."""
    i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    return (20.0 - 0.5 * i + 0.02 * j).astype(np.float32), (lon0 + 0.6 * j + 0.03 * i).astype(np.float32)


def float_word(value):
    return struct.unpack(">i", struct.pack(">f", value))[0]


def lalo_blocks(lat, lon, ulline=ULLINE, ulelem=ULELEM, latres=LATRES, lonres=LONRES, lat_offset=0, order=">"):
    """The 128 words of a LALO nav block and the aux block that holds the two tables (as 32-bit words)."""
    rows, cols = lat.shape
    nav = np.zeros(128, dtype=np.int64)
    nav[0] = int.from_bytes(b"LALO", "big" if order == ">" else "little")  # the word as read in the file's order
    nav[65], nav[66] = rows, cols
    nav[67:71] = [float_word(float(lat.min())), float_word(float(lon.min())),
                  float_word(float(lat.max())), float_word(float(lon.max()))]
    nav[71], nav[72], nav[75], nav[76] = latres, lonres, ulline, ulelem
    n = rows * cols
    nav[77], nav[78], nav[79] = 2 * n * 4, lat_offset, lat_offset + n * 4
    aux = np.zeros(lat_offset // 4 + 2 * n, dtype=np.int32)
    aux[lat_offset // 4:lat_offset // 4 + n] = lat.ravel().view(np.int32)
    aux[lat_offset // 4 + n:] = lon.ravel().view(np.int32)
    return tuple(int(w) for w in nav), tuple(int(w) for w in aux)


def make_nav(lat=None, lon=None, order=">", **kwargs):
    if lat is None:
        lat, lon = make_tables()
    nav_words, aux_words = lalo_blocks(lat, lon, order=order, **kwargs)
    nav = AreaNav(navblock=nav_words, auxblock=aux_words, byte_order=order, int_format="i")
    nav.setImageStart(0, 0)
    nav.setRes(1, 1)
    nav.setMag(1, 1)
    nav.setStart(0, 0)
    return nav


def java_latlon(lat, lon, rlin, rele, ulline=ULLINE, ulelem=ULELEM, latres=LATRES, lonres=LONRES):
    """LALOnav.toLatLon(double[][]) for one point, in the same order and in float arithmetic."""
    rows, cols = lat.shape
    lrlin, lrele = ulline + (rows - 1) * latres, ulelem + (cols - 1) * lonres
    if rlin < ulline or rlin > lrlin or rele < ulelem or rele > lrele:
        return np.nan, np.nan
    row, col = (int(rlin) - ulline) // latres, (int(rele) - ulelem) // lonres
    tl = row * cols + col
    tr, bl = tl + 1, tl + cols
    br = bl + 1
    if row >= rows - 1:
        bl, br = tl, tl + 1
    if col >= cols - 1:
        tr, br = tl, bl
    flat_lat, flat_lon = lat.ravel(), lon.ravel()
    if any(flat_lat[k] == 0 and flat_lon[k] == 0 for k in (tl, tr, bl, br)):
        return np.nan, np.nan
    frac_row = np.float32((int(rlin) - ulline) % latres) / np.float32(latres)
    frac_col = np.float32((int(rele) - ulelem) % lonres) / np.float32(lonres)
    out = []
    for table in (flat_lat, flat_lon):
        ax, bx, cx = table[tr] - table[tl], table[bl] - table[tl], (table[tl] + table[br]) - (table[bl] + table[tr])
        out.append(float(ax * frac_col + bx * frac_row + cx * frac_row * frac_col + table[tl]))
    return tuple(out)


def test_the_block_is_read_in_either_byte_order():
    lat, lon = make_tables()
    for order in (">", "<"):
        nav = make_nav(lat, lon, order=order)
        assert type(nav) is LALONav
        assert (nav.rows, nav.cols, nav.latres, nav.lonres, nav.ulline, nav.ulelem) == (
            ROWS, COLS, LATRES, LONRES, ULLINE, ULELEM)
        assert (nav.minlat, nav.maxlon) == (pytest.approx(float(lat.min())), pytest.approx(float(lon.max())))
        np.testing.assert_array_equal(nav.latData, lat.ravel())


def test_the_tables_can_start_anywhere_in_the_aux_block():
    lat, lon = make_tables()
    nav = make_nav(lat, lon, lat_offset=40)
    np.testing.assert_array_equal(nav.lonData, lon.ravel())


def test_bad_blocks_are_rejected():
    lat, lon = make_tables()
    nav_words, aux_words = lalo_blocks(lat, lon)
    with pytest.raises(ValueError, match="needs the aux block"):
        AreaNav(navblock=nav_words, byte_order=">", int_format="i")
    with pytest.raises(ValueError, match="80 are needed"):
        AreaNav(navblock=nav_words[:60], auxblock=aux_words, byte_order=">", int_format="i")
    with pytest.raises(ValueError, match="do not fit"):
        AreaNav(navblock=nav_words, auxblock=aux_words[:-3], byte_order=">", int_format="i")
    for index, value, message in ((65, 0, "0 rows"), (71, 0, "spacing"), (72, -4, "spacing")):
        bad = list(nav_words)
        bad[index] = value
        with pytest.raises(ValueError, match=message):
            AreaNav(navblock=tuple(bad), auxblock=aux_words, byte_order=">", int_format="i")
    nav = LALONav.__new__(LALONav, navblock=nav_words, auxblock=aux_words, byte_order=">", int_format="i")
    nav.navtype = "GVAR"
    with pytest.raises(ValueError, match="Invalid navigation type"):
        nav.__init__(navblock=nav_words, auxblock=aux_words)


def test_the_tie_points_are_the_tables():
    lat, lon = make_tables()
    nav = make_nav(lat, lon)
    lines, elems = np.meshgrid(ULLINE + LATRES * np.arange(ROWS), ULELEM + LONRES * np.arange(COLS), indexing="ij")
    got_lat, got_lon = nav.to_latlon(lines, elems)
    np.testing.assert_array_equal(got_lat, lat.astype(float))
    np.testing.assert_array_equal(got_lon, lon.astype(float))


def test_to_latlon_is_the_java_algorithm_pixel_by_pixel():
    lat, lon = make_tables()
    nav = make_nav(lat, lon)
    lines, elems = np.meshgrid(np.arange(ULLINE - 3, ULLINE + 4 * LATRES + 4),
                               np.arange(ULELEM - 2, ULELEM + 5 * LONRES + 3), indexing="ij")
    got_lat, got_lon = nav.to_latlon(lines, elems)
    expected = np.array([[java_latlon(lat, lon, float(lines[r, c]), float(elems[r, c])) for c in range(lines.shape[1])]
                         for r in range(lines.shape[0])])
    np.testing.assert_array_equal(got_lat, expected[..., 0])
    np.testing.assert_array_equal(got_lon, expected[..., 1])
    assert np.isnan(got_lat[0, 0]) and np.isfinite(got_lat[5, 5])  # outside the table: NaN


def test_the_line_and_element_are_truncated_to_whole_pixels():
    """The Java takes (int) of the image line and element, so the fractional part of a pixel is dropped."""
    nav = make_nav()
    whole = nav.to_latlon(ULLINE + 11.0, ULELEM + 6.0)
    assert nav.to_latlon(ULLINE + 11.9, ULELEM + 6.6) == whole


def test_the_edges_of_the_table_use_the_neighbours_that_exist():
    lat, lon = make_tables()
    nav = make_nav(lat, lon)
    last_line, last_elem = ULLINE + (ROWS - 1) * LATRES, ULELEM + (COLS - 1) * LONRES
    corner = nav.to_latlon(last_line, last_elem)
    assert corner == (float(lat[-1, -1]), float(lon[-1, -1]))
    between = nav.to_latlon(last_line, last_elem - 2)  # last row, half way between the last two columns
    assert between[0] == pytest.approx(float(lat[-1, -2] + lat[-1, -1]) / 2, abs=1e-5)
    assert between[1] == pytest.approx(float(lon[-1, -2] + lon[-1, -1]) / 2, abs=1e-5)
    assert np.isnan(nav.to_latlon(last_line + 1, last_elem)[0])  # one line past the table


def test_missing_tie_points_blank_the_pixels_next_to_them():
    lat, lon = make_tables()
    lat[2, 3] = 95.0  # not a latitude: stored as 0, 0 = missing
    nav = make_nav(lat, lon)
    assert nav.latData[2 * COLS + 3] == 0 and nav.lonData[2 * COLS + 3] == 0
    for row, col in ((1, 2), (1, 3), (2, 2), (2, 3)):  # the four cells that have this tie point as a corner
        assert np.isnan(nav.to_latlon(ULLINE + row * LATRES + 1, ULELEM + col * LONRES + 1)[0])
    assert np.isfinite(nav.to_latlon(ULLINE + 0 * LATRES, ULELEM + 0 * LONRES)[0])
    assert np.isfinite(nav.to_latlon(ULLINE + 3 * LATRES, ULELEM + 4 * LONRES)[0])
    lon_bad = make_tables()
    lon_bad[1][0, 0] = 400.0  # not a longitude (the limit is 360)
    assert np.isnan(make_nav(*lon_bad).to_latlon(ULLINE, ULELEM)[0])


def test_nan_and_the_area_to_image_conversion():
    nav = make_nav()
    assert np.isnan(nav.to_latlon(np.nan, ULELEM)[0]) and np.isnan(nav.to_latlon(ULLINE, np.nan)[1])
    nav.setRes(2, 2)
    nav.setImageStart(ULLINE, ULELEM)
    inner = make_nav().to_latlon(ULLINE + 20.0, ULELEM + 8.0)
    assert nav.to_latlon(10.0, 4.0) == inner  # image line ULLINE + 2 * 10, element ULELEM + 2 * 4


def test_the_round_trip_gives_back_the_pixel():
    nav = make_nav()
    lines, elems = np.meshgrid(ULLINE + np.arange(0, 4 * LATRES + 1, 3), ULELEM + np.arange(0, 5 * LONRES + 1, 2),
                               indexing="ij")
    lat, lon = nav.to_latlon(lines, elems)
    back_lines, back_elems = nav.to_lin_ele(lat, lon)
    np.testing.assert_allclose(back_lines, lines, atol=2e-3)
    np.testing.assert_allclose(back_elems, elems, atol=2e-3)


def test_to_lin_ele_is_nan_off_the_table_and_for_nan():
    nav = make_nav()
    lat, lon = nav.to_latlon(ULLINE + 10.0, ULELEM + 5.0)
    line, elem = nav.to_lin_ele(np.array([lat, lat + 20.0, np.nan, lat]), np.array([lon, lon, lon, np.nan]))
    assert np.isfinite(line[0]) and np.isfinite(elem[0])
    assert np.isnan(line[1:]).all() and np.isnan(elem[1:]).all()


def test_to_lin_ele_skips_the_missing_cells_and_wraps_the_longitude():
    lat, lon = make_tables(lon0=170.0)  # longitudes 170 to 173.15
    nav = make_nav(lat, lon)
    pixel = nav.to_latlon(ULLINE + 10.0, ULELEM + 6.0)
    line, elem = nav.to_lin_ele(pixel[0], pixel[1] - 360.0)
    assert (line, elem) == (pytest.approx(ULLINE + 10.0, abs=2e-3), pytest.approx(ULELEM + 6.0, abs=2e-3))
    lat[2, 3] = 95.0  # a missing tie point: the grid there is unknown, so there is no pixel for it
    gap = make_nav(lat, lon).to_lin_ele(float(make_tables(lon0=170.0)[0][2, 3]), float(lon[2, 3]))
    assert np.isnan(gap[0])


def test_the_list_interface_has_elements_first_like_visad():
    nav = make_nav()
    lat, lon = nav.to_latlon(np.array([ULLINE + 10.0]), np.array([ULELEM + 6.0]))
    out = nav.toLatLon([[ULELEM + 6.0], [ULLINE + 10.0]])  # [[elements], [lines]]
    assert out[0][0] == lat[0] and out[1][0] == lon[0]
    line_elem = nav.toLinEle(out)
    assert line_elem[nav.indexLine][0] == pytest.approx(ULLINE + 10.0, abs=2e-3)
    assert line_elem[nav.indexEle][0] == pytest.approx(ULELEM + 6.0, abs=2e-3)
    assert nav.toLatLon2(np.array([ULLINE + 10.0]), np.array([ULELEM + 6.0]))[0][0] == lat[0]


@pytest.mark.parametrize("endian", [">", "<"])
def test_a_file_is_located_through_its_aux_block(tmp_path, endian):
    """AreaFile hands the aux block to the navigation, in either byte order, and get_lonlats uses it."""
    lat, lon = make_tables(16, 24)
    nav_words, aux_words = lalo_blocks(lat, lon, ulline=1000, ulelem=2000, latres=8, lonres=4)
    spec = FakeArea(GOES7_IR, mode_a_data([100]), source_type="LALO", ul_line=1000, ul_element=2000, line_res=8,
                    elem_res=4, nav_words=np.array(nav_words), aux_words=np.array(aux_words, dtype=np.int32),
                    endian=endian)
    area = AreaFile(write_area_file(tmp_path / "lalo.area", spec))
    got_lon, got_lat = area.get_lonlats()
    np.testing.assert_array_equal(got_lat, lat.astype(float))
    np.testing.assert_array_equal(got_lon, lon.astype(float))
