"""Tests for reading and calibrating pre-GVAR SMS/GOES AREA files with ``AreaFile``.

The files are small fake ones; see ``fake_area_files`` for how they are made. The fixtures
(``mode_a_ir_file``, ``mode_a_vis_file``, ``mode_aa_file``, ``goes7_aaa_file``) are in ``conftest.py``.
"""

import logging

import numpy as np
import pytest

from pyarea.file import AreaFile
from tests.fake_area_files import (
    AAA_AB1,
    AAA_AB2,
    AAA_COUNTS,
    AAA_IFAB,
    CAL_WORDS,
    DIR_WORDS,
    GOES3_IR,
    GOES5_IR,
    GOES6_IR,
    GOES7_IR,
    IR_WINDOW,
    NADIR_COL,
    NADIR_ROW,
    NAV_WORDS,
    SHAPE,
    SUB_LON,
    W_CAL_OFFSET,
    W_NAV_OFFSET,
    FakeArea,
    aaa_spec,
    make_nav_words,
    mode_a_data,
    mode_aa_spec,
    patch_dir_word,
    planck_temperature,
    write_area_file,
)


class TestDirectory:
    """What the directory says about pre-GVAR sensors."""

    def test_satellite_and_band(self, mode_a_ir_file, mode_a_vis_file):
        """SMS and GOES-1..7 sensors name their platform and band."""
        ir = AreaFile(mode_a_ir_file).directory
        vis = AreaFile(mode_a_vis_file).directory
        assert (ir.satellite, ir.band, ir.sensor) == ("GOES-3", "IR", "GOES-3")
        assert (vis.satellite, vis.band) == ("GOES-3", "VIS")

    def test_source_and_calibration_type(self, mode_a_ir_file):
        """Both are 4 character codes."""
        area = AreaFile(mode_a_ir_file)
        assert area.source_type == "VISR"
        assert area.cal_type == "BRIT"

    @pytest.mark.parametrize("endian", [">", "<"])
    def test_byte_order_is_detected(self, tmp_path, endian):
        """Read big and little endian files alike."""
        spec = FakeArea(GOES3_IR, mode_a_data([100, 200]), endian=endian, **IR_WINDOW)
        area = AreaFile(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        assert area.directory.byte_order == endian
        np.testing.assert_array_equal(area.read_data()[0, :2], [100, 200])
        lon, _ = area.get_lonlats()
        assert lon[NADIR_ROW, NADIR_COL] == pytest.approx(SUB_LON, abs=0.1)

    def test_two_byte_data_follows_the_byte_order(self, tmp_path):
        """Two byte counts are read in the byte order of the file, as native integers."""
        spec = aaa_spec(GOES7_IR)
        spec.endian = "<"
        spec.data = np.where(np.arange(3)[None, None, :] == 1, 3000, np.full((SHAPE[0], SHAPE[1], 3), 1000))
        area = AreaFile(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec))
        counts = area.read_channel(7)[0]
        assert counts.max() == 3000
        assert counts.dtype.isnative

    def test_truncated_file_uses_the_lines_present(self, tmp_path, caplog):
        """Data and navigation use the number of lines that are really there."""
        spec = FakeArea(GOES3_IR, mode_a_data([100]), claimed_lines=SHAPE[0] + 3, **IR_WINDOW)
        with caplog.at_level(logging.WARNING):
            area = AreaFile(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        assert "directory claims" in caplog.text
        assert area.lines == area.actual_lines == SHAPE[0]
        assert area.data.shape == (1, SHAPE[0], SHAPE[1])
        assert area.get_lonlats()[0].shape == SHAPE


class TestModeA:
    """Single-band VISR files."""

    def test_brightness_temperature(self, mode_a_ir_file):
        """Apply the documented two-piece VISSR formula."""
        counts = np.array([0, 100, 175, 176, 200, 255])
        expected = np.where(counts >= 176, 418.0 - counts, 330.0 - counts / 2.0)
        temperature = AreaFile(mode_a_ir_file).read_ir_temperature_a()
        assert temperature.dtype == np.float32
        np.testing.assert_allclose(temperature[NADIR_ROW, :6], expected)

    def test_visible_counts(self, mode_a_vis_file):
        """Return the raw counts."""
        np.testing.assert_array_equal(AreaFile(mode_a_vis_file).read_vis()[NADIR_ROW, :6], [0, 4, 128, 252, 252, 255])

    def test_unnormalized_reflectance(self, mode_a_vis_file):
        """Scale the 6-bit counts to percent and clip to 0-100."""
        expected = np.clip(np.array([0, 4, 128, 252, 252, 255]) / 252.0 * 100.0, 0, 100)
        res = AreaFile(mode_a_vis_file).read_vis("unnormalized_reflectance")
        np.testing.assert_allclose(res[NADIR_ROW, :6], expected, rtol=1e-6)

    def test_single_band_read_of_a_multiband_file(self, goes7_aaa_file):
        """Multiband files are read per channel."""
        with pytest.raises(ValueError, match="single-band"):
            AreaFile(goes7_aaa_file).read_data()

    def test_temperature_of_a_visible_file(self, mode_a_vis_file):
        """The IR formula does not apply to visible data."""
        with pytest.raises(ValueError, match="not IR"):
            AreaFile(mode_a_vis_file).read_ir_temperature_a()

    def test_temperature_of_a_multiband_file(self, goes7_aaa_file):
        """The single-band formula does not apply to multiband files."""
        with pytest.raises(NotImplementedError, match="multiband"):
            AreaFile(goes7_aaa_file).read_ir_temperature_a()

    def test_temperature_of_wide_counts(self, tmp_path):
        """The formula assumes 8-bit counts."""
        spec = FakeArea(GOES3_IR, np.full((SHAPE[0], SHAPE[1], 1), 300), bytes_per_elem=2, **IR_WINDOW)
        area = AreaFile(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        with pytest.raises(ValueError, match="8 bits"):
            area.read_ir_temperature_a()

    def test_visible_counts_of_an_infrared_file(self, mode_a_ir_file):
        """Visible counts only exist for visible files."""
        with pytest.raises(ValueError, match="not VIS"):
            AreaFile(mode_a_ir_file).read_vis()

    def test_unsupported_visible_calibration(self, mode_a_vis_file):
        """Only counts and unnormalized reflectance are available."""
        with pytest.raises(ValueError, match="Unsupported VIS calibration"):
            AreaFile(mode_a_vis_file).read_vis("bogus")

    def test_unsupported_element_size(self, tmp_path):
        """Only one and two byte elements are supported."""
        spec = FakeArea(GOES3_IR, mode_a_data([100]), bytes_per_elem=4, **IR_WINDOW)
        area = AreaFile(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec))
        with pytest.raises(NotImplementedError, match="bytes per element"):
            area.read_data()


class TestModeAA:
    """VAS files with per-line calibration entries in the line prefix."""

    def test_brightness_temperature(self, mode_aa_file):
        """Subtract the offset, scale to radiance and invert the Planck function."""
        corrected = 10000 - 400  # already a multiple of 8
        expected = planck_temperature(corrected * 2.0 ** (0 - 7), (8512, 1286, 0.3212, 0.9974))
        res = AreaFile(mode_aa_file).read_ir_temperature_aa(8)
        assert res.dtype == np.float32
        assert res[NADIR_ROW, NADIR_COL] == pytest.approx(expected, abs=1e-3)
        assert 200 < expected < 330

    def test_channel_missing_on_all_lines_is_nan(self, mode_aa_file):
        """A band without calibration entry has no data."""
        assert np.isnan(AreaFile(mode_aa_file).read_ir_temperature_aa(7)).all()

    def test_zero_counts_are_nan(self, tmp_path):
        """Counts of zero are the no-data value."""
        area = AreaFile(write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", mode_aa_spec(raw=0)))
        assert np.isnan(area.read_ir_temperature_aa(8)).all()

    def test_request_for_another_format(self, mode_a_ir_file):
        """Mode AA temperatures only exist for Mode AA files."""
        with pytest.raises(ValueError, match="not a Mode AA"):
            AreaFile(mode_a_ir_file).read_ir_temperature_aa(8)

    def test_band_out_of_range(self, mode_aa_file):
        """The visible channel is not covered by the Mode AA formula."""
        with pytest.raises(ValueError, match="1-12"):
            AreaFile(mode_aa_file).read_ir_temperature_aa(13)

    def test_short_calibration_region(self, tmp_path, caplog):
        """A file without room for the calibration entries cannot be calibrated."""
        spec = mode_aa_spec()
        spec.cal_len = 50
        area = AreaFile(write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", spec))
        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="no Mode AA calibration"):
            area.read_ir_temperature_aa(8)
        assert "Mode AA calibration region is only 50 bytes" in caplog.text

    def test_without_band_list(self, tmp_path, mode_aa_file, caplog):
        """A single-band file with a short prefix is read without the band list."""
        spec = mode_aa_spec()
        spec.prefixes = spec.prefixes[:, :632]
        area = AreaFile(write_area_file(tmp_path / "goes05.1983.229.2231-2.AREA_IR", spec))
        with caplog.at_level(logging.WARNING):
            assert area.band_list is None
        assert "too short for a per-line band list" in caplog.text
        np.testing.assert_allclose(area.read_ir_temperature_aa(8),
                                   AreaFile(mode_aa_file).read_ir_temperature_aa(8))


class TestModeAAA:
    """VAS files with a calibration block and a per-line band list."""

    @pytest.mark.parametrize(("channel", "row"), [(8, 0), (8, 1), (7, 0), (7, 1), (10, 1)])
    def test_counts_follow_the_per_line_band_list(self, goes7_aaa_file, channel, row):
        """Find each channel in whichever band slot it is on that line."""
        counts, present = AreaFile(goes7_aaa_file).read_channel(channel)
        assert present[row]
        assert (counts[row] == AAA_COUNTS[channel]).all()

    def test_channel_codes_agree_with_the_prefix_groups(self, goes7_aaa_file):
        """The redundant channel groups in the prefix are available as a cross-check."""
        area = AreaFile(goes7_aaa_file)
        assert area.line_channel_spin["channel"].shape == (SHAPE[0], 13)
        assert area.band_list.shape == (SHAPE[0], 3)

    def test_radiance(self, goes7_aaa_file):
        """Apply the radiance equation of the calibration block."""
        expected = (AAA_AB2 * AAA_COUNTS[8] / 32.0 - AAA_AB1) / 2.0 ** (15 - AAA_IFAB)
        np.testing.assert_allclose(AreaFile(goes7_aaa_file).read_ir_radiance(8)[NADIR_ROW], expected, rtol=1e-6)

    @pytest.mark.parametrize(
        ("sensor_source", "constants"),
        [
            (GOES6_IR, (8438.0, 1283.0, 0.3263, 0.9973)),
            (GOES7_IR, (8525.0, 1287.0, 0.3408, 0.9973)),
        ])
    def test_brightness_temperature_uses_satellite_constants(self, tmp_path, sensor_source, constants):
        """GOES-6 and GOES-7 have their own Planck constants."""
        area = AreaFile(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", aaa_spec(sensor_source)))
        expected = planck_temperature(AAA_COUNTS[8], constants)
        assert area.read_ir_temperature_aaa(8)[NADIR_ROW, NADIR_COL] == pytest.approx(expected, abs=1e-3)

    def test_satellite_without_constants(self, tmp_path):
        """Do not guess constants for satellites that have none."""
        area = AreaFile(write_area_file(tmp_path / "goes05.1992.236.1301.AREA_IR", aaa_spec(GOES5_IR)))
        with pytest.raises(NotImplementedError, match="GOES-6 and GOES-7"):
            area.read_ir_temperature_aaa(8)

    def test_channel_without_planck_constants(self, goes7_aaa_file):
        """Refuse channels that have no verified Planck constants."""
        with pytest.raises(NotImplementedError, match="No verified Planck"):
            AreaFile(goes7_aaa_file).read_ir_temperature_aaa(14)

    def test_channel_without_calibration(self, goes7_aaa_file):
        """Refuse channels that the calibration block does not describe."""
        with pytest.raises(ValueError, match="not in the calibration block"):
            AreaFile(goes7_aaa_file).read_ir_radiance(39)

    def test_channel_absent_on_some_lines_is_nan(self, tmp_path):
        """Lines that do not carry the channel have no data."""
        spec = aaa_spec(GOES7_IR)
        spec.prefixes[1, 632:632 + 3] = (7, 10, 0)  # channel 8 is missing on odd lines
        res = AreaFile(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec)).read_ir_temperature_aaa(8)
        assert np.isfinite(res[0]).all()
        assert np.isnan(res[1]).all()

    def test_band_list_shorter_than_the_number_of_bands(self, tmp_path, caplog):
        """Without a usable band list the channels cannot be told apart."""
        spec = aaa_spec(GOES7_IR)
        spec.prefixes = spec.prefixes[:, :633]
        area = AreaFile(write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", spec))
        with caplog.at_level(logging.WARNING), pytest.raises(ValueError, match="band list"):
            area.read_channel(8)
        assert "too short for a per-line band list" in caplog.text

    def test_short_calibration_block(self, tmp_path):
        """A calibration block that is too short gives no radiance calibration."""
        path = write_area_file(tmp_path / "goes07.1992.236.1301.AREA_IR", aaa_spec(GOES7_IR))
        patch_dir_word(path, W_CAL_OFFSET, (DIR_WORDS + NAV_WORDS + CAL_WORDS) * 4 - 40)
        area = AreaFile(path)
        assert area.aaa_cal is None
        with pytest.raises(ValueError, match="calibration block"):
            area.read_ir_radiance(8)


class TestNavigation:
    """Earth location through the file."""

    def test_nadir(self, mode_a_ir_file):
        """The pixel nearest nadir is at the sub-satellite longitude and on the equator."""
        lon, lat = AreaFile(mode_a_ir_file).get_lonlats()
        assert lon[NADIR_ROW, NADIR_COL] == pytest.approx(SUB_LON, abs=0.1)
        assert lat[NADIR_ROW, NADIR_COL] == pytest.approx(0.0, abs=0.1)

    def test_subset_of_the_grid(self, mode_a_ir_file):
        """A subset of lines and elements is located like the full grid."""
        area = AreaFile(mode_a_ir_file)
        lon, lat = area.get_lonlats()
        sub_lon, sub_lat = area.get_lonlats(lines=np.array([2, 5]), elems=np.array([3, 7, 9]))
        np.testing.assert_allclose(sub_lon, lon[np.ix_([2, 5], [3, 7, 9])])
        np.testing.assert_allclose(sub_lat, lat[np.ix_([2, 5], [3, 7, 9])])

    def test_navigation_object(self, mode_a_ir_file):
        """``AreaFile.navigation`` hands out the GOES navigation, set up for this file."""
        nav = AreaFile(mode_a_ir_file).navigation
        assert type(nav).__name__ == "GOESNav"
        assert nav.getSubpoint()[1] == pytest.approx(SUB_LON, abs=0.01)

    def test_nadir_sampling(self, mode_a_ir_file):
        """The nadir spacing follows the line and element resolution."""
        sampling = AreaFile(mode_a_ir_file).get_nadir_sampling()
        assert sampling["sub_lon"] == pytest.approx(SUB_LON, abs=0.01)

    def test_no_orbital_elements(self, tmp_path):
        """Pre-1980 files can have a zeroed navigation block."""
        nav = make_nav_words()
        nav[6:12] = 0
        spec = FakeArea(GOES3_IR, mode_a_data([100]), nav_words=nav, **IR_WINDOW)
        with pytest.raises(ValueError, match="orbital"):
            AreaFile(write_area_file(tmp_path / "goes03.1979.234.1515.AREA_IR", spec)).get_lonlats()

    def test_no_nav_block(self, mode_a_ir_file):
        """Without a nav block there is nothing to earth-locate with."""
        patch_dir_word(mode_a_ir_file, W_NAV_OFFSET, 0)
        area = AreaFile(mode_a_ir_file)
        assert area.nav is None
        assert area.navigation is None
        with pytest.raises(ValueError, match="no GOES navigation"):
            area.get_lonlats()
        with pytest.raises(ValueError, match="no GOES navigation"):
            area.get_nadir_sampling()

    def test_nav_block_of_another_type(self, mode_a_ir_file):
        """A nav block of an unknown type gives no navigation."""
        with open(mode_a_ir_file, "r+b") as f:
            f.seek(DIR_WORDS * 4)
            f.write(b"XXXX")
        area = AreaFile(mode_a_ir_file)
        assert area.navigation is None
        with pytest.raises(ValueError, match="no GOES navigation"):
            area.get_lonlats()


class TestMalformedFiles:
    """Broken directories."""

    def test_undetectable_byte_order_warns(self, tmp_path, caplog):
        """Fall back to big endian, with a warning, when neither byte order looks right."""
        spec = FakeArea(99, mode_a_data([100]), **IR_WINDOW)
        with caplog.at_level(logging.WARNING):
            area = AreaFile(write_area_file(tmp_path / "unknown.AREA_IR", spec))
        assert "Could not confidently detect the AREA byte order" in caplog.text
        assert area.directory.byte_order == ">"
        assert area.directory.satellite is None

    def test_truncated_directory(self, tmp_path):
        """Refuse a file that ends inside the directory."""
        path = tmp_path / "short.AREA_IR"
        path.write_bytes(bytes(100))
        with pytest.raises(ValueError, match="Truncated AREA directory"):
            AreaFile(str(path))
