"""Tests for the GVAR (GOES-8 onward) calibration, using a synthetic calibration block."""
import numpy as np
import pytest

from pyarea import utils
from pyarea.calibration import (
    GVAR_PLANCK_COEFFS,
    Calibrator,
    G8GVARCalibrator,
    GVARCalibrator,
    brit_to_temperature,
    cal_type_from_name,
    calibrated_unit,
    temperature_to_brit,
)


def to_gould(value):
    """Encode a float as a 32-bit Gould word (the inverse of ``utils.gould2native``)."""
    if value == 0:
        return 0
    mag = abs(value)
    exp = 70
    while mag >= 2 ** 24 / 16 ** (70 - exp) * 1.0 and exp < 127 or mag * 16 ** (70 - exp) >= 2 ** 24:
        exp += 1
    while mag * 16 ** (70 - exp) < 2 ** 20 and exp > 0:
        exp -= 1
    mant = int(round(mag * 16 ** (70 - exp)))
    word = (exp << 24) | mant
    if value < 0:
        word = (-word) & 0xFFFFFFFF
    return word


@pytest.mark.parametrize("value", [1.0, -2.3561944, 0.0523, 1234.5, -0.5])
def test_gould_round_trip(value):
    assert utils.gould2native(to_gould(value)) == pytest.approx(value, rel=1e-5)


VIS_BIAS, VIS_G1, VIS_G2, RAD_TO_ALB = -0.5, 0.01, 1.5, 0.002
IR_GAIN, IR_BIAS = 0.2, -1.0


def imager_block(sid=70, with_ir=(IR_GAIN, IR_BIAS)):
    """Imager block: 8 detectors of visible coefficients, then IR bands stored rotated for sid <= 77."""
    vis = [VIS_BIAS] * 8 + [VIS_G1] * 8 + [VIS_G2] * 8 + [RAD_TO_ALB]
    gain, bias = with_ir
    # logical band order: IR bands 2..5 each with gain/bias, detector 0 and 1 identical
    gains = [gain * (i + 1) for i in range(4)]
    biases = [bias * (i + 1) for i in range(4)]
    shift = 0 if sid > 77 else 2
    stored_b = [biases[(i - shift) % 4] for i in range(4)]
    stored_g = [gains[(i - shift) % 4] for i in range(4)]
    words = vis + stored_b * 2 + stored_g * 2
    return [to_gould(v) for v in words], gains, biases


def test_visible_albedo_and_brit():
    cal = GVARCalibrator(70, imager_block()[0])
    counts = np.array([320.0, 640.0, 0.0])
    f = counts / 32
    expected = np.maximum((f ** VIS_G2 + f * VIS_G1 + VIS_BIAS) * RAD_TO_ALB, 0.0)
    alb = cal.calibrate_array(counts, 1, Calibrator.CAL_ALB)
    np.testing.assert_allclose(alb, expected * 100, rtol=1e-4, atol=1e-6)
    brit = cal.calibrate_array(counts, 1, Calibrator.CAL_BRIT)
    np.testing.assert_array_equal(brit, np.minimum(np.floor(np.sqrt(100 * expected) * 25.5 + 0.5), 255))
    huge = cal.calibrate_array(np.array([60000.0]), 1, Calibrator.CAL_BRIT)
    assert huge[0] == 255
    with pytest.raises(ValueError):
        cal.calibrate_array(counts, 1, Calibrator.CAL_TEMP)


@pytest.mark.parametrize("band", [2, 3, 4, 5])
def test_ir_radiance_and_temperature(band):
    block, gains, biases = imager_block()
    cal = GVARCalibrator(70, block)
    counts = np.array([3200.0, 4800.0])
    rad = cal.calibrate_array(counts, band, Calibrator.CAL_RAD)
    np.testing.assert_allclose(rad, (counts / 32 - biases[band - 2]) / gains[band - 2], rtol=1e-4)
    consts = {k: v[band - 1] for k, v in GVAR_PLANCK_COEFFS[8]["imager"].items()}
    expected = (consts["FK2"] / np.log(consts["FK1"] / rad.astype(float) + 1) - consts["TC1"]) / consts["TC2"]
    temp = cal.calibrate_array(counts, band, Calibrator.CAL_TEMP)
    np.testing.assert_allclose(temp, expected, rtol=1e-4)
    assert temp.dtype == np.float32


def test_ir_offset_depends_on_sensor():
    """Before sensor 78 the IR bands are stored rotated by two; from 78 on they are not."""
    for sid in (70, 78):
        block, gains, biases = imager_block(sid)
        cal = GVARCalibrator(sid, block)
        np.testing.assert_allclose(cal.irGainCoef[0], gains)
        np.testing.assert_allclose(cal.irBiasCoef[0], biases)


def test_band_6_uses_band_5_coefficients():
    cal = GVARCalibrator(70, imager_block()[0])
    c = np.array([4000.0])
    np.testing.assert_array_equal(cal.calibrate_array(c, 6, Calibrator.CAL_RAD),
                                  cal.calibrate_array(c, 5, Calibrator.CAL_RAD))


def test_raw_and_precalibrated_pass_through():
    c = np.array([1.0, 2.0])
    np.testing.assert_array_equal(GVARCalibrator(70, imager_block()[0]).calibrate_array(c, 2, Calibrator.CAL_RAW), c)
    np.testing.assert_array_equal(GVARCalibrator(70, None).calibrate_array(c, 2, Calibrator.CAL_TEMP), c)


def test_unknown_satellite_and_short_block():
    with pytest.raises(NotImplementedError):
        GVARCalibrator(78, imager_block(78)[0]).calibrate_array(np.array([100.0]), 2, Calibrator.CAL_TEMP)
    with pytest.raises(ValueError, match="needed"):
        GVARCalibrator(70, [0] * 10)


def test_sounder_uses_scale_2():
    n = 4
    words = [to_gould(v) for v in [VIS_BIAS] * n + [VIS_G1] * n + [VIS_G2] * n + [RAD_TO_ALB]
             + [-1.0] * 18 + [0.5] * 18]
    cal = GVARCalibrator(71, words)
    rad = cal.calibrate_array(np.array([100.0]), 1, Calibrator.CAL_RAD)
    assert rad[0] == pytest.approx((100.0 - -1.0) / 0.5, rel=1e-5)


def test_lists_units_and_brit_conversion():
    assert cal_type_from_name(" temp ") == Calibrator.CAL_TEMP
    with pytest.raises(ValueError):
        cal_type_from_name("xyz")
    assert calibrated_unit(Calibrator.CAL_TEMP) == "K"
    assert calibrated_unit(Calibrator.CAL_BRIT) is None
    cal = G8GVARCalibrator(70, imager_block()[0])
    assert Calibrator.CAL_ALB in cal.calibratedList(1)
    assert Calibrator.CAL_TEMP in cal.calibratedList(4)
    t = np.array([300.0, 200.0])
    np.testing.assert_allclose(brit_to_temperature(temperature_to_brit(t)), t, atol=1.0)


def test_read_calibrated_from_file(tmp_path):
    from pyarea.file import AreaFile
    from tests.fake_area_files import IR_WINDOW, SHAPE, FakeArea, patch_dir_word, write_area_file

    block, gains, biases = imager_block()
    words = np.zeros(128, dtype=np.int64)
    words[:len(block)] = block
    data = np.full(SHAPE + (1,), 4000)
    spec = FakeArea(70, data, source_type="GVAR", cal_type="RAW", bytes_per_elem=2, cal_words=words, **IR_WINDOW)
    path = write_area_file(tmp_path / "gvar.area", spec)
    patch_dir_word(path, 18, 1 << 3)  # spectral band map: band 4
    area = AreaFile(path)
    assert area.directory.bands == [4]
    rad = area.read_calibrated("RAD")
    assert rad.shape == SHAPE
    np.testing.assert_allclose(rad, (4000 / 32 - biases[2]) / gains[2], rtol=1e-4)
    assert np.isfinite(area.read_calibrated("TEMP")).all()
