"""Tests for the sensor source table and what the directory makes of it."""

import pytest

from pyarea.directory import _DirectoryMixin
from pyarea.file import AreaFile
from pyarea.sensors import SENSOR_NAMES, SENSORS, get_sensor
from tests.fake_area_files import IR_WINDOW, FakeArea, mode_a_data, write_area_file


@pytest.mark.parametrize(
    ("number", "platform", "instrument", "band"),
    [
        (16, "SMS-1", "visible", "VIS"),
        (17, "SMS-1", "infrared", "IR"),
        (27, "GOES-4", "infrared and water vapor (VAS)", "IR"),
        (32, "GOES-7", "visible, Block 1 Auxiliary Data", "VIS"),
        (33, "GOES-7", "infrared", "IR"),
        (6, "PDUS Meteosat", "water vapor", "IR"),
        (12, "GMS", "visible", "VIS"),
        (70, "GOES-8", "imager", None),
        (186, "GOES-16", "imager", None),
        (187, "GOES-16", "Level 2 Products", None),
        (190, "GOES-18", "imager", None),
        (192, "GOES-19", "imager", None),
        (174, "EWS-G1", "imager", None),
        (179, "EWS-G3", "sounder", None),
        (221, "GMS-1", None, None),
        (287, "Himawari-9", None, None),
        (365, "MTG-S1", "sounder", None),
        (65, "NOAA-15", None, None),
        (86, "Himawari-8", None, None),
        (104, "TERRA-AER", None, None),
    ])
def test_sensor_entries(number, platform, instrument, band):
    """The names are split into platform, instrument and band."""
    sensor = get_sensor(number)
    assert (sensor.platform, sensor.instrument, sensor.band) == (platform, instrument, band)
    assert sensor.name == SENSOR_NAMES[number]


@pytest.mark.parametrize("number", [-1, 35 + 100, 99, 5000])
def test_unknown_numbers(number):
    """Numbers that are not in the table have no sensor."""
    assert get_sensor(number) is None


def test_every_entry_has_a_platform():
    """Every name in the table parses to something."""
    assert all(sensor.platform for sensor in SENSORS.values())
    assert len(SENSORS) == len(SENSOR_NAMES)


def test_pre_gvar_numbers_are_sms_and_goes_1_to_7():
    """Numbers 16-33 are SMS-1, SMS-2 and GOES-1..7, each visible then infrared."""
    platforms = ["SMS-1", "SMS-2"] + [f"GOES-{n}" for n in range(1, 8)]
    for index, number in enumerate(range(16, 34)):
        sensor = SENSORS[number]
        assert sensor.platform == platforms[index // 2]
        assert sensor.band == ("VIS", "IR")[index % 2]


@pytest.mark.parametrize(
    ("number", "satellite", "band", "short_name"),
    [(17, "SMS-1", "IR", "SMS-1"), (32, "GOES-7", "VIS", "GOES-7"), (70, "GOES-8", None, "GOES8"),
     (15, "ATS 6", "IR", None), (65, "NOAA-15", None, "NOAA-15"), (99, None, None, "")])
def test_directory_properties(tmp_path, number, satellite, band, short_name):
    """The directory reports platform and band from its sensor number."""
    area = AreaFile(write_area_file(tmp_path / "x.AREA", FakeArea(number, mode_a_data([100]), **IR_WINDOW)))
    directory = area.directory
    assert (directory.satellite, directory.band) == (satellite, band)
    assert directory.sensor == short_name


def test_sensors_tuple_is_indexed_by_number():
    """``area_directory.sensors`` keeps working for code that indexes it by sensor number."""
    names = _DirectoryMixin.sensors
    assert names[24] == "GOES-3 visible"
    assert names[1] == "Test Patterns"
    assert names[59] == ""


def test_table_matches_the_mcidas_users_guide():
    """Spot checks against Appendix C ("Sensor Source Numbers") of the McIDAS Users Guide."""
    expected = {51: "Meteosat-8 (MSG-1)", 80: "ERBE", 108: "TERRA GUS", 188: "GOES 17 imager", 222: "GMS-2",
                223: "GMS-3", 304: "NOAA-23 (JPSS-4)", 357: "Meteosat-14 (MTG-I3)"}
    for number, name in expected.items():
        assert SENSOR_NAMES[number] == name
    assert len(SENSOR_NAMES) >= 196
