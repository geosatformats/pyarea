"""Tests for the AREA directory (``pyarea.directory``) and the default calibrator."""
import numpy as np
import pytest

from pyarea.calibration import Calibrator, DefaultCalibrator, brit_to_temperature
from pyarea.directory import area_directory
from pyarea.utils import compose_area_name


def make_words(**words):
    """A plausible GOES-7 infrared directory as 64 words; keyword arguments set words by 0-based index name."""
    w = np.zeros(64, dtype=np.int32)
    w[1], w[2], w[3], w[4] = 4, 33, 92236, 130100
    w[8], w[9], w[10], w[13], w[18] = 10, 20, 1, 1, 1
    w[33], w[34] = 1024, 512
    for index, value in words.items():
        w[int(index.lstrip('w'))] = value
    return w


def directory(words, order='>'):
    return area_directory(words.astype(f'{order}i4').tobytes())


def test_equality_compares_all_words_whatever_the_byte_order():
    words = make_words()
    assert directory(words) == directory(words)
    assert directory(words, '>') == directory(words, '<')
    assert directory(words) != directory(make_words(w8=11))
    assert directory(words) != 'not a directory'


def test_directories_can_be_hashed_and_used_in_sets():
    words = make_words()
    assert len({directory(words), directory(words, '<'), directory(make_words(w8=11))}) == 2


@pytest.mark.parametrize(('map1', 'map2', 'count', 'expected'), [
    (1 << 7, 0, 1, [8]),
    (0, 1 << 7, 1, [40]),  # bands above 32 with a count of 1
    (0b111, 0, 1, [1]),  # more bits than bands: stop at the count
    (0b101, -(2 ** 31), 3, [1, 3, 64]),  # the sign bit of the second word
    (-(2 ** 31), 0, 1, [32]),  # the sign bit of the first word
])
def test_bands_come_from_both_map_words_up_to_the_band_count(map1, map2, count, expected):
    assert directory(make_words(w13=count, w18=map1, w19=map2)).bands == expected


def test_start_time_uses_the_actual_start_when_either_word_is_set():
    nominal = directory(make_words()).nominal_time
    assert directory(make_words()).start_time == nominal
    assert directory(make_words(w45=92236, w46=130500)).start_time.minute == 5
    assert directory(make_words(w45=92236)).start_time != nominal


def test_visr_source_type_gives_the_original_source_type():
    words = make_words()
    words[51] = int.from_bytes(b'VISR', 'big')
    assert directory(words).source_type_name == 'VISR'
    words[56] = int.from_bytes(b'AAA ', 'big')
    assert directory(words).source_type_name == 'AAA'
    words[51] = int.from_bytes(b'GVAR', 'big')
    assert directory(words).source_type_name == 'GVAR'


def test_str_lists_the_sensor_number_byte_order_and_offsets():
    text = str(directory(make_words(), '<'))
    assert 'Sensor source number: 33' in text
    assert 'little-endian' in text
    assert 'nav 512' in text
    assert 'data 1024' in text


def test_check_reports_nothing_for_a_sound_directory():
    assert directory(make_words()).check(file_size=4096) == []
    assert directory(make_words()).check() == []


def test_check_reports_what_is_wrong():
    problems = directory(make_words(w2=999, w10=3, w34=8192)).check(file_size=4096)
    assert len(problems) == 3
    assert any('999' in p and 'sensor' in p for p in problems)
    assert any('3 bytes per element' in p for p in problems)
    assert any('nav block offset 8192' in p for p in problems)
    assert any('no data block' in p for p in directory(make_words(w33=0)).check())


def test_comment_cards_that_repeat_become_a_list():
    adir = directory(make_words(w63=3))
    cards = ['Center latitude = 12.5', 'Valid calibration unit = A', 'Valid calibration unit = B']
    adir.add_comments(b''.join(c.ljust(80).encode() for c in cards))
    assert adir.center_latitude == 12.5
    assert adir.valid_calibration_unit == ['A', 'B']
    assert adir.ssp == (None, None)  # there is no longitude card


def test_a_comment_card_never_replaces_a_directory_field():
    adir = directory(make_words(w63=1))
    adir.add_comments(b'lines = 99'.ljust(80))
    assert adir.lines == 10


def test_compose_area_name_uses_the_nominal_time():
    assert compose_area_name(directory(make_words())) == 'GOES-7_B1_199208231301.pyarea'


@pytest.fixture
def default_calibrator():
    return DefaultCalibrator()


def test_default_calibrator_passes_values_through_before_a_type_is_set(default_calibrator):
    """VisAD gives 0 here; passing the pixels through means a file already in TEMP gives its own kelvin."""
    assert default_calibrator.calibrate_pixel(7.0, 1, Calibrator.CAL_TEMP) == 7.0
    np.testing.assert_array_equal(default_calibrator.calibrate_array([280, 290.5], 1, Calibrator.CAL_TEMP),
                                  np.array([280, 290.5], dtype=np.float32))


@pytest.mark.parametrize('cal_type', range(Calibrator.CAL_MIN, Calibrator.CAL_MAX + 1))
def test_default_calibrator_passes_values_through_for_every_type(default_calibrator, cal_type):
    assert default_calibrator.setCalType(cal_type) == 0
    values = np.array([[0, 5], [100, 255]])
    result = default_calibrator.calibrate_array(values, 1, Calibrator.CAL_ALB)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, values)
    assert default_calibrator.calibrate_pixel(42.0, 1, Calibrator.CAL_TEMP) == 42.0


def test_default_calibrator_ignores_the_pre_calibrated_flag(default_calibrator):
    default_calibrator.setCalType(Calibrator.CAL_TEMP)
    plain = default_calibrator.calibrate_array([180, 200], 1, Calibrator.CAL_RAW)
    default_calibrator.isPreCalibrated = True
    np.testing.assert_array_equal(default_calibrator.calibrate_array([180, 200], 1, Calibrator.CAL_RAW), plain)


def test_default_calibrator_rejects_unknown_types_and_lists_raw_and_brightness(default_calibrator):
    assert default_calibrator.setCalType(0) == -1
    assert default_calibrator.setCalType(6) == -1
    assert default_calibrator.curCalType == 0
    assert default_calibrator.calibratedList(1, True) == default_calibrator.calibratedList(1, False) == [
        Calibrator.CAL_RAW, Calibrator.CAL_BRIT]
    assert default_calibrator.calibratedUnit(Calibrator.CAL_TEMP) is None


def test_default_calibrators_do_not_share_their_calibration_type():
    first, second = DefaultCalibrator(), DefaultCalibrator()
    first.setCalType(Calibrator.CAL_RAW)
    assert second.curCalType == 0


def test_default_calibrator_does_not_change_the_input(default_calibrator):
    values = np.array([1.0, 2.0], dtype=np.float32)
    default_calibrator.calibrate_array(values, 1, Calibrator.CAL_RAW)[0] = 99
    assert values[0] == 1.0


@pytest.mark.parametrize(('brit', 'temperature'), [
    (0, 330.0), (1, 329.0), (3, 328.0), (175, 242.0), (176, 242.0), (177, 241.0), (255, 163.0), (2.9, 329.0),
])
def test_brit_to_temperature_truncates_like_visad(brit, temperature):
    assert brit_to_temperature(brit) == temperature


def test_calibration_unit_and_scale_factor_follow_visad():
    assert directory(make_words()).calibration_unit_name is None
    assert directory(make_words()).calibration_scale_factor == 1
    adir = directory(make_words(w57=int.from_bytes(b'K   ', 'big'), w58=10))
    assert adir.calibration_unit_name == 'K'
    assert adir.calibration_scale_factor == 10
    little = directory(make_words(w57=int.from_bytes(b'K   ', 'big'), w58=10), '<')
    assert little.calibration_unit_name == 'K'
    assert directory(make_words(w58=-2139062144)).calibration_scale_factor == 1


def test_read_calibrated_gives_the_stored_pixels_when_the_file_already_holds_that_type(tmp_path):
    from pyarea.file import AreaFile
    from tests.fake_area_files import GOES7_IR, IR_WINDOW, SHAPE, FakeArea, write_area_file

    values = [180, 250, 290, 310]  # kelvin needs two bytes per element
    columns = np.resize(np.asarray(values, dtype=np.uint16), SHAPE[1])
    data = np.tile(columns[None, :, None], (SHAPE[0], 1, 1))
    path = write_area_file(tmp_path / 'mcfetch_temp.area',
                           FakeArea(GOES7_IR, data, cal_type='TEMP', bytes_per_elem=2, **IR_WINDOW))
    area = AreaFile(path)
    temperature = area.read_calibrated('TEMP')
    assert temperature.dtype == np.float32
    np.testing.assert_array_equal(temperature, area.read_data().astype(np.float32))
    assert set(np.unique(temperature)) == set(values)
    with pytest.raises(ValueError, match='only GVAR files can be calibrated'):
        area.read_calibrated('RAD')


def test_read_calibrated_divides_by_the_scaling_of_an_adde_file(tmp_path):
    """A GOES-16 McFETCH file in TEMP: units word 'K', scaling 100, 19241 for 192.41 K."""
    from pyarea.file import AreaFile
    from tests.fake_area_files import GOES7_IR, IR_WINDOW, SHAPE, FakeArea, patch_dir_word, write_area_file

    columns = np.resize(np.array([19241, 25000, 31184], dtype=np.uint16), SHAPE[1])
    data = np.tile(columns[None, :, None], (SHAPE[0], 1, 1))
    path = write_area_file(tmp_path / 'abi_temp.area',
                           FakeArea(GOES7_IR, data, cal_type='TEMP', bytes_per_elem=2, **IR_WINDOW))
    patch_dir_word(path, 57, int.from_bytes(b'K   ', 'big'))
    patch_dir_word(path, 58, 100)
    area = AreaFile(path)
    assert area.directory.calibration_unit_name == 'K'
    temperature = area.read_calibrated('TEMP')
    assert temperature.dtype == np.float32
    assert set(np.unique(temperature)) == {np.float32(192.41), np.float32(250.0), np.float32(311.84)}
    np.testing.assert_array_equal(area.read_data(), data[:, :, 0])  # the stored pixels are untouched


@pytest.mark.parametrize('endian', ['>', '<'])
def test_four_byte_pixels_are_read_as_signed_integers(tmp_path, endian):
    from pyarea.file import AreaFile
    from tests.fake_area_files import GOES7_IR, IR_WINDOW, SHAPE, FakeArea, write_area_file

    values = np.array([-5, 0, 70000, 123456789], dtype=np.int64)
    columns = np.resize(values, SHAPE[1])
    data = np.tile(columns[None, :, None], (SHAPE[0], 1, 1))
    path = write_area_file(tmp_path / 'four_byte.area',
                           FakeArea(GOES7_IR, data, cal_type='RAD', bytes_per_elem=4, endian=endian, **IR_WINDOW))
    area = AreaFile(path)
    assert area.directory.bytes_per_element == 4
    pixels = area.read_data()
    assert pixels.dtype == np.int32
    np.testing.assert_array_equal(pixels, data[:, :, 0])
    assert area.npdtype == f'{endian}i4'


def test_read_calibrated_scales_four_byte_radiance(tmp_path):
    from pyarea.file import AreaFile
    from tests.fake_area_files import GOES7_IR, IR_WINDOW, SHAPE, FakeArea, patch_dir_word, write_area_file

    columns = np.resize(np.array([123456789, 2500000], dtype=np.int64), SHAPE[1])
    data = np.tile(columns[None, :, None], (SHAPE[0], 1, 1))
    path = write_area_file(tmp_path / 'abi_rad.area',
                           FakeArea(GOES7_IR, data, cal_type='RAD', bytes_per_elem=4, **IR_WINDOW))
    patch_dir_word(path, 58, 1000)
    radiance = AreaFile(path).read_calibrated('RAD')
    assert radiance.dtype == np.float32
    np.testing.assert_allclose(sorted(set(np.unique(radiance))), [2500.0, 123456.789], rtol=1e-7)
