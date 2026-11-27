import ctypes as C
import logging
import os
import struct

import numpy as np

from pyarea.sensors import SENSORS, get_sensor
from pyarea.utils import yyyddd_hhmmss2datetime

_, n = os.path.split(os.path.abspath(__file__))
logger = logging.getLogger(n)

# the sensor names indexed by sensor number ("" where a number is not used)
_SENSOR_NAMES = tuple(SENSORS[n].name if n in SENSORS else "" for n in range(max(SENSORS) + 1))

# http://www.ssec.wisc.edu/mcidas/doc/prog_man/current/formats-1.html#13797
HEADER_FIELDS = (
    # words 1-8
    ('relative_position_within_dataset', C.c_int32),
    ('image_type', C.c_int32),
    ('sensor_source_number', C.c_int32),
    ('yyyddd', C.c_int32),
    ('hhmmss', C.c_int32),
    ('line_ul', C.c_int32),
    ('element_ul', C.c_int32),
    ('_reserved1', C.c_int32),

    # 9-16
    ('lines', C.c_int32),
    ('elements', C.c_int32),
    ('bytes_per_element', C.c_int32),
    ('line_res', C.c_int32),
    ('element_res', C.c_int32),
    ('spectral_band_count', C.c_int32),
    ('line_prefix_length', C.c_int32),
    ('project', C.c_int32),

    # 17-24
    ('file_yyyddd', C.c_int32),
    ('file_hhmmss', C.c_int32),
    ('spectral_band_map_1_32', C.c_int32),
    ('spectral_band_map_33_64', C.c_int32),
    ('sensor_specific', C.c_int32 * 4),

    # 25-32
    ('memo', C.c_char * 32),

    # 33-36
    ('_reserved2', C.c_int32),
    ('data_block_offset', C.c_int32),
    ('nav_block_offset', C.c_int32),
    ('validity_code', C.c_int32),

    # 37-44
    ('program_data_load', C.c_int32 * 8),

    # 45-48
    ('band8_source_goesaa', C.c_int32),
    ('image_yyyddd', C.c_int32),
    ('image_hhmmss_or_ms', C.c_int32),
    ('start_scan', C.c_int32),

    # 49-56
    ('prefix_doc_length', C.c_int32),
    ('prefix_cal_length', C.c_int32),
    ('prefix_band_length', C.c_int32),
    ('source_type', C.c_char * 4),
    ('cal_type', C.c_char * 4),
    ('_reserved3', C.c_int32 * 3),

    # 57-64
    ('original_source_type', C.c_char*4),
    ('units', C.c_int32),
    ('scaling', C.c_int32),
    ('aux_block_offset', C.c_int32),
    ('aux_block_length', C.c_int32),
    ('_reserved4', C.c_int32),
    ('cal_block_offset', C.c_int32),
    ('comment_count', C.c_int32)
)


class _DirectoryMixin:
    """Behaviour shared by the big- and little-endian directory structures."""

    sensors = _SENSOR_NAMES

    def _setup(self, dir_bytes):
        self.dir_bytes = dir_bytes
        self.nominal_time = yyyddd_hhmmss2datetime(self.yyyddd, self.hhmmss)
        self.creation_time = yyyddd_hhmmss2datetime(self.file_yyyddd, self.file_hhmmss)
        # as in VisAD's AreaDirectory the actual start is used unless both of its words are zero
        if self.image_yyyddd != 0 or self.image_hhmmss_or_ms != 0:
            self.start_time = yyyddd_hhmmss2datetime(self.image_yyyddd, self.image_hhmmss_or_ms)
        else:
            self.start_time = self.nominal_time

        # Word 19 maps bands 1-32 and word 20 bands 33-64 (bit 0 of word 19 is band 1). Word 20 is only
        # documented as satellite specific for older sensors, so, like VisAD, stop at the declared band count:
        # a map can have more bits set than bands (the AVHRR 3a/3b case).
        self.bands_1_32 = [i + 1 for i in range(32) if (self.spectral_band_map_1_32 >> i) & 1]
        self.bands_33_64 = [i + 33 for i in range(32) if (self.spectral_band_map_33_64 >> i) & 1]
        self.bands = (self.bands_1_32 + self.bands_33_64)[:max(self.spectral_band_count, 0)]

        self.comment_cards = list() if self.comment_count > 0 else None
        self._card_names = set()


    @property
    def words(self):
        """The 64 directory words as native int32 values, whatever byte order the file was written in."""
        return np.frombuffer(self.dir_bytes, dtype=f'{self.byte_order}i4').astype(np.int32)

    def __eq__(self, other):
        """Equal when all 64 words are, like ``AreaDirectory.equals`` in VisAD (the byte order does not matter)."""
        if not isinstance(other, _DirectoryMixin):
            return NotImplemented
        return bool(np.array_equal(self.words, other.words))

    def __hash__(self):
        return hash(self.words.tobytes())

    @property
    def buffer(self):
        return C.string_at(C.byref(self), C.sizeof(self))

    def __repr__(self):
        return self.__str__()

    @property
    def source_type_name(self):
        """The image source type (``VISR``, ``GVAR``...); for ``VISR`` the original source type when there is one."""
        source = self.source_type.decode(errors='replace').strip('\x00 ')
        original = self.original_source_type.decode(errors='replace').strip('\x00 ')
        return original if source.upper() == 'VISR' and original else source

    @property
    def calibration_unit_name(self):
        """The unit of the stored values (directory word 58, ASCII), or None where the file gives none."""
        raw = self.dir_bytes[57 * 4:58 * 4]
        if self.units in (0, _MCMISSING):
            return None
        return raw.decode('ascii', errors='replace').strip('\x00 ') or None

    @property
    def calibration_scale_factor(self):
        """The scaling factor of the stored values (word 59); 1 where it is 0 or missing, as in VisAD."""
        return 1 if self.scaling in (0, _MCMISSING) else self.scaling

    def check(self, file_size=None):
        """Sanity checks on the directory; returns a list of messages, empty when nothing looks wrong.

        Args:
            file_size: The size of the file in bytes, to check the block offsets against; skipped when omitted.
                Truncated data (fewer lines than the directory claims) is reported by :class:`pyarea.file.AreaFile`,
                which knows the line length.
        """
        problems = []
        if self.sensor_info is None:
            problems.append(f'Sensor source number {self.sensor_source_number} is not in the sensor table.')
        if self.bytes_per_element not in (1, 2, 4):
            problems.append(f'{self.bytes_per_element} bytes per element; only 1, 2 or 4 are defined.')
        if self.lines <= 0 or self.elements <= 0:
            problems.append(f'The image size is {self.lines} lines x {self.elements} elements.')
        if self.data_block_offset <= 0:
            problems.append('There is no data block offset.')
        if file_size is not None:
            for name in ('nav', 'cal', 'aux', 'data'):
                offset = getattr(self, f'{name}_block_offset')
                if offset >= file_size or offset < 0:
                    problems.append(f'The {name} block offset {offset} is outside the file ({file_size} bytes).')
        return problems

    def __str__(self):

        r = f'\n{"-"*35} \n' \
            f'{self.sensor} {self.cal_type.decode().strip()} {self.source_type_name} '\
            f'image scanned at {self.start_time}\n' \
            f'{"-"*35}\n' \
            f'Nominal time: {self.nominal_time}\n'\
            f'Bands: {self.bands}\n'\
            f'Y resolution: {self.line_res}\n'\
            f'X resolution: {self.element_res}\n'\
            f'Size: {str(self.size)}\n' \
            f'Data width: {self.bytes_per_element}\n'\
            f'Bounding box: {self.imgbox}\n'\
            f'Sensor source number: {self.sensor_source_number}\n'\
            f'Byte order: {"big" if self.byte_order == ">" else "little"}-endian\n'\
            f'Line prefix length: {self.line_prefix_length}\n'\
            f'Block offsets: nav {self.nav_block_offset}, cal {self.cal_block_offset}, '\
            f'aux {self.aux_block_offset}, data {self.data_block_offset}\n'

        if self.comment_cards:
            r+= 'Comment cards:\n'
            for cc in self.comment_cards:
                r+= f'\t{cc}\n'
        return r

    def add_comments(self, comment_bytes=None):
        card_start = 0
        for ci in range(self.comment_count):
            card_end = card_start+80
            card_bytes = comment_bytes[card_start:card_end]
            card_txt = struct.unpack(f'{self.byte_order}80s', card_bytes)[0].decode('utf-8')
            #card_txt = card_bytes.decode('utf-8', 'ignore')

            try:
                try:
                    card_name, sval = card_txt.split('=')
                except Exception:
                    self.comment_cards.append(str(card_txt))
                    continue

                card_name = card_name.strip()
                try:  # float
                    val = float(sval)
                except ValueError:
                    try:  # int
                        val = int(sval)
                    except ValueError:
                        val = sval.strip()

                # clean special chars except ""
                card_name = card_name.replace(' ', '_')
                card_name = card_name.replace('(', '')
                card_name = card_name.replace(')', '')
                card_name = card_name.replace('[', '')
                card_name = card_name.replace(']', '')
                card_name = card_name.replace('{', '')
                card_name = card_name.replace('}', '')
                card_name = card_name.lower()
                # print(f'{card_name} = {val}')
                self.comment_cards.append(f'{card_name} = {val}')
                if card_name not in self._card_names:
                    if not hasattr(self, card_name):  # never replace a directory field or method
                        setattr(self, card_name, val)
                        self._card_names.add(card_name)
                else:  # a card that comes more than once becomes a list of its values
                    old_val = getattr(self, card_name)
                    if isinstance(old_val, list):
                        old_val.append(val)
                    else:
                        setattr(self, card_name, [old_val, val])
            except Exception as e:
                logger.error(f'Failed to parse card number {ci} on {self.id_text}. Error is {str(e)}')

            card_start = card_end

    @property
    def id_text(self):
        return (f'{self.sensor} {self.cal_type.decode()} {self.source_type_name} '
                f'image scanned at {self.start_time}\n')

    @property
    def size(self):
        return self.lines, self.elements

    @property
    def imgbox(self):
        """
            Property
            Computes are returns the bounding box as a tuple (stline, endline, stelem, endelem)
        """
        start_line = self.line_ul
        end_line = self.line_ul + self.lines
        start_elem = self.element_ul
        end_elem = self.element_ul + self.elements
        return start_line, end_line, start_elem, end_elem


    @property
    def vis_imgbox(self):
        """
            Property
            Computes are returns the bounding box as a tuple (stline, endline, stelem, endelem)
        """
        start_line = self.line_ul
        end_line = self.line_ul + self.lines*self.line_res
        start_elem = self.element_ul
        end_elem = self.element_ul + self.elements*self.element_res
        return start_line, end_line, start_elem, end_elem

    @property
    def sensor_name(self):
        """The full name for the sensor/source number, e.g. ``GOES-3 infrared`` (empty if unknown)."""
        number = self.sensor_source_number
        return self.sensors[number] if 0 <= number < len(self.sensors) else ""

    @property
    def sensor_info(self):
        """The :class:`pyarea.sensors.Sensor` for this file's sensor number, or None if it is not in the table."""
        return get_sensor(self.sensor_source_number)

    @property
    def sensor(self):
        """Short platform name, e.g. ``GOES-3`` or ``GOES8`` (None if there isn't one)."""
        if 16 <= self.sensor_source_number <= 33:  # SMS-1 to GOES-7
            return self.satellite
        text = self.sensor_name
        if ' ' in text:
            stext = text.split(' ')
            if 'GOES' in stext:
                return ''.join(stext[:2])
            return self.satellite or text  # e.g. "GMS infrared" -> GMS
        else:
            return text

    @property
    def satellite(self):
        """The platform name from the sensor table (``SMS-1``, ``GOES-3``, ``NOAA-15``...), or None."""
        info = self.sensor_info
        return None if info is None else info.platform

    @property
    def band(self):
        """``VIS`` or ``IR`` for visible and infrared sensors in the sensor table, otherwise None."""
        info = self.sensor_info
        return None if info is None else info.band

    @property
    def ssp(self):
        try:
            return self.center_latitude, self.center_longitude
        except Exception:
            return None, None


    def get_units(self):
        unit_options = list()
        if self.comment_count:
            for card in self.comment_cards:
                if 'unit' in card:
                    n, v = card.split('=')
                    uname, ucomment = v.strip().split(' "')
                    unit_options.append(uname)

        if unit_options:
            return unit_options[self.units]


class _AreaDirectoryBE(_DirectoryMixin, C.BigEndianStructure):
    _pack_ = 1
    _layout_ = "ms"  # what Python used implicitly; spelled out since 3.14 deprecates the default
    _fields_ = HEADER_FIELDS


class _AreaDirectoryLE(_DirectoryMixin, C.LittleEndianStructure):
    _pack_ = 1
    _layout_ = "ms"  # what Python used implicitly; spelled out since 3.14 deprecates the default
    _fields_ = HEADER_FIELDS


_MCMISSING = -2139062144  # 0x80808080, the McIDAS "missing" value
_DIRECTORY_CLASSES = {'>': _AreaDirectoryBE, '<': _AreaDirectoryLE}
DIRECTORY_BYTES = 256


def _directory_plausibility(words):
    """Score how plausible a directory is when its 64 words are read in one byte order."""
    names = _DirectoryMixin.sensors
    sensor, lines, elements = int(words[2]), int(words[8]), int(words[9])
    score = 10 if 0 <= sensor < len(names) and names[sensor] else 0
    score += 5 * (int(words[1]) == 4)  # the AREA version number
    return score + int(0 < lines < 200000) + int(0 < elements < 200000)


def detect_byte_order(dir_bytes):
    """Return ``'>'`` or ``'<'``, whichever reads the directory as the more plausible one.

    The AREA spec says files are big-endian, but plenty of real ones were
    written little-endian, so both are tried and scored on the sensor number,
    the version word and the image size. Ties go to big-endian.
    """
    scores = {order: _directory_plausibility(np.frombuffer(dir_bytes[:DIRECTORY_BYTES], dtype=f'{order}i4'))
              for order in ('>', '<')}
    best = max(scores, key=lambda order: (scores[order], order == '>'))
    if scores[best] < 10:
        logger.warning('Could not confidently detect the AREA byte order (scores %s); assuming big-endian.', scores)
        return '>'
    return best


def area_directory(dir_bytes, byte_order=None):
    """Parse the 256-byte AREA directory.

    Args:
        dir_bytes: The first 256 bytes of the file.
        byte_order: ``'>'`` or ``'<'``; detected from the content when omitted.

    Returns:
        A ctypes structure with the directory words as attributes and the
        byte order it was read with in ``.byte_order``.
    """
    if len(dir_bytes) < DIRECTORY_BYTES:
        raise ValueError(f'Truncated AREA directory: {len(dir_bytes)} bytes, expected {DIRECTORY_BYTES}')
    dir_bytes = bytes(dir_bytes[:DIRECTORY_BYTES])
    if byte_order is None:
        byte_order = detect_byte_order(dir_bytes)
    directory = _DIRECTORY_CLASSES[byte_order].from_buffer_copy(dir_bytes)
    directory.byte_order = byte_order
    directory._setup(dir_bytes)
    return directory
