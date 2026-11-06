import ctypes as C
import logging
import os
import typing

import numpy as np
import sys
from pyarea.utils import yyyddd_hhmmss2datetime

_, n = os.path.split(os.path.abspath(__file__))
logger = logging.getLogger(n)



import struct


# sensor numbers 16-33 are SMS-1, SMS-2 and GOES-1..7, each as a visible number followed by an infrared one
_PREGVAR_FIRST_SENSOR = 16
_PREGVAR_PLATFORMS = ("SMS-1", "SMS-2") + tuple(f"GOES-{n}" for n in range(1, 8))

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

    sensors = (
    "Derived Data",
    "Test Patterns",
    "Graphics",
    "Miscellaneous",
    "PDUS Meteosat visible",
    "PDUS Meteosat infrared",
    "PDUS Meteosat water vapor",
    "Radar",
    "Miscellaneous Aircraft Data",
    "Raw Meteosat",
    "Composite image",
    "Topography image",
    "GMS visible",
    "GMS infrared",
    "ATS 6 visible",
    "ATS 6 infrared",
    "SMS-1 visible",
    "SMS-1 infrared",
    "SMS-2 visible",
    "SMS-2 infrared",
    "GOES-1 visible",
    "GOES-1 infrared",
    "GOES-2 visible",
    "GOES-2 infrared",
    "GOES-3 visible",
    "GOES-3 infrared",
    "GOES-4 visible (VAS)",
    "GOES-4 infrared and water vapor (VAS)",
    "GOES-5 visible (VAS)",
    "GOES-5 infrared and water vapor (VAS)",
    "GOES-6 visible",
    "GOES-6 infrared",
    "GOES-7 visible, Block 1 supplemental data",
    "GOES-7 infrared",
    "FY-2B",
    "FY-2C",
    "FY-2D",
    "FY-2E",
    "FY-2F",
    "FY-2G",
    "FY-2H",
    "TIROS-N (POES)",
    "NOAA-6", "NOAA-7",
    "NOAA-8", "NOAA-9",
    "Venus",
    "Voyager 1",
    "Voyager 2",
    "Galileo",
    "Hubble space telescope",
    "MSG-1",
    "MSG-2",
    "MSG-3",
    "Meteosat-3",
    "Meteosat-4",
    "Meteosat-5",
    "Meteosat-6",
    "Meteosat-7",
    "",
    "NOAA-10",
    "NOAA-11",
    "NOAA-12",
    "NOAA-13",
    "NOAA-14",
    "NOAA-15",
    "NOAA-16",
    "NOAA-17",
    "NOAA-18",
    "NOAA-19",
    "GOES 8 imager",  # 70
    "GOES 8 sounder",
    "GOES 9 imager",
    "GOES 9 sounder",
    "GOES 10 imager",
    "GOES 10 sounder",
    "GOES 11 imager",
    "GOES 11 sounder",
    "GOES 12 imager",
    "GOES 12 sounder",
    "ERBE",
    "",
    "GMS-4",
    "GMS-5",
    "GMS-6",
    "GMS-7",
    "GMS-8",
    "DMSP F-8",
    "DMSP F-9",
    "DMSP F-10",
    "DMSP F-11",
    "DMSP F-12",
    "DMSP F-13",
    "DMSP F-14",
    "DMSP F-15",  # 94
    "FY-1B",
    "FY-1C",
    "FY-1D",
    "",
    "",
    "",
    "TERRA L1B",  # 101
    "TERRA CLD",
    "TERRA GEO",
    "TERRA-AER",
    "",
    "TERRA TOP",
    "TERRA ATM",
    "TERRA GUE",
    "TERRA RET",
    "",
    "AQUA L1B",# 111
    "AQUA CLD",
    "AQUA GEO",
    "AQUA AER",
    "",
    "AQUA TOP",
    "AQUA ATM",
    "AQUA GUE",
    "AQUA RET",
    "",  # 120
    "", "", "", "", "", "", "", "", "", "",  # 130
    "", "", "", "", "", "", "", "", "", "",  # 140
    "", "", "", "", "", "", "", "", "", "",  # 150
    "", "", "", "", "", "", "", "", "",
    "TERRA NDVI",  # 160
    "TERRA CREF",
    "", "", "", "", "", "", "", "",
    "AQUA NDVI",  # 170
    "AQUA CREF",
    "", "", "", "", "", "", "", "",
    "GOES 13 imager",  # 180
    "GOES 13 sounder",
    "GOES 14 imager",
    "GOES 14 sounder",
    "GOES 15 imager",
    "GOES 15 sounder",
    "GOES 16 imager",
    "GOES 16 sounder",
    "",  # 188
    "", "",  # 190
    "", "", "", "", "DMSP F-16",  # 195
    "DMSP F-17",
    "", "", "",  # 199
    "AIRS L1B",  # 200
    "", "", "", "", "", "", "", "", "",
    "AMSR-E L1B",  # 210
    "AMSR-E RAIN",
    "", "", "", "", "", "", "", "", "",  # 220
    "", "", "", "", "", "", "", "", "",
    "Kalpana-1",  # 230
    "", "", "", "", "", "", "", "", "",
    "MetOp-A",  # 240
    "MetOp-B",
    "MetOp-C",
    "")

    def _setup(self, dir_bytes):
        self.dir_bytes = dir_bytes
        self.nominal_time = yyyddd_hhmmss2datetime(self.yyyddd, self.hhmmss)
        self.creation_time = yyyddd_hhmmss2datetime(self.file_yyyddd, self.file_hhmmss)
        if self.image_yyyddd != 0 and self.image_hhmmss_or_ms !=0:
            self.start_time = yyyddd_hhmmss2datetime(self.image_yyyddd, self.image_hhmmss_or_ms)
        else:
            self.start_time = self.nominal_time



        # print('{0:032b}'.format(self.spectral_band_map_1_32))
        # for i in range(32):
        #     print(i, self.spectral_band_map_1_32 & (1 << i) == self.spectral_band_map_1_32)


        #next code actually does check if n'th bit is set where n is in range (0, 32)
        #for the second word it adds  32 offset to reach bands 33-64
        self.bands_1_32 = [i+1  for i in range(32) if self.spectral_band_map_1_32&(1<<i)>0]
        if self.spectral_band_count >32:
            self.bands_33_64 = [32+i+1  for i in range(32) if self.spectral_band_map_33_64&(1<<i)>0]
        else:
            self.bands_33_64 = []
        self.bands = self.bands_1_32 + self.bands_33_64

        self.comment_cards = list() if self.comment_count > 0 else None


    def __eq__(self, other):
        return self.nominalTime == other.nominalTime and self.imgbox == other.imgbox and self.byte_depth == other.byte_depth


    @property
    def buffer(self):
        return C.string_at(C.byref(self), C.sizeof(self))

    def __repr__(self):
        return self.__str__()

    def __str__(self):

        r = f'\n{"-"*35} \n' \
            f'{self.sensor} {self.cal_type.decode().strip()} {self.source_type.decode().strip()} image scanned at {self.start_time}\n' \
            f'{"-"*35}\n' \
            f'Nominal time: {self.nominal_time}\n'\
            f'Bands: {self.bands}\n'\
            f'Y resolution: {self.line_res}\n'\
            f'X resolution: {self.element_res}\n'\
            f'Size: {str(self.size)}\n' \
            f'Data width: {self.bytes_per_element}\n'\
            f'Bounding box: {self.imgbox}\n'

        if self.comment_cards:
            r+= f'Comment cards:\n'
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
                except Exception as e:
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
                if not hasattr(self, card_name):
                    setattr(self, card_name, val)

                else:
                    old_val = getattr(self, card_name, 'None')
                    if isinstance(old_val, typing.List):
                        old_val.append(val)

                    else:
                        old_val = list(old_val)
                    setattr(self, card_name, old_val)
            except Exception as e:
                logger.error(f'Failed to parse card number {ci} on {self.id_text}. Error is {str(e)}')

            card_start = card_end

    @property
    def id_text(self):
        return f'{self.sensor} {self.cal_type.decode()} {self.source_type.decode()} image scanned at {self.start_time}\n'

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
    def sensor(self):
        """Short platform name, e.g. ``GOES-3`` or ``GOES8`` (None if there isn't one)."""
        if self.satellite is not None:
            return self.satellite
        text = self.sensor_name
        if ' ' in text:
            stext = text.split(' ')
            if 'GOES' in stext:
                return ''.join(stext[:2])
        else:
            return text

    @property
    def _pregvar_index(self):
        """0-17 for the SMS-1..GOES-7 sensor numbers (visible, infrared, ...), otherwise None."""
        index = self.sensor_source_number - _PREGVAR_FIRST_SENSOR
        return index if 0 <= index < 2 * len(_PREGVAR_PLATFORMS) else None

    @property
    def satellite(self):
        """Pre-GVAR platform name (``SMS-1``, ``GOES-3``...) or None for any other source."""
        index = self._pregvar_index
        return None if index is None else _PREGVAR_PLATFORMS[index // 2]

    @property
    def band(self):
        """``VIS`` or ``IR`` for pre-GVAR SMS/GOES sensor numbers, otherwise None."""
        index = self._pregvar_index
        return None if index is None else ("VIS", "IR")[index % 2]

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
