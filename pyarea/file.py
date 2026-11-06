from functools import cached_property
from io import BytesIO
from pyarea.directory import DIRECTORY_BYTES, area_directory
from pyarea.calibration import (
    MODE_AA_DELTA_F,
    aaa_radiance,
    aaa_temperature,
    mode_a_ir_temperature,
    mode_aa_code_to_band,
    mode_aa_temperature,
    vas_planck_coeffs,
    vis_unnormalized_reflectance,
)
from pyarea.nav import AreaNav
from pyarea.utils import INT32SIZE
import struct
import numpy as np
import os
import logging
_, name = os.path.split(__file__)
logger = logging.getLogger(name)

_AAA_PRE_BANDLIST_LEN = 4 + 512 + 116  # validity code, common documentation, VAS calibration info
_AAA_CAL_MIN_WORDS = 117

class AreaFile(BytesIO):
    """A McIDAS AREA file: directory, navigation, calibration and image data.

    Besides the generic AREA parts this reads the pre-GVAR SMS/GOES-1..7
    VISSR/VAS products (Mode A, AA and AAA), including their per-line band
    lists and calibration; see the ``read_*`` methods.
    """

    INTEGER_FORMAT = 'i'

    def __init__(self, source=None, lazy=True):

        if not isinstance(source, bytes):
            self.is_remote = False
            self.source = source
            with open(source, 'rb') as src:
                d = src.read()
                super().__init__(d)

        else:
            self.is_remote = True
            self.source = None
            super().__init__(source)

        self.sz = len(self.getbuffer())

        self.directory = area_directory(self.getbuffer()[:DIRECTORY_BYTES])
        self.adir_bytes = self.directory.dir_bytes

        # as per http://www.ssec.wisc.edu/mcidas/doc/prog_man/current/formats-1.html#34971
        # after the directory bytes the blocks are, in order: nav, cal, aux, data, comments
        self.navigation_offset = self.directory.nav_block_offset
        self.calibration_offset = self.directory.cal_block_offset
        self.aux_offset = self.directory.aux_block_offset
        self.data_offset = self.directory.data_block_offset

        # the directory's own prefix length is what the data is laid out with; the documented parts
        # (validity code, documentation, calibration, band list) do not always add up to it in real files
        self.line_prefix_len = self.directory.line_prefix_length
        parts = (self.directory.prefix_doc_length + self.directory.prefix_cal_length
                 + self.directory.prefix_band_length + (4 if self.directory.validity_code != 0 else 0))
        if parts != self.line_prefix_len:
            logger.debug('Line prefix length %d does not match its parts (%d) in %s.',
                         self.line_prefix_len, parts, self.source)

        self.nav_bytes = self.cal_bytes = self.aux_bytes = None
        self.nav = self.cal = self.aux = None
        self.has_navigation = self.has_calibration = self.has_auxiliary = False
        self.nav_size = self.cal_size = self.aux_size = 0
        self._read_blocks()

        logger.debug(f'Has navigation: {self.has_navigation},  has calibration: {self.has_calibration}, '
                     f'Has auxiliary: {self.has_auxiliary}')
        self.areaNav = None

        # setup some info that will be accessed while reading data etc
        self.datawidth = self.directory.bytes_per_element
        self.num_bands = self.directory.spectral_band_count
        self.elements = self.directory.elements
        self.claimed_lines = self.directory.lines

        self.line_data_len = self.num_bands * self.elements * self.datawidth
        self.line_len = self.line_prefix_len + self.line_data_len
        self.lines = self._actual_lines()
        self.data_block_len = self.lines * self.line_len

        self.prefix_len = self.line_prefix_len // self.datawidth
        self.data_len = self.data_block_len // self.datawidth

        if lazy:
            self.data = None
        else:
            self.data = self.__readdata__()

    def _block_end(self, offset):
        """End of the block starting at ``offset``: the next block's start, or the end of the file."""
        starts = [o for o in (self.navigation_offset, self.calibration_offset, self.aux_offset, self.data_offset)
                  if o > offset]
        return min(starts) if starts else self.sz

    def _read_blocks(self):
        """Read the nav, cal and aux blocks as 32-bit words, each from its own offset."""
        fmt = self.directory.byte_order
        for name, offset in (('nav', self.navigation_offset), ('cal', self.calibration_offset),
                             ('aux', self.aux_offset)):
            if offset <= 0 or offset >= self.sz:
                continue
            size = self._block_end(offset) - offset
            size -= size % INT32SIZE
            if size <= 0:
                continue
            raw = bytes(self.getbuffer()[offset:offset + size])
            words = struct.unpack(f'{fmt}{size // INT32SIZE}{self.INTEGER_FORMAT}', raw)
            setattr(self, f'{name}_size', size)
            setattr(self, f'{name}_bytes', raw)
            setattr(self, name, words)
            setattr(self, f'has_{"navigation" if name == "nav" else "calibration" if name == "cal" else "auxiliary"}',
                    True)

    def _actual_lines(self):
        """The number of lines really present in the data block.

        Truncated files are real, so this can be less than the directory's
        line count; everything that sizes an array uses this number.
        """
        if self.data_offset <= 0 or self.line_len <= 0:
            return self.claimed_lines
        actual = max(0, min(self.claimed_lines, (self.sz - self.data_offset) // self.line_len))
        if actual != self.claimed_lines:
            logger.warning('%s: directory claims %d lines but the data block only has %d.',
                           self.source, self.claimed_lines, actual)
        return actual

    @property
    def actual_lines(self):
        """Alias of :attr:`lines`: the number of lines present in the file."""
        return self.lines

    def _line_array(self):
        """The data block as a ``(lines, bytes per line)`` uint8 array, line prefixes included."""
        raw = self.getbuffer()[self.data_offset:self.data_offset + self.lines * self.line_len]
        return np.frombuffer(raw, dtype=np.uint8).reshape(self.lines, self.line_len)

    def _pixel_array(self):
        """The pixels as a native-endian ``(lines, elements, bands)`` array (BIP, as in the file)."""
        if self.datawidth not in (1, 2):
            raise NotImplementedError(f'{self.source}: {self.datawidth} bytes per element is not supported.')
        pixel_bytes = self._line_array()[:, self.line_prefix_len:self.line_prefix_len + self.line_data_len]
        if self.datawidth == 1:
            return pixel_bytes.reshape(self.lines, self.elements, self.num_bands)
        file_dtype = np.dtype(f'{self.directory.byte_order}u2')
        pixels = pixel_bytes.copy().view(file_dtype).reshape(self.lines, self.elements, self.num_bands)
        return pixels.astype(file_dtype.newbyteorder('='), copy=False)

    def __readdata__(self):
        """Read the image as a ``(bands, lines, elements)`` array."""
        return np.ascontiguousarray(self._pixel_array().transpose(2, 0, 1))

    def __getattribute__(self, name):
        if name == 'data':
            value = object.__getattribute__(self, name)
            if value is None:
                self.data = self.__readdata__()
                return self.data
            else:
                return object.__getattribute__(self, name)
        else:
            return object.__getattribute__(self, name)

    @property
    def istim(self):
        """
        Computes an approximation of scan time in miliseconds of a swath if the image was produced by
        GOES 8-15 instruments
        :return:
        """
        sens = self.areaDirectory.sensor
        if sens.startswith('GOES'):

            try:
                instr = sens[-2:]
                iinstr = int(instr)
                if iinstr >= 8:
                    return int(round((self.origNumElements * self.resolution[1]) * 954) / 20834.)
            except Exception as e:
                pass

    @property
    def imgbox(self):
        """
            Property
            Computes and returns the bounding box as a tuple (stline, endline, stelem, endelem)
        """
        start_line = self.directory.line_ul
        end_line = start_line + self.lines
        start_elem = self.directory.element_ul
        end_elem = start_elem + self.elements
        return start_line, end_line, start_elem, end_elem

    @property
    def resolution(self):
        return self.directory.line_res, self.directory.element_res

    @property
    def size(self):
        return self.lines, self.elements

    @property
    def npdtype(self):
        return f'{self.directory.byte_order}u{self.datawidth}'


    @property
    def file_name(self):
        """
            Property
            returns the file name of the AREA file
        """
        if self.source:
            _, name = os.path.split(self.source)
            return name


    @property
    def navigation(self):
        """The navigation object for this file's nav block, or None if it has none or is of an unsupported type."""
        if self.areaNav is None and self.nav is not None:
            nav = AreaNav(navblock=self.nav, byte_order=self.directory.byte_order, int_format=self.INTEGER_FORMAT)
            if nav is not None:
                nav.setImageStart(self.directory.line_ul, self.directory.element_ul)
                nav.setRes(*self.resolution)
                nav.setMag(1, 1)
                nav.setStart(0, 0)
                self.areaNav = nav
        return self.areaNav

    # ------------------------------------------------------------------
    # Pre-GVAR (SMS/GOES-1..7 VISSR/VAS) products

    @property
    def source_type(self):
        """The 4-character source type from the directory (``VISR``, ``VAS``, ``AAA``, ``GVAR``...)."""
        return self.directory.source_type.decode('ascii', errors='replace').strip('\x00').strip()

    @property
    def cal_type(self):
        """The 4-character calibration type from the directory (``RAW``, ``BRIT``...)."""
        return self.directory.cal_type.decode('ascii', errors='replace').strip('\x00').strip()

    @cached_property
    def aaa_cal(self):
        """The Mode AAA calibration block as ``{"sensor_source", "iab": {channel: (AB1, AB2)}, "ifab": {channel: IFAB}}``.

        None for any other kind of file, or if the block is too short to be one.
        """
        if self.source_type not in ('AAA', 'VAS') or self.cal is None or len(self.cal) < _AAA_CAL_MIN_WORDS:
            return None
        words = self.cal
        return {
            'sensor_source': int(words[0]),
            'iab': {chan: (int(words[3 + (chan - 1) * 2]), int(words[4 + (chan - 1) * 2])) for chan in range(1, 39)},
            'ifab': {chan: int(words[79 + (chan - 1)]) for chan in range(1, 39)},
        }

    @cached_property
    def band_list(self):
        """The channel (1-38) in each band slot of each line, shape ``(lines, bands)``; 0 for an unused slot.

        Read from the line prefix. The order and presence of channels can
        change from line to line, so every line is resolved on its own.
        None for files without such a prefix.
        """
        if self.source_type not in ('AAA', 'VAS'):
            return None
        band_list_len = self.line_prefix_len - _AAA_PRE_BANDLIST_LEN
        if band_list_len < max(self.num_bands, 1):
            logger.warning('%s: line prefix of %d bytes is too short for a per-line band list.',
                           self.source, self.line_prefix_len)
            return None
        lines = self._line_array()
        # only the first num_bands bytes are data, the rest is padding to a word
        return lines[:, _AAA_PRE_BANDLIST_LEN:_AAA_PRE_BANDLIST_LEN + self.num_bands].copy()

    @cached_property
    def mode_aa_cal(self):
        """The per-line Mode AA calibration entries, or None.

        A dict of ``(lines, 13)`` arrays: ``channel``, ``spin``, ``deltaf_code`` and ``ysubz``.
        """
        if self.source_type != 'VAS':
            return None
        d = self.directory
        n_entries, entry_bytes, header_bytes = 13, 8, 12
        if d.prefix_cal_length < header_bytes + n_entries * entry_bytes:
            logger.warning('%s: Mode AA calibration region is only %d bytes; calibration is unavailable.',
                           self.source, d.prefix_cal_length)
            return None
        start = (4 if d.validity_code != 0 else 0) + d.prefix_doc_length + header_bytes
        raw = self._line_array()[:, start:start + n_entries * entry_bytes]
        entries = raw.copy().view(f'{d.byte_order}i2').reshape(self.lines, n_entries, 4)
        return {
            'channel': entries[:, :, 0],
            'spin': entries[:, :, 1],
            'deltaf_code': entries[:, :, 2],
            'ysubz': entries[:, :, 3],
        }

    @cached_property
    def line_channel_spin(self):
        """The 13 channel/spin-count groups of each line's prefix: ``{"channel", "num_spins"}``, ``(lines, 13)``.

        A second copy of the channel numbers, useful to cross-check :attr:`band_list`. None without a band list.
        """
        if self.band_list is None:
            return None
        start = 4 + 512 + 12  # validity code, common documentation, day/time/scan header
        raw = self._line_array()[:, start:start + 104]
        groups = raw.copy().view(f'{self.directory.byte_order}u2').reshape(self.lines, 13, 4)
        return {'channel': groups[:, :, 0], 'num_spins': groups[:, :, 1]}

    def read_data(self):
        """The image of a single-band file as a ``(lines, elements)`` array."""
        if self.num_bands != 1:
            raise ValueError(f'{self.source}: read_data() is for single-band files, this one has {self.num_bands}; '
                             f'use read_channel() (Mode AAA) or read_ir_temperature_mode_aa() (Mode AA).')
        return self.data[0]

    def _gather_slot(self, slots, present):
        """Pick one band slot per line out of the pixel array; absent lines get 0."""
        pixels = self._pixel_array()
        out = np.take_along_axis(pixels, np.broadcast_to(slots[:, None, None], pixels.shape[:2] + (1,)),
                                 axis=2)[:, :, 0].copy()
        out[~present] = 0
        return out

    def read_channel(self, channel):
        """Raw counts of one VAS channel (1-38) of a Mode AAA file.

        Returns:
            ``(counts, present)``: counts of shape ``(lines, elements)``, and a boolean per line that is False where
            the channel is not in that line's band list (its counts are 0 and must not be used).
        """
        if self.band_list is None:
            raise ValueError(f'{self.source}: there is no per-line band list.')
        match = self.band_list == channel
        present = match.any(axis=1)
        return self._gather_slot(np.where(present, match.argmax(axis=1), 0), present), present

    def read_ir_radiance(self, channel):
        """Mode AAA radiance of a VAS channel as float32, NaN on lines without the channel."""
        if self.aaa_cal is None:
            raise ValueError(f'{self.source}: there is no Mode AAA calibration block (source type {self.source_type!r}).')
        lookup = 8 if channel == 13 else channel
        cal = self.aaa_cal
        if lookup not in cal['iab'] or lookup not in cal['ifab']:
            raise ValueError(f'{self.source}: channel {lookup} is not in the calibration block.')
        ab1, ab2 = cal['iab'][lookup]
        counts, present = self.read_channel(channel)
        return np.where(present[:, None], aaa_radiance(counts, ab1, ab2, cal['ifab'][lookup]), np.nan).astype(np.float32)

    def read_ir_temperature_aaa(self, channel):
        """Mode AAA brightness temperature (K) of a VAS channel as float32."""
        coeffs = vas_planck_coeffs(self.directory.satellite, channel)
        return aaa_temperature(self.read_ir_radiance(8 if channel == 13 else channel), coeffs)

    def _mode_aa_line_calibration(self, band):
        """Per-line ``(cal_present, delta_f, ysubz)`` for an IR band of a Mode AA file."""
        cal = self.mode_aa_cal
        match = mode_aa_code_to_band(cal['channel']) == band
        cal_present = match.any(axis=1)
        slot = np.where(cal_present, match.argmax(axis=1), 0)
        rows = np.arange(match.shape[0])
        deltaf_code = cal['deltaf_code'][rows, slot]
        table = np.array(MODE_AA_DELTA_F)
        valid = (deltaf_code >= 0) & (deltaf_code < len(table))
        delta_f = np.where(valid, table[np.clip(deltaf_code, 0, len(table) - 1)], 0)
        return cal_present, delta_f, cal['ysubz'][rows, slot].astype(np.float64)

    def _mode_aa_raw_counts(self, band):
        """``(raw, present)``: the raw counts of an IR band of a Mode AA file and the lines that carry it."""
        if self.band_list is None:
            raw = self.read_data().astype(np.float64)
            return raw, np.ones(raw.shape[0], dtype=bool)
        match = mode_aa_code_to_band(self.band_list) == band
        present = match.any(axis=1)
        return self._gather_slot(np.where(present, match.argmax(axis=1), 0), present).astype(np.float64), present

    def read_ir_temperature_mode_aa(self, band):
        """Mode AA brightness temperature (K) of IR band 1-12 as float32, NaN where it is missing."""
        if self.source_type != 'VAS':
            raise ValueError(f"{self.source}: not a Mode AA ('VAS') file.")
        if not 1 <= band <= 12:
            raise ValueError(f'Mode AA IR band must be 1-12 (got {band}); band 13 is visible.')
        if self.mode_aa_cal is None:
            raise ValueError(f'{self.source}: there is no Mode AA calibration data.')
        cal_present, delta_f, ysubz = self._mode_aa_line_calibration(band)
        raw, raw_present = self._mode_aa_raw_counts(band)
        valid_line = cal_present & raw_present
        t = mode_aa_temperature(raw, delta_f, ysubz, band)
        return np.where(valid_line[:, None], t, np.nan).astype(np.float32)

    def read_ir_temperature(self):
        """Mode A (single band) IR brightness temperature (K) as float32."""
        if self.directory.band != 'IR':
            raise ValueError(f'{self.source}: band is {self.directory.band!r}, not IR.')
        if self.source_type in ('AAA', 'VAS'):
            raise NotImplementedError(f'{self.source}: this is a multiband {self.source_type!r} file; use '
                                      f'read_ir_temperature_aaa() or read_ir_temperature_mode_aa().')
        counts = self.read_data()
        if counts.max() > 255:
            raise ValueError(f'{self.source}: raw values exceed 8 bits (max {counts.max()}).')
        return mode_a_ir_temperature(counts)

    def read_vis(self, calibration='counts'):
        """VIS counts, or ``calibration='unnormalized_reflectance'`` for a linear rescale to percent (0-100)."""
        if self.directory.band != 'VIS':
            raise ValueError(f'{self.source}: band is {self.directory.band!r}, not VIS.')
        counts = self.read_data().astype(np.float32)
        if calibration == 'counts':
            return counts
        if calibration == 'unnormalized_reflectance':
            return vis_unnormalized_reflectance(counts)
        raise ValueError(f'Unsupported VIS calibration {calibration!r}.')

    def get_nadir_sampling(self):
        """The sub-satellite point, altitude and nadir line/element spacing (see ``GOESNav.get_nadir_sampling``)."""
        return self._goes_nav().get_nadir_sampling()

    def _goes_nav(self):
        nav = self.navigation
        if nav is None or not hasattr(nav, 'get_nadir_sampling'):
            raise ValueError(f'{self.source}: there is no GOES navigation block.')
        return nav

    def get_lonlats(self, lines=None, elems=None):
        """Earth-locate pixels with the pre-GVAR GOES navigation.

        Args:
            lines: 0-based line indices in this file; all present lines by default.
            elems: 0-based element indices; all by default.

        Returns:
            ``(lon, lat)`` in degrees, shape ``(len(lines), len(elems))``, NaN off the earth.
        """
        nav = self._goes_nav()
        lines = np.arange(self.lines) if lines is None else lines
        elems = np.arange(self.elements) if elems is None else elems
        line_grid, elem_grid = np.meshgrid(lines, elems, indexing='ij')
        # file coordinates are 1-based in McIDAS; the navigation adds the ul line/element and resolution itself
        lat, lon = nav.to_latlon(line_grid + 1, elem_grid + 1)
        return lon, lat

    def __str__(self):
        if self.source:
            ads = f'AREA File: {self.source} \n'
        else:
            ads = '\n'
        ads += str(self.directory)
        return ads

    @property
    def nominal_datetime(self):
        return self.directory.nominal_time
    @property
    def scan_datetime(self):
        return self.directory.start_time
    @property
    def creation_datetime(self):
        return self.directory.creation_time

    def save(self, to_file_path=None, compress_level=None):
        """
        Saves the AreFile to disk. Optionally supply  a sring representing the coverage of  the AreaFile to be incorporated in the name
        Optiaonally compress to gzip
        :param folder: str, full path to folder where data will be saved
        :param coverage: str, the coverage ex SH, NH for goes images
        :param compress_level: int, if provided  the image will be compressed using gzip and specified compresslevel
        :return: str, the full path of the saved file
        """

        assert os.path.isabs(to_file_path), 'to_file_path arg %s is not an absolute path' % to_file_path
        try:
            import gzip
            with open(to_file_path, 'wb') as aft:
                aft.seek(0)
                aft.write(self.adir_bytes)
                if self.navigation_offset>0 and self.nav_size>0:
                    aft.seek(self.navigation_offset)
                    aft.write(self.nav_bytes)
                if self.calibration_offset > 0 and self.cal_size > 0:
                    aft.seek(self.calibration_offset)
                    aft.write(self.cal_bytes)
                if self.aux_offset > 0 and self.aux_size > 0:
                    aft.seek(self.aux_offset)
                    aft.write(self.aux_bytes)
                #if theer is not datLoc there si not AreaFile
                aft.seek(self.data_offset)
                #TODO line prefix bytes

                for bi in range(self.num_bands):
                    aft.write(self.data[bi,:,:].tobytes())


            if compress_level is not None:

                afpz = to_file_path + '.z'
                with open(to_file_path, 'rb') as f_in, gzip.open(afpz, 'wb', compresslevel=compress_level) as f_out:
                    for l in f_in:
                        f_out.write(l)
                os.remove(to_file_path)
                return afpz

            return  to_file_path
        except Exception as e:

            raise

    @property
    def imc_flip(self):
        """
        Computes

        Line Prefix COntent
              Validity Code: 4 bytes
              Documentation: 76 bytes. The documentation region
            consists of the following:
            - Block Header CRC: 2 bytes. Last three bits only;
            bit is set if Block Header copy is good. This data
            is usually 00,07.
            - Scan Status: 4 bytes (OGE Table 3-6)
            - Year, Day, Time from Block 0: 8 bytes
            - Block Header: 30 bytes (OGE Table 3-5)
              Bytes 17-24 now contain the time the block was disseminated from
              the ground station.
            - Additional Line Documentation: 32 bytes. 16 10-
            bit fields, right-justified. (OGE Table 3-7)
            The 6 bit field on the left hand side is not zero filled.
            To get the 10 bit field a logical AND against 03FF (hex) must be used.

        :return:
        """
        if self.directory.source_type.decode() in ['GVAR', 'VISR']:

            if self.directory.line_prefix_length == 0:

                raise Exception('Cannot extract imc anf flip flags  because the image lines have no  documentation ' )
        else:
            raise  Exception('The Area file was not produced from a GVAR stream. The IMC anf FLIP flags do not apply on this case.')

        import ctypes as C

        if self.directory.byte_order == '>':
            klass = C.BigEndianStructure
        else:
            klass = C.LittleEndianStructure

        class ISCAN(klass):

            """
                ' 0 Frame Start',
                ' 1 Frame End',
                ' 2 Frame break ( lines lost )',
                ' 3 Line break ( Pixels lost )',
                ' 4 Priority 1 frame data ',
                ' 5 Priority 2 frame data ',
                ' 6 East-to-west scan',
                ' 7 South-to-north scan',
                ' 8 IMC active',
                ' 9 Lost header block',
                '10 Lost trailer block',
                '11 Lost telemetry data',
                '12 (Star sense) time break',
                '13 Side 2 (secondary) active',
                '14 Visible normalization active',
                '15 IR calibration active',
                '16 yaw-flipped mode (GOES-10)',
                '17 IR detector 1 data not valid',
                '18 IR detector 2 data not valid',
                '19 IR detector 3 data not valid',
                '20 IR detector 4 data not valid',
                '21 IR detector 5 data not valid',
                '22 IR detector 6 data not valid',
                '23 IR detector 7 data not valid',
                '24 Visible detector 1 data not valid',
                '25 Visible detector 2 data not valid',
                '26 Visible detector 3 data not valid',
                '27 Visible detector 4 data not valid',
                '28 Visible detector 5 data not valid',
                '29 Visible detector 6 data not valid',
                '30 Visible detector 7 data not valid',
                '31 Visible detector 8 data not valid'
            """
            _pack_ = 1

            _fields_ = (('frame_start', C.c_uint8, 1),  # 0
                        ('frame_end', C.c_uint8, 1),  # 1
                        ('frame_break', C.c_uint8, 1),  # 2
                        ('line_break', C.c_uint8, 1),  # 3
                        ('priority1', C.c_uint8, 1),  # 4
                        ('priority2', C.c_uint8, 1),  # 5
                        ('west_to_east', C.c_uint8, 1),  # 6
                        ('north_to_south', C.c_uint8, 1),  # 7
                        ('imc_active', C.c_uint8, 1),  # 8
                        ('header_lost', C.c_uint8, 1),  # 9
                        ('trailer_lost', C.c_uint8, 1),  # 10
                        ('telemetry_lost', C.c_uint8, 1),  # 11
                        ('time_break', C.c_uint8, 1),  # 12
                        ('side2', C.c_uint8, 1),  # 13
                        ('vis_normalization', C.c_uint8, 1),  # 14
                        ('ir_calibration', C.c_uint8, 1),  # 15
                        ('yaw_flip', C.c_uint8, 1),  # 16
                        ('irdet1_invalid', C.c_uint8, 1), ('irdet2_invalid', C.c_uint8, 1),
                        ('irdet3_invalid', C.c_uint8, 1), ('irdet4_invalid', C.c_uint8, 1),
                        ('irdet5_invalid', C.c_uint8, 1), ('irdet6_invalid', C.c_uint8, 1),
                        ('irdet7_invalid', C.c_uint8, 1), ('visdet1_invalid', C.c_uint8, 1),
                        ('visdet2_invalid', C.c_uint8, 1), ('visdet3_invalid', C.c_uint8, 1),
                        ('visdet4_invalid', C.c_uint8, 1), ('visdet5_invalid', C.c_uint8, 1),
                        ('visdet6_invalid', C.c_uint8, 1), ('visdet7_invalid', C.c_uint8, 1),
                        ('visdet8_invalid', C.c_uint8, 1),)

            def bstring(self):
                return ''.join([str(getattr(self, e[0])) for e in self._fields_])


        '''
        line prefix length = doc + cal + band + 4 (if valcode is present)
        line data length = nbands*nele*nbytes
        line length = line prefix length + line data length
        data block length = nlines * line length
        '''

        try:


            mask = self.__datamask__().ravel()
            data = self.data


            # #
            # for b in range(self.num_bands):

            # t he line prefix is 4 bytes, so we view it as 1 byte
            prfx1 = self.data_block[~mask].reshape(self.lines, self.prefix_len).view('u1')
            # some story

            # valid = prfx1.view('%si4' % self.BYTE_ORDER)

            if self.directory.validity_code:  # this is the validity code, if its value is present in yhe header
                # one vcould also use byte position 23 or 24 , to chcek valid lines using block0 header
                # valid_lines_mask = valid[:,0] == self.valcode
                # valid_lines = prfx1[:,INT32SIZE:][valid_lines_mask]
                lines = prfx1[:, INT32SIZE:]
            else:
                lines = prfx1
            valid_lines_mask = lines[:, 22].view(bool)
            valid_lines = lines[valid_lines_mask]
            if not lines.size > 0:
                raise Exception('The Area files dodo not have any valid lines as per block0 data form the line prefix')
            aline = valid_lines[0]
            astatus = ISCAN.from_buffer_copy(aline[2:6])
            return astatus.imc_active, astatus.yaw_flip  # print aline  # BO = self.BYTE_ORDER  # HEADER = [('BlockID', 'B'), ('WordSize', 'B'), ('WordCount', 'H'), ('ProductID', 'H'),  #           ('RepeatFlag', 'B'), ('VersionNumber', 'B'), ('DataValid', 'B'), ('AsciiBin', 'B'),  #           ('SPS_ID', 'B'), ('RangeWord', 'B'), ('BlockCount', 'H'), ('Spares_1', '2B'),  #           ('SPS_Time', '8B'), ('Spares_2', '4B'), ('CRC', 'H')]  #  # HEADER = [(e[0], BO + e[1]) for e in HEADER]  # for i in range(10):  #     # h = np.frombuffer(valid_lines[i,:],dtype=HEADER, count=1, offset=14)  #     # print h['RangeWord'].item()>>4  #     from pygvar.classes import ISCAN  #     b = valid_lines[i, 2:6]  #     a = ISCAN.from_buffer_copy(b)  #     print a.imc_active, a.yaw_flip

            # first_line_scandatetime = bcdtime2datetime(bytes=valid_lines[0, 6:14], byte_order=self.BYTE_ORDER)  # last_line_scandatetime = bcdtime2datetime(bytes=valid_lines[-1, 6:14], byte_order=self.BYTE_ORDER)

        except Exception as e:
                raise Exception(f'Failed to fetch IMC and FLIP flags from {self.__class__.__name__} object, Error is {e}' )
    @property
    def resolution(self):
        return self.directory.line_res, self.directory.element_res

if __name__ == '__main__':

    #import pylab
    logger = logging.getLogger()
    logging.basicConfig()
    logger.setLevel('DEBUG')
    afp = '/media/d/data/satellite_goes_r/adde/incoming/0000000001.OR_ABI-L1b-RadF-M3C07_G16_s2018166103041000000_e2018166104118000000_c2018166104122000000.area'
    afpz = '/media/d/data/satellite_goes_r/adde/incoming/0000000001.OR_ABI-L1b-RadF-M3C13_G16_s2018226120048000000_e2018226121125000000_c2018226121129000000.area.z'
    #afp = '/home/jano/sw/06091408 GOES-13 Imager Complete Thu 09-Jun-2016 1408.VIN'

    with AreaFile(source=afp, lazy=True) as a:
        logger.info(a)
    # import gzip
    # with gzip.open(afpz) as cf:
    #     with AreaFile(source=cf.read(), lazy=True) as a:
    #         logger.info(a)
    #         band = a.directory.bands[0]
    #         for band in range(1,17):
    #             band_chunk_start = 1 + (band - 1) * 18
    #             # band_chunki_start = ninstr_bytes + (band-1)* n_calib_elements*4
    #             band_chunk_end = band_chunk_start + 18
    #
    #             calib_info = a.cal[band_chunk_start:band_chunk_end]
    #             print (dict(zip(range(1,len(calib_info)+1),calib_info)))

            # for b in a.data:
            #     pylab.imshow(b)
            #     pylab.show()









