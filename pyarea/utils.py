import numpy as np
import datetime
import struct
import ctypes as C
#binary values representing a missing value in McIDAS
MCMISSING = int('0x80808080', 16)


def is_mcmissing(word):
    """True if a 32-bit directory or navigation word is the McIDAS "missing" value.

    ``MCMISSING`` is the unsigned bit pattern, but the words of a block are usually read as signed integers, where
    the same pattern is -2139062144; both are recognised.
    """
    return (int(word) & 0xFFFFFFFF) == MCMISSING
INT32SIZE = np.int32().itemsize

def yyyddd_hhmmss2datetime(yyyddd, hhmmss ):
    """
        Convert year_day (yyyddd) and time (hhmmss) to datetime
        @:args
            @yyyddd, int, represents the year and day-of-year. The yyy values are the actual year modulo 1900
            @hhmmss, int, represents the time in hhmmss
        @returns
            @datetime, the corresponding datetime object
    """

    shhmmss = str(hhmmss)
    ls = len(shhmmss)
    if ls != 6: #use custom logic
        if ls <= 2: #exceptionally, only seconds are present
            hh = 0
            mm = 0
            ss = int(shhmmss)

        elif ls == 3:
            hh = 0
            mm = int(shhmmss[0])
            ss = int(shhmmss[1:])

        elif ls == 4:
            hh = 0
            mm = int(shhmmss[:2])
            ss = int(shhmmss[2:])

        elif ls == 5:
            hh = int(shhmmss[0])
            mm = int(shhmmss[1:3])
            ss = int(shhmmss[3:])

        else :
            raise ValueError(f'Unable to convert integer argument {hhmmss} into hour, minute, second')



    else:
        hh = int(shhmmss[:2])
        mm = int(shhmmss[2:4])
        ss = int(shhmmss[4:])

    year = ((yyyddd // 1000) % 1900) + 1900
    yday = yyyddd % 1000
    month, day = get_month_day(year, yday, True)
    return datetime.datetime(year, month, day,hour=hh,minute=mm,second=ss)

def get_month_day(year, day, one_based=False):
        if one_based:  # if Jan 1st is 1 instead of 0
                day -= 1
        dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day)
        return dt.month, dt.day


def gould2native(number):
        """
            Converts a number from Gould floating point format to native floating point
            an example conversion:
            input integer 4 bytes = 3201977607
            input value (hex): BD DA 4D 07
            a) convert to binary:
                1011 1110 1101 1010 0100 1101 0000 0111
            b) sign bit is set, so take twos complement:
                0100 0001 0010 0101 1011 0010 1111 1001
            c) convert this back to hex: 41 25 B2 F9
            d) mantissa = 2470649
            e) exponent = 65
            f) tempVal = mantissa / 16 exp (70 - 65)
            g) outputVal = tempVal * sign = -2.3561944

        """

        #the input is an 4 byte int of a hexadecimal float in Gould format
        #we need to prepare byte masks in order to extract the most significant byte (byte 4)
        # as well the first three bytes. I am not sure but my feeling is that OGE doc implies Gould is
        #big endian byte order and thus the section 3.5.4 labels bytes in an opposite way (byte1 is the most significant byte)
        #however the nav header is in Little endian and I use its labeling for bytes
        #prepare masks
        byte012_mask = int('0xffffff',16) # 0000 0000 1111 1111 1111 1111 1111 1111
        byte3_mask = int('0xFFFFFFFF', 16) # 1111 1111 0000 0000 0000 0000 0000 0000
        sign_bit_mask = int('0x80000000',16) # 10000000000000000000000000000000
        exponent_mask = int('0x7F000000', 16) # '01111111000000000000000000000000
        #compute sign
        sign = -1 if number & sign_bit_mask else 1
        if sign < 0:
            #two's complement for negative numbers
            number = (number ^ byte3_mask) + 1
        #compute exonent
        exponent = (number & exponent_mask) >> 24
        if exponent == 0: #account for bias
            exponent = 64
        #compute matissa
        mantissa = number & byte012_mask
        tempVal = 16 ** float(70 - exponent)
        nativeVal = mantissa / tempVal
        nativeVal = nativeVal * sign

        return nativeVal

def mcPI2F(value):
    val = -value if value < 0 else value
    fvalue = float(val/10000) + float((val/100) % 100) / 60. + (val % 100) / 3600.
    return -fvalue if value < 0 else fvalue

def mcF2PI(value):
    fval = -value if value < 0 else value
    j = int(3600*fval + .5)
    val = 10000 * (j/3600) + 100 * ((j/60) % 60 ) + j % 60
    return -val if value < 0 else val

def leapyear(year):
    return 366 - ((year%4) + 3)/4


_field_names_ = names = 'year_1000', 'year_100', 'year_10', 'year_1', 'day_100', 'day_10', 'day_1', 'hour_10', 'hour_1', 'min_10', 'min_1', 'sec_10', 'sec_1', 'msec_100', 'msec_10', 'msec_1'

_field_types_ = [C.c_uint8]* len(_field_names_)
class BCDDateTime(C.BigEndianStructure, ):

    _fields_ = tuple([(_field_names_[i], _field_types_[i], 4) for i in range(len(_field_names_))])

    @staticmethod
    def get_month_day(year, day, one_based=False):
        if one_based:  # if Jan 1st is 1 instead of 0
                day -= 1
        dt = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day)
        return dt.month, dt.day

    def python_value(self):
        # compose year
        d = self.fields
        year = d['year_1000'] * 1000 + d['year_100'] * 100 + d['year_10'] * 10 + d['year_1']
        yday = d['day_100'] * 100 + d['day_10'] * 10 + d['day_1']
        month, day = self.get_month_day(year, yday, True)
        hour = d['hour_10'] * 10 + d['hour_1']
        minute = d['min_10'] * 10 + d['min_1']
        second = d['sec_10'] * 10 + d['sec_1']
        milisecond = d['msec_100'] * 100 + d['msec_10'] * 10 + d['msec_1']
        return datetime.datetime(year, month, day, hour, minute, second, milisecond*1000)

    @property
    def fields(self):
        d = {}

        for fdef in self._fields_:
            v = getattr(self, fdef[0])
            try:
                d[fdef[0]] = v.fields
            except AttributeError:
                cls_name = fdef[1].__class__.__name__
                if cls_name == 'PyCArrayType':
                    return dict([('%s[%d]' % (fdef[0], i), repr(e)) for i, e in enumerate(v)])
                else:
                    d[fdef[0]] = v

        return d

_BCD_NAMES = ('year_1000', 'year_100', 'year_10', 'year_1', 'day_100', 'day_10', 'day_1', 'hour_10', 'hour_1',
              'min_10', 'min_1', 'sec_10', 'sec_1', 'msec_100', 'msec_10', 'msec_1')


def bcd_words_to_datetime(word1, word2):
    """Decode a BCD time stored in two 32-bit words, as ``YYYYDDDH`` and ``HMMSSmmm`` hexadecimal digits.

    The words are integers, i.e. already decoded in the byte order of the file they came from, so this does not
    depend on that byte order. ``0x19962562, 0x14515000`` is 1996 day 256 (September 12), 21:45:15.000.
    """
    chars = struct.pack('>2I', int(word1) & 0xFFFFFFFF, int(word2) & 0xFFFFFFFF)
    nibbles = [value for byte in chars for value in (byte >> 4, byte & 15)]
    d = dict(zip(_BCD_NAMES, nibbles))
    year = d['year_1000'] * 1000 + d['year_100'] * 100 + d['year_10'] * 10 + d['year_1']
    yday = d['day_100'] * 100 + d['day_10'] * 10 + d['day_1']
    month, day = get_month_day(year, yday, True)
    hour = d['hour_10'] * 10 + d['hour_1']
    minute = d['min_10'] * 10 + d['min_1']
    second = d['sec_10'] * 10 + d['sec_1']
    millisecond = d['msec_100'] * 100 + d['msec_10'] * 10 + d['msec_1']
    return datetime.datetime(year, month, day, hour, minute, second, millisecond * 1000)  # microseconds


def bcdtime2datetime(bytes=None, byte_order=None):
    """Convert a BCD time given as the 8 bytes it occupies in a file (two 32-bit words) to a datetime.

    ``byte_order`` is the byte order of the file's 32-bit words, ``'>'`` or ``'<'``. The bytes of each word are in
    that order, so in a little-endian file the digits of each word appear reversed.
    """
    assert byte_order in ('>', '<'), 'byte_order has to be supplied, as > or <'
    word1, word2 = struct.unpack(f'{byte_order}2I', bytearray(bytes))
    return bcd_words_to_datetime(word1, word2)


def julday(month, day, year, hour=0, minute=0, second=0):
    """
    Julian date from month, day, and year (can be scalars or arrays)

    Parameters
    ----------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32
        Year.
    hour : numpy.ndarray or int32, optional
        Hour.
    minute : numpy.ndarray or int32, optional
        Minute.
    second : numpy.ndarray or int32, optional
        Second.



    Returns
    -------
    jul : numpy.ndarray or double
        Julian day.
    """
    month = np.array(month)
    day = np.array(day)
    inJanFeb = month <= 2
    jy = year - inJanFeb
    jm = month + 1 + inJanFeb * 12

    jul = np.int32(np.floor(365.25 * jy) +
                   np.floor(30.6001 * jm) + (day + 1720995.0))
    ja = np.int32(0.01 * jy)
    jul += 2 - ja + np.int32(0.25 * ja)

    jul = jul + hour / 24.0 - 0.5 + minute / 1440.0 + second / 86400.0

    return jul


def caldat(julian):
    """
    Calendar date (month, day, year) from julian date, inverse of 'julday()'
    Return value:  month, day, and year in the Gregorian
    Works only for years past 1582!

    Parameters
    ----------
    julian : numpy.ndarray or double
        Julian day.

    Returns
    -------
    month : numpy.ndarray or int32
        Month.
    day : numpy.ndarray or int32
        Day.
    year : numpy.ndarray or int32
        Year.
    """
    jn = np.int32(((np.array(julian) + 0.000001).round()))

    jalpha = np.int32(((jn - 1867216) - 0.25) / 36524.25)
    ja = jn + 1 + jalpha - (np.int32(0.25 * jalpha))
    jb = ja + 1524
    jc = np.int32(6680.0 + ((jb - 2439870.0) - 122.1) / 365.25)
    jd = np.int32(365.0 * jc + (0.25 * jc))
    je = np.int32((jb - jd) / 30.6001)

    day = jb - jd - np.int32(30.6001 * je)
    month = je - 1
    month = (month - 1) % 12 + 1
    year = jc - 4715
    year = year - (month > 2)

    return month, day, year


def compose_area_name(adir=None,coverage=None, ):

    band = adir.bands[0]
    nom_dt = adir.nominalTime
    date_str = nom_dt.date().strftime('%Y%m%d')
    time_str = nom_dt.time().strftime('%H%M')
    cov = '_%s' % coverage if coverage else ''
    return '%s_B%s_%s%s%s.pyarea' % (adir.sensor, band, date_str, time_str, cov)