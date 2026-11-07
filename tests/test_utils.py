"""Tests for pyarea.utils."""
import datetime
import struct

import numpy as np
import pytest

from pyarea.utils import bcd_words_to_datetime, bcdtime2datetime, is_mcmissing


@pytest.mark.parametrize(("word", "missing"), [(-2139062144, True), (0x80808080, True), (np.int32(-2139062144), True),
                                               (0, False), (-1, False), (2139062144, False)])
def test_is_mcmissing(word, missing):
    """The McIDAS missing value is recognised whether the word was read signed or unsigned."""
    assert is_mcmissing(word) is missing


# 1996 day 256 (12 September), 21:45:15.000, as the two 32-bit words YYYYDDDH and HMMSSmmm
BCD_WORDS = (0x19962562, 0x14515000)
BCD_TIME = datetime.datetime(1996, 9, 12, 21, 45, 15)


@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_bcdtime2datetime_in_either_byte_order(byte_order):
    """The 8 bytes of a little-endian file have the digits of each word reversed; both decode to the same time."""
    raw = struct.pack(f"{byte_order}2I", *BCD_WORDS)
    assert bcdtime2datetime(bytes=raw, byte_order=byte_order) == BCD_TIME
    assert bcdtime2datetime(bytes=list(raw), byte_order=byte_order) == BCD_TIME
    assert bcdtime2datetime(bytes=np.frombuffer(raw, dtype=np.uint8), byte_order=byte_order) == BCD_TIME


def test_bcdtime2datetime_needs_a_byte_order():
    with pytest.raises(AssertionError):
        bcdtime2datetime(bytes=bytes(8), byte_order=None)


def test_bcd_words_accept_signed_words_and_milliseconds():
    """Words read as signed integers (negative when the top bit is set) decode the same as unsigned ones."""
    assert bcd_words_to_datetime(*BCD_WORDS) == BCD_TIME
    # 09:00:00.000 puts the hour's units digit, 9, in the top nibble of the second word: its top bit is set
    w1, w2 = 0x19962560, 0x90000000
    signed2 = w2 - (1 << 32)
    assert signed2 < 0
    expected = datetime.datetime(1996, 9, 12, 9, 0, 0)
    assert bcd_words_to_datetime(w1, w2) == expected
    assert bcd_words_to_datetime(w1, signed2) == expected
    assert bcd_words_to_datetime(0x19962562, 0x14515123) == BCD_TIME.replace(microsecond=123000)


def _visad_epoch_minutes(e0, e1):
    """Port of the BCD loop in VisAD's GVARnav.java (integer division as in Java): minutes since 1950-01-01."""
    t0 = t1 = 0
    power10 = 1
    for _ in range(8):
        t0 += (e0 & 0xF) * power10
        t1 += (e1 & 0xF) * power10
        e0 >>= 4
        e1 >>= 4
        power10 *= 10
    year = t0 // 10000
    day = int((t0 - year * 10000) * 0.1)
    iab = (t0 - year * 10000 - day * 10) * 10
    nbc = t1 // 10000000
    iac = t1 - nbc * 10000000
    hour = iab + nbc
    minutes = int(iac * 0.00001)
    seconds = (t1 - ((t1 - iac) + minutes * 100000)) * 0.001
    j = day + 1461 * (year + 4799) // 4 - 3 * ((year + 4899) // 100) // 4 - 2465022
    return j * 1440.0 + hour * 60.0 + minutes + seconds / 60.0


def test_bcd_decoder_agrees_with_visad_gvarnav():
    rng = np.random.default_rng(1)
    start = datetime.datetime(1975, 1, 1)
    for offset in rng.integers(0, 45 * 365 * 86400, 2000):
        t = start + datetime.timedelta(seconds=int(offset), milliseconds=int(rng.integers(0, 1000)))
        digits1 = f"{t.year:04d}{t.timetuple().tm_yday:03d}{t.hour // 10}"
        digits2 = f"{t.hour % 10}{t.minute:02d}{t.second:02d}{t.microsecond // 1000:03d}"
        w1, w2 = int(digits1, 16), int(digits2, 16)
        decoded = bcd_words_to_datetime(w1, w2)
        assert decoded == t
        minutes = (decoded - datetime.datetime(1950, 1, 1)).total_seconds() / 60.0
        assert abs(_visad_epoch_minutes(w1, w2) - minutes) < 1e-6
