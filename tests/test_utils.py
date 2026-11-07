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


def test_nav_bcd2datetime_matches_for_a_little_endian_navigation_block():
    """AreaNav.bcd2datetime takes words as they were read from the file, so the file's byte order does not matter."""
    from tests.test_gvar_nav import load, make_nav

    nav = make_nav(load("goes8_gvar_nav.json"))
    assert nav.byte_order == "<"
    assert nav.bcd2datetime(*BCD_WORDS) == BCD_TIME
    assert nav.flywheel is False
