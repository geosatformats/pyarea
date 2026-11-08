"""Builds a GMSX navigation block (a GMS-5 like geometry) for tests, by encoding the values the way sv0100 reads them."""
import math

DTIMS = 50000.0  # MJD of the start of the scan


def _put(stream, offset, nbytes, power, value):
    """Store ``value`` as the sign and magnitude integer ``value * 10**power`` in ``nbytes`` big endian bytes."""
    magnitude = int(round(abs(value) * 10.0 ** power))
    assert magnitude < 1 << (8 * nbytes - 1), (offset, value)
    raw = bytearray(magnitude.to_bytes(nbytes, "big"))
    if value < 0:
        raw[0] |= 0x80
    stream[offset:offset + nbytes] = raw


def _to_words(stream):
    """Spread the byte stream over the 640 words of the block, as GMSXnav's constructor reads it back."""
    def word(chunk):
        return int.from_bytes(chunk, "big", signed=True)

    words = [0] * 640
    words[0] = word(b"GMSX")
    for i in range(126):
        words[1 + i] = word(stream[4 * i:4 * i + 4])
    position = 504
    for block in range(4):
        start = 128 + 127 * block + block  # 128, 256, 384, 512
        for i in range(127):
            words[start + i] = word(stream[position:position + 4])
            position += 4
    for marker in (127, 255, 383, 511, 639):
        words[marker] = word(b"MORE")
    return tuple(words)


def make_gmsx_navblock(beta0=math.radians(319.96), beta_slope=math.radians(-10.0), sublon=140.0, sat_lon=140.0,
                       spin_rpm=100.0, sun_ra=0.04, sun_ra_rate=-10.0, sun_dec=5.0, gst0=358.8,
                       attitude=(0.001, 0.0015)):
    """A navigation block with GMS-5 like numbers: a visible frame of 10000 lines x 9164 pixels (35 microradians),
    the infrared one a quarter of that, and a scan that starts at DTIMS and takes 12.5 minutes.

    GMSXnav navigates with the visible values (mode 1), the infrared ones are only decoded.
    The values at DTIMS are ``beta0`` (radians, the sun-Earth angle), ``sun_ra`` and ``gst0`` (degrees); the defaults
    make the scan of the Earth centred on the sub-satellite point with 0.04 and 358.8 degrees so that the sun right
    ascension and the sidereal time wrap around inside the interval the scan falls in.
    The attitude sub-blocks 1 to 9 and the orbit sub-blocks 1 to 7 are written where VisAD reads them (offset
    ``(i - 1) * stride + start`` with i from 0, so sub-block 0 overlaps other words and is left alone).
    """
    stream = bytearray(2536 + 64)
    put = lambda offset, n, power, value: _put(stream, offset, n, power, value)  # noqa: E731
    put(0, 6, 8, DTIMS)
    put(240, 6, 8, spin_rpm)
    for offset, value in ((6, 35e-6), (10, 140e-6)):
        put(offset, 4, 8, value)
    for offset, value in ((14, 35e-6), (18, 140e-6)):
        put(offset, 4, 10, value)
    for offset, value in ((22, 5000.5), (26, 1250.5), (110, 1250.5), (114, 1250.5),
                          (30, 4582.5), (34, 1145.5), (118, 1145.5), (122, 1145.5)):
        put(offset, 4, 4, value)
    for offset, value in ((38, 4), (42, 2), (46, 10000), (50, 2500), (54, 9164), (58, 2291)):
        put(offset, 4, 0, value)
    for offset in (62, 66, 70):
        put(offset, 4, 10, 0.0)
    for offset in (74, 90, 106):
        put(offset, 4, 7, 1.0)
    for offset in (78, 82, 86, 94, 98, 102):
        put(offset, 4, 10, 0.0)
    put(198, 6, 6, sublon)
    put(204, 6, 6, 0.0)

    for i in range(1, 10):  # attitude sub-blocks: every 0.01 day from DTIMS - 0.01 at i = 1
        j = (i - 1) * 48 + 256
        t = DTIMS + 0.01 * (i - 2)
        put(j, 6, 8, t)
        put(j + 12, 6, 8, attitude[0])
        put(j + 18, 6, 11, attitude[1])
        put(j + 24, 6, 8, (beta0 + beta_slope * (t - DTIMS)) % (2 * math.pi))
        put(j + 30, 6, 8, spin_rpm)
    for i in range(1, 8):  # orbit sub-blocks: every 0.02 day from DTIMS - 0.03 at i = 1
        j = (i - 1) * 200 + 736
        t = DTIMS + 0.02 * (i - 1) - 0.03
        radius = 42164.0e3
        lon = math.radians(sat_lon)
        put(j, 6, 8, t)
        put(j + 48, 6, 6, radius * math.cos(lon) + 10.0 * i)
        put(j + 54, 6, 6, radius * math.sin(lon) - 7.0 * i)
        put(j + 60, 6, 6, 2000.0 * i)
        put(j + 84, 6, 8, (gst0 + 360.985647 * (t - DTIMS)) % 360.0)
        put(j + 102, 6, 8, (sun_ra + sun_ra_rate * (t - DTIMS)) % 360.0)
        put(j + 108, 6, 8, sun_dec)
        put(j + 128, 6, 12, 1.0)
        put(j + 152, 6, 12, 1.0)
        put(j + 176, 6, 12, 1.0)
        for offset, power in ((134, 14), (140, 14), (146, 14), (158, 16), (164, 12), (170, 16)):
            put(j + offset, 6, power, 0.0)
    return _to_words(stream)
