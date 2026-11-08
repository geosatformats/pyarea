"""AreaNav(...) picks the navigation class from the first nav word, like AREAnav.makeAreaNav in VisAD."""
import struct

import pytest

from pyarea.nav import ABINNav, AreaNav, GMSXNav, GOESNav, GVARNav, LALONav, RECTNav


def _block(code, order=">"):
    word = struct.unpack(f"{order}i", code.encode("ascii"))[0] if isinstance(code, str) else code
    return tuple([word] + [0] * 639)


def _pick(code, order=">"):
    # __new__ only, so that the subclass constructors do not have to accept an empty block
    return AreaNav.__new__(AreaNav, navblock=_block(code, order), byte_order=order, int_format="i")


@pytest.mark.parametrize("order", [">", "<"])
@pytest.mark.parametrize(
    ("code", "klass"),
    [("GVAR", GVARNav), ("GOES", GOESNav), ("RECT", RECTNav), ("GMSX", GMSXNav), ("LALO", LALONav), ("ABIN", ABINNav)])
def test_nav_type_selects_the_class(code, klass, order):
    nav = _pick(code, order)
    assert type(nav) is klass
    assert nav.navtype == code and nav.byte_order == order


@pytest.mark.parametrize("code", ["GVA ", "MSG ", "PS  ", "    ", "ZZZZ", "gvar", 0, -2139062144])
def test_unknown_nav_types_are_not_matched_by_prefix(code):
    with pytest.raises(NotImplementedError, match="not implemented"):
        _pick(code)


def test_the_selection_does_not_depend_on_the_order_of_the_subclasses():
    assert {type(_pick("GOES")) for _ in range(20)} == {GOESNav}


def test_navblock_and_byte_order_are_required():
    with pytest.raises(RuntimeError):
        AreaNav(byte_order=">", int_format="i")
    with pytest.raises(RuntimeError):
        AreaNav(navblock=_block("GVAR"), int_format="i")
    with pytest.raises(RuntimeError):
        AreaNav(navblock=_block("GVAR"), byte_order=">")


@pytest.mark.parametrize("code", ["LALO", "ABIN"])
def test_navigations_that_are_not_ported_say_so(code):
    with pytest.raises(NotImplementedError, match=code):
        AreaNav(navblock=_block(code), byte_order=">", int_format="i")


def test_gmsx_block_is_accepted_on_python_3():
    """The constructor compared bytes with a str and packed a str, so no GMSX block was ever accepted."""
    nav = AreaNav(navblock=_block("GMSX"), byte_order=">", int_format="i")
    assert type(nav) is GMSXNav


def test_gmsx_rejects_other_nav_types():
    with pytest.raises(Exception, match="Invalid navigation type"):
        GMSXNav(navblock=_block("GVAR"), byte_order=">", int_format="i")
