"""Tests for pyarea.utils."""
import numpy as np
import pytest

from pyarea.utils import is_mcmissing


@pytest.mark.parametrize(("word", "missing"), [(-2139062144, True), (0x80808080, True), (np.int32(-2139062144), True),
                                               (0, False), (-1, False), (2139062144, False)])
def test_is_mcmissing(word, missing):
    """The McIDAS missing value is recognised whether the word was read signed or unsigned."""
    assert is_mcmissing(word) is missing
