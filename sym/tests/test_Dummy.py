# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
from .. import Backend
from . import AVAILABLE_BACKENDS


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_Dummy(key):
    be = Backend(key)
    d0 = be.Dummy()
    d1 = be.Dummy()
    assert d0 == d0
    assert d0 != d1
