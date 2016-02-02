# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
from .. import Backend


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Dummy(key):
    be = Backend(key)
    d0 = be.Dummy()
    d1 = be.Dummy()
    assert d0 == d0
    assert d0 != d1
