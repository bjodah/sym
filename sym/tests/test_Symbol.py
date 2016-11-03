# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
from .. import Backend


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Symbol(key):
    be = Backend(key)
    x = be.Symbol('x')
    y = be.Symbol('y')
    assert x != y
    assert x == be.Symbol('x')
    assert x != be.Symbol('y')
    assert x - x == 0
    assert x*x == x**2  # this is starting to look like a CAS requirement
