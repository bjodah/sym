# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import pytest
from .. import Backend
from . import AVAILABLE_BACKENDS


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_symarray(key):
    be = Backend(key)
    x0, x1, x2 = be.symarray('x', 3)

    assert be.symarray('x', (3, 2)).shape == (3, 2)
