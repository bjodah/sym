# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest

from .. import Backend
from . import AVAILABLE_BACKENDS


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_banded_jacobian(key):
    be = Backend(key)
    n = 3
    x = be.real_symarray('x', n)
    exprs = [-x[0]] + [x[i-1] - x[i] for i in range(1, n-1)] + [x[-2]]
    bj = be.banded_jacobian(exprs, x, 1, 0)
    cb = be.Lambdify(x, bj)
    inp = np.arange(3.0, 3.0 + n)
    out = cb(inp)
    assert np.allclose(out, [[-1, -1, 0], [1, 1, 0]])
