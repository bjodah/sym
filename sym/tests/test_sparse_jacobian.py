# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest

from .. import Backend
from . import AVAILABLE_BACKENDS


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_sparse_jacobian_csr(key):
    be = Backend(key)
    n = 3
    x = be.real_symarray('x', n)
    exprs = [-x[0]] + [x[i-1] - x[i] for i in range(1, n-1)] + [x[-2]]
    sj, colptrs, rowvals = be.sparse_jacobian_csc(exprs, x)
    cb = be.Lambdify(x, sj)
    inp = np.arange(3.0, 3.0 + n)
    out = cb(inp)
    assert np.allclose(out, [-1, 1, -1, 1])
    assert np.all(colptrs == np.array([0, 2, 4, 4], dtype=int))
    assert np.all(rowvals == np.array([0, 1, 1, 2], dtype=int))


@pytest.mark.parametrize('key', AVAILABLE_BACKENDS)
def test_sparse_jacobian_csc(key):
    be = Backend(key)
    n = 3
    x = be.real_symarray('x', n)
    exprs = [-x[0]] + [x[i-1] - x[i] for i in range(1, n-1)] + [x[-2]]
    sj, rowptrs, colvals = be.sparse_jacobian_csr(exprs, x)
    cb = be.Lambdify(x, sj)
    inp = np.arange(3.0, 3.0 + n)
    out = cb(inp)
    assert np.allclose(out, [-1, 1, -1, 1])
    assert np.all(rowptrs == np.array([0, 1, 3, 4], dtype=int))
    assert np.all(colvals == np.array([0, 0, 1, 1], dtype=int))