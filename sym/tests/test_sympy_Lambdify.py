# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np
import pytest
import sympy as sp
from sympy.utilities.lambdify import NUMPY_TRANSLATIONS
from .._sympy_Lambdify import create_Lambdify


lambdify_numpy_array = create_Lambdify(
    sp.MatrixBase, sp.sympify, sp.printing.lambdarepr.NumPyPrinter,
    sp.IndexedBase, sp.Symbol, NUMPY_TRANSLATIONS)[1]

try:
    import numba
except ImportError:
    numba = None


def test_lambdify_numpy_array():
    args = x, y = sp.symbols('x y')
    expr = x + sp.atan(y)
    cb = lambdify_numpy_array(args, expr)
    inp = np.array([17, 1])
    ref = 17 + np.arctan(1)
    assert np.allclose(cb(inp), ref)


def test_lambdify_numpy_array__broadcast():
    args = x, y = sp.symbols('x y')
    expr = x + sp.atan(y)
    cb = lambdify_numpy_array(args, expr)
    inp = np.array([[17, 1], [18, 2]])
    ref = [17 + np.arctan(1), 18 + np.arctan(2)]
    assert np.allclose(cb(inp), ref)

    inp2 = np.array([
        [[17, 1], [18, 2]],
        [[27, 21], [28, 22]]
    ])
    ref2 = [
        [17 + np.arctan(1), 18 + np.arctan(2)],
        [27 + np.arctan(21), 28 + np.arctan(22)]
    ]
    assert np.allclose(cb(inp2), ref2)


@pytest.mark.skipif(numba is None, reason='numba not available')
def test_lambdify_numpy_array__numba():
    args = x, y = sp.symbols('x y')
    expr = x + sp.atan(y)
    cb = lambdify_numpy_array(args, expr, use_numba=True)
    n = 500
    inp = np.empty((n, 2))
    inp[:, 0] = np.linspace(0, 1, n)
    inp[:, 1] = np.linspace(-10, 10, n)
    assert np.allclose(cb(inp), inp[:, 0] + np.arctan(inp[:, 1]))
