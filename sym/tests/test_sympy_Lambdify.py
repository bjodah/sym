# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from .._sympy_Lambdify import lambdify_numpy_array

from sympy import symbols, atan
import numpy as np


def test_lambdify_numpy_array():
    args = x, y = symbols('x y')
    expr = x + atan(y)
    cb = lambdify_numpy_array(args, expr)
    inp = np.array([17, 1])
    ref = 17 + np.arctan(1)
    assert np.allclose(cb(inp), ref)


def test_lambdify_numpy_array__broadcast():
    args = x, y = symbols('x y')
    expr = x + atan(y)
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
