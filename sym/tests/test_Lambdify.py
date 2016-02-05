# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

import numpy as np

import pytest
from .. import Backend

# This tests Lambdify (see SymEngine), it offers essentially the same
# functionality as SymPy's lambdify but works for arbitrarily long input


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify_single_arg(key):
    be = Backend(key)
    x = be.Symbol('x')
    lmb = be.Lambdify([x], [x**2])
    assert np.allclose([4], lmb([2.0]))


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify_matrix(key):
    be = Backend(key)
    x, y = arr = be.symarray('x', 2)
    mat = be.Matrix(2, 2, [x, 1+y, 2*y*x**2, 3])
    lmb = be.Lambdify(arr, mat)
    result = lmb([3, 5])
    assert result.shape == (2, 2)
    assert np.allclose(result, [[3, 6], [90, 3]])


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify_matrix(key):
    be = Backend(key)
    x = be.Symbol('x')
    y = be.Symbol('y')
    a = be.Matrix(2, 1, [x+y, y*x**2])
    b = be.Matrix(2, 1, [x, y])
    J = a.jacobian(b)
    lmb = be.Lambdify(b, J)
    result = lmb([3, 5])
    assert result.shape == (2, 2)
    assert np.allclose(result, [[1, 1], [2*3*5, 3**2]])
