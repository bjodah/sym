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
def test_Lambdify_jacobian(key):
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


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym',),
                                       Backend.backends.keys()))
def test_broadcast(key):  # test is from symengine test suite
    be = Backend(key)
    a = np.linspace(-np.pi, np.pi)
    inp = np.vstack((np.cos(a), np.sin(a))).T  # 50 rows 2 cols
    x, y = be.symbols('x y')
    distance = be.Lambdify([x, y], [be.sqrt(x**2 + y**2)])
    assert np.allclose(distance([inp[0, 0], inp[0, 1]]), [1])
    dists = distance(inp)
    assert dists.shape == (50, 1)
    assert np.allclose(dists, 1)
