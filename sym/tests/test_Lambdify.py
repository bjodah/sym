# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from functools import reduce
from operator import add

import math
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


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym',),
                                       Backend.backends.keys()))
def test_broadcast_multiple_extra_dimensions(key):
    se = Backend(key)
    inp = np.arange(12.).reshape((4, 3, 1))
    x = se.symbols('x')
    cb = se.Lambdify([x], [x**2, x**3])
    assert np.allclose(cb([inp[0, 2]]), [4, 8])
    out = cb(inp)
    assert out.shape == (4, 3, 2)
    assert abs(out[2, 1, 0] - 7**2) < 1e-14
    assert abs(out[2, 1, 1] - 7**3) < 1e-14
    assert abs(out[-1, -1, 0] - 11**2) < 1e-14
    assert abs(out[-1, -1, 1] - 11**3) < 1e-14


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym',),
                                       Backend.backends.keys()))
def test_more_than_255_args(key):
    # SymPy's lambdify can handle at most 255 arguments
    # this is a proof of concept that this limitation does
    # not affect SymEngine's Lambdify class
    se = Backend(key)
    for n in [130, 257]:
        x = se.symarray('x', n)
        p, q, r = 17, 42, 13
        terms = [i*s for i, s in enumerate(x, p)]
        exprs = [reduce(add, terms), r + x[0], -99]
        callback = se.Lambdify(x, exprs)
        input_arr = np.arange(q, q + n*n).reshape((n, n))
        out = callback(input_arr)
        ref = np.empty((n, 3))
        coeffs = np.arange(p, p + n)
        for i in range(n):
            ref[i, 0] = coeffs.dot(np.arange(q + n*i, q + n*(i+1)))
            ref[i, 1] = q + n*i + r
        ref[:, 2] = -99
        assert np.allclose(out, ref)


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify(key):
    se = Backend(key)
    n = 7
    args = x, y, z = se.symbols('x y z')
    l = se.Lambdify(args, [x+y+z, x**2, (x-y)/z, x*y*z])
    assert np.allclose(l(range(n, n+len(args))),
                       [3*n+3, n**2, -1/(n+2), n*(n+1)*(n+2)])


def _get_2_to_2by2_numpy(se):
    args = x, y = se.symbols('x y')
    exprs = np.array([[x+y+1.0, x*y],
                      [x/y, x**y]])
    l = se.Lambdify(args, exprs)

    def check(A, inp):
        X, Y = inp
        assert abs(A[0, 0] - (X+Y+1.0)) < 1e-15
        assert abs(A[0, 1] - (X*Y)) < 1e-15
        assert abs(A[1, 0] - (X/Y)) < 1e-15
        assert abs(A[1, 1] - (X**Y)) < 1e-13
    return l, check


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify_2dim_numpy(key):
    se = Backend(key)
    lmb, check = _get_2_to_2by2_numpy(se)
    for inp in [(5, 7), np.array([5, 7]), [5.0, 7.0]]:
        A = lmb(inp)
        assert A.shape == (2, 2)
        check(A, inp)


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify_invalid_args(key):
    se = Backend(key)
    x = se.Symbol('x')
    log = se.Lambdify([x], [se.log(x)])
    div = se.Lambdify([x], [1/x])
    assert math.isnan(log([-1])[0])
    assert math.isinf(-log([0])[0])
    assert math.isinf(div([0])[0])
    assert math.isinf(-div([-0])[0])
