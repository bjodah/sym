# -*- coding: utf-8 -*-
from __future__ import (absolute_import, division, print_function)

from functools import reduce
from operator import add, mul

import math
import numpy as np

import pytest
from pytest import raises
from .. import Backend

# This tests Lambdify (see SymEngine), it offers essentially the same
# functionality as SymPy's lambdify but works for arbitrarily long input

be = Backend('symengine')
x = be.Symbol('x')
try:
    be.Lambdify([x], [x**2], order='F')
except:
    SYME_ORDER_SKIP = ('symengine', 'sympysymengine')
else:
    SYME_ORDER = ()


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify_single_arg(key):
    be = Backend(key)
    x = be.Symbol('x')
    lmb = be.Lambdify([x], [x**2])
    assert np.allclose([4], lmb([2.0]))


@pytest.mark.parametrize('key', Backend.backends.keys())
def test_Lambdify_Abs(key):
    if key == 'symengine':
        return  # currently no Abs in symengine.py

    be = Backend(key)
    x = be.Symbol('x')
    lmb = be.Lambdify([x], [be.Abs(x)])
    assert np.allclose([2], lmb([-2.0]))


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
    dists = distance(inp)
    assert np.allclose(distance([inp[0, 0], inp[0, 1]]), [1])
    assert dists.shape == (50, 1)
    assert np.allclose(dists, 1)


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym',),
                                       Backend.backends.keys()))
def test_broadcast_shapes(key):  # test is from symengine test suite
    be = Backend(key)
    x, y = be.symbols('x y')
    lmb = be.Lambdify([x, y], [x+y, x-y, x/y])
    assert lmb(np.asarray([2, 3])).shape == (3,)
    assert lmb(np.asarray([[2, 3]])).shape == (1, 3)
    assert lmb(np.asarray([[[2, 3]]])).shape == (1, 1, 3)
    assert lmb(np.arange(5*7*6*2).reshape((5, 7, 6, 2))).shape == (5, 7, 6, 3)


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym', 'symcxx') + SYME_ORDER_SKIP,
                                       Backend.backends.keys()))
def test_broadcast_multiple_extra_dimensions(key):
    se = Backend(key)
    inp = np.arange(12.).reshape((4, 3, 1))
    x = se.symbols('x')
    cb = se.Lambdify([x], [x**2, x**3])
    assert np.allclose(cb([inp[0, 2]]), [4, 8])
    out = cb(inp)
    assert out.shape == (4, 3, 1, 2)
    out = out.squeeze()
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


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym',),
                                       Backend.backends.keys()))
def test_Lambdify_invalid_args(key):
    se = Backend(key)
    x = se.Symbol('x')
    log = se.Lambdify([x], [se.log(x)])
    div = se.Lambdify([x], [1/x])
    assert math.isnan(log([-1])[0])
    assert math.isinf(-log([0])[0])
    assert math.isinf(div([0])[0])
    assert math.isinf(-div([-0])[0])


def test_Lambdify_mpamath_mpf():
    import mpmath
    from mpmath import mpf
    mpmath.mp.dps = 30
    p0 = [mpf('0.7'), mpf('1.3')]
    p1 = [3]
    be = Backend('sympy')
    x, y, z = map(be.Symbol, 'xyz')
    lmb = be.Lambdify([x, y, z], [x*y*z - 1, -1 + be.exp(-y) + be.exp(-z) - 1/x], module='mpmath')
    p = np.concatenate((p0, p1))
    lmb(p)


def _Lambdify_heterogeneous_output(se):
    x, y = se.symbols('x, y')
    args = se.DenseMatrix(2, 1, [x, y])
    v = se.DenseMatrix(2, 1, [x**3 * y, (x+1)*(y+1)])
    jac = v.jacobian(args)
    exprs = [jac, x+y, v, (x+1)*(y+1)]
    lmb = se.Lambdify(args, *exprs)
    inp0 = 7, 11
    inp1 = 8, 13
    inp2 = 5, 9
    inp = np.array([inp0, inp1, inp2])
    o_j, o_xpy, o_v, o_xty = lmb(inp)
    for idx, (X, Y) in enumerate([inp0, inp1, inp2]):
        assert np.allclose(o_j[idx, ...], [[3 * X**2 * Y, X**3],
                                           [Y + 1, X + 1]])
        assert np.allclose(o_xpy[idx, ...], [X+Y])
        assert np.allclose(o_v[idx, ...], [[X**3 * Y], [(X+1)*(Y+1)]])
        assert np.allclose(o_xty[idx, ...], [(X+1)*(Y+1)])


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym', 'symcxx'),
                                       Backend.backends.keys()))
def test_Lambdify_heterogeneous_output(key):
    _Lambdify_heterogeneous_output(se=Backend(key))


def _test_Lambdify_scalar_vector_matrix(se):
    args = x, y = se.symbols('x y')
    vec = se.DenseMatrix([x+y, x*y])
    jac = vec.jacobian(se.DenseMatrix(args))
    f = se.Lambdify(args, x**y, vec, jac)
    assert f.n_exprs == 3
    s, v, m = f([2, 3])
    assert s == 2**3
    assert np.allclose(v, [[2+3], [2*3]])
    assert np.allclose(m, [
        [1, 1],
        [3, 2]
    ])

    for inp in [[2, 3, 5, 7], np.array([[2, 3], [5, 7]])]:
        s2, v2, m2 = f(inp)
        assert np.allclose(s2, [2**3, 5**7])
        assert np.allclose(v2, [
            [[2+3], [2*3]],
            [[5+7], [5*7]]
        ])
        assert np.allclose(m2, [
            [
                [1, 1],
                [3, 2]
            ],
            [
                [1, 1],
                [7, 5]
            ]
        ])


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym', 'symcxx') + SYME_ORDER_SKIP,
                                       Backend.backends.keys()))
def test_Lambdify_scalar_vector_matrix(key):
    _test_Lambdify_scalar_vector_matrix(se=Backend(key))


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym', 'symcxx') + SYME_ORDER_SKIP,
                                       Backend.backends.keys()))
def test_Lambdify_gh174(key):
    # Tests array broadcasting if the expressions form an N-dimensional array
    # of say shape (k, l, m) and it contains 'n' arguments (x1, ... xn), then
    # if the user provides a Fortran ordered (column-major) input array of shape
    # (n, o, p, q), then the returned array will be of shape (k, l, m, o, p, q)
    se = Backend(key)
    args = x, y = se.symbols('x y')
    vec1 = se.DenseMatrix([x, x**2, x**3])
    assert vec1.shape == (3, 1)
    assert np.asarray(vec1).shape == (3, 1)
    lmb1 = se.Lambdify([x], vec1)
    out1 = lmb1(3)
    assert out1.shape == (3, 1)
    assert np.all(out1 == [[3], [9], [27]])
    assert lmb1([2, 3]).shape == (2, 3, 1)
    lmb1.order = 'F'  # change order
    out1a = lmb1([2, 3])
    assert out1a.shape == (3, 1, 2)
    ref1a_squeeze = [[2, 3],
                     [4, 9],
                     [8, 27]]
    assert np.all(out1a.squeeze() == ref1a_squeeze)
    assert out1a.flags['F_CONTIGUOUS']
    assert not out1a.flags['C_CONTIGUOUS']

    lmb2c = se.Lambdify(args, vec1, x+y, order='C')
    lmb2f = se.Lambdify(args, vec1, x+y, order='F')
    for out2a in [lmb2c([2, 3]), lmb2f([2, 3])]:
        assert np.all(out2a[0] == [[2], [4], [8]])
        assert out2a[0].ndim == 2
        assert out2a[1] == 5
        assert out2a[1].ndim == 0
    inp2b = np.array([
        [2.0, 3.0],
        [1.0, 2.0],
        [0.0, 6.0]
    ])
    raises(ValueError, lambda: (lmb2c(inp2b.T)))
    out2c = lmb2c(inp2b)
    out2f = lmb2f(np.asfortranarray(inp2b.T))
    assert out2c[0].shape == (3, 3, 1)
    assert out2f[0].shape == (3, 1, 3)
    for idx, (_x, _y) in enumerate(inp2b):
        assert np.all(out2c[0][idx, ...] == [[_x], [_x**2], [_x**3]])

    assert np.all(out2c[1] == [5, 3, 6])
    assert np.all(out2f[1] == [5, 3, 6])
    assert out2c[1].shape == (3,)
    assert out2f[1].shape == (3,)

    def _mtx3(_x, _y):
        return [[_x**row_idx + _y**col_idx for col_idx in range(3)]
                for row_idx in range(4)]
    mtx3c = np.array(_mtx3(x, y), order='C')
    mtx3f = np.array(_mtx3(x, y), order='F')
    lmb3c = se.Lambdify([x, y], x*y, mtx3c, vec1, order='C')
    lmb3f = se.Lambdify([x, y], x*y, mtx3f, vec1, order='F')
    inp3c = np.array([[2., 3], [3, 4], [5, 7], [6, 2], [3, 1]])
    inp3f = np.asfortranarray(inp3c.T)
    raises(ValueError, lambda: (lmb3c(inp3c.T)))
    out3c = lmb3c(inp3c)
    assert out3c[0].shape == (5,)
    assert out3c[1].shape == (5, 4, 3)
    assert out3c[2].shape == (5, 3, 1)  # user can apply numpy.squeeze if they want to.
    for a, b in zip(out3c, lmb3c(np.ravel(inp3c))):
        assert np.all(a == b)

    out3f = lmb3f(inp3f)
    assert out3f[0].shape == (5,)
    assert out3f[1].shape == (4, 3, 5)
    assert out3f[2].shape == (3, 1, 5)  # user can apply numpy.squeeze if they want to.
    for a, b in zip(out3f, lmb3f(np.ravel(inp3f, order='F'))):
        assert np.all(a == b)

    for idx, (_x, _y) in enumerate(inp3c):
        assert out3c[0][idx] == _x*_y
        assert out3f[0][idx] == _x*_y
        assert np.all(out3c[1][idx, ...] == _mtx3(_x, _y))
        assert np.all(out3f[1][..., idx] == _mtx3(_x, _y))
        assert np.all(out3c[2][idx, ...] == [[_x], [_x**2], [_x**3]])
        assert np.all(out3f[2][..., idx] == [[_x], [_x**2], [_x**3]])


def _get_Ndim_args_exprs_funcs(order, se):
    args = x, y = se.symbols('x y')

    # Higher dimensional inputs
    def f_a(index, _x, _y):
        a, b, c, d = index
        return _x**a + _y**b + (_x+_y)**-d

    nd_exprs_a = np.zeros((3, 5, 1, 4), dtype=object, order=order)
    for index in np.ndindex(*nd_exprs_a.shape):
        nd_exprs_a[index] = f_a(index, x, y)

    def f_b(index, _x, _y):
        a, b, c = index
        return b/(_x + _y)

    nd_exprs_b = np.zeros((1, 7, 1), dtype=object, order=order)
    for index in np.ndindex(*nd_exprs_b.shape):
        nd_exprs_b[index] = f_b(index, x, y)
    return args, nd_exprs_a, nd_exprs_b, f_a, f_b


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym', 'symcxx') + SYME_ORDER_SKIP,
                                       Backend.backends.keys()))
def test_Lambdify_Ndimensional_order_C(key):
    se = Backend(key)
    args, nd_exprs_a, nd_exprs_b, f_a, f_b = _get_Ndim_args_exprs_funcs(order='C', se=se)
    lmb4 = se.Lambdify(args, nd_exprs_a, nd_exprs_b, order='C')
    nargs = len(args)

    inp_extra_shape = (3, 5, 4)
    inp_shape = inp_extra_shape + (nargs,)
    inp4 = np.arange(reduce(mul, inp_shape)*1.0).reshape(inp_shape, order='C')
    out4a, out4b = lmb4(inp4)
    assert out4a.ndim == 7
    assert out4a.shape == inp_extra_shape + nd_exprs_a.shape
    assert out4b.ndim == 6
    assert out4b.shape == inp_extra_shape + nd_exprs_b.shape
    raises(ValueError, lambda: (lmb4(inp4.T)))
    for b, c, d in np.ndindex(inp_extra_shape):
        _x, _y = inp4[b, c, d, :]
        for index in np.ndindex(*nd_exprs_a.shape):
            assert np.isclose(out4a[(b, c, d) + index], f_a(index, _x, _y))
        for index in np.ndindex(*nd_exprs_b.shape):
            assert np.isclose(out4b[(b, c, d) + index], f_b(index, _x, _y))


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym', 'symcxx') + SYME_ORDER_SKIP,
                                       Backend.backends.keys()))
def test_Lambdify_Ndimensional_order_F(key):
    se = Backend(key)
    args, nd_exprs_a, nd_exprs_b, f_a, f_b = _get_Ndim_args_exprs_funcs(order='F', se=se)
    lmb4 = se.Lambdify(args, nd_exprs_a, nd_exprs_b, order='F')
    nargs = len(args)

    inp_extra_shape = (3, 5, 4)
    inp_shape = (nargs,)+inp_extra_shape
    inp4 = np.arange(reduce(mul, inp_shape)*1.0).reshape(inp_shape, order='F')
    out4a, out4b = lmb4(inp4)
    assert out4a.ndim == 7
    assert out4a.shape == nd_exprs_a.shape + inp_extra_shape
    assert out4b.ndim == 6
    assert out4b.shape == nd_exprs_b.shape + inp_extra_shape
    raises(ValueError, lambda: (lmb4(inp4.T)))
    for b, c, d in np.ndindex(inp_extra_shape):
        _x, _y = inp4[:, b, c, d]
        for index in np.ndindex(*nd_exprs_a.shape):
            assert np.isclose(out4a[index + (b, c, d)], f_a(index, _x, _y))
        for index in np.ndindex(*nd_exprs_b.shape):
            assert np.isclose(out4b[index + (b, c, d)], f_b(index, _x, _y))


@pytest.mark.parametrize('key', filter(lambda k: k not in ('pysym', 'symcxx') + SYME_ORDER_SKIP,
                                       Backend.backends.keys()))
def test_Lambdify_inp_exceptions(key):
    se = Backend(key)
    args = x, y = se.symbols('x y')
    lmb1 = se.Lambdify([x], x**2)
    raises(ValueError, lambda: (lmb1([])))
    assert lmb1(4) == 16
    assert np.all(lmb1([4, 2]) == [16, 4])

    lmb2 = se.Lambdify(args, x**2+y**2)
    assert lmb2([2, 3]) == 13
    raises(ValueError, lambda: lmb2([]))
    raises(ValueError, lambda: lmb2([2]))
    raises(ValueError, lambda: lmb2([2, 3, 4]))
    assert np.all(lmb2([2, 3, 4, 5]) == [13, 16+25])

    def _mtx(_x, _y):
        return [
            [_x-_y, _y**2],
            [_x+_y, _x**2],
            [_x*_y, _x**_y]
        ]

    mtx = np.array(_mtx(x, y), order='F')
    lmb3 = se.Lambdify(args, mtx, order='F')
    inp3a = [2, 3]
    assert np.all(lmb3(inp3a) == _mtx(*inp3a))
    inp3b = np.array([2, 3, 4, 5, 3, 2, 1, 5])
    for inp in [inp3b, inp3b.tolist(), inp3b.reshape((2, 4), order='F')]:
        out3b = lmb3(inp)
        assert out3b.shape == (3, 2, 4)
        for i in range(4):
            assert np.all(out3b[..., i] == _mtx(*inp3b[2*i:2*(i+1)]))
    raises(ValueError, lambda: lmb3(inp3b.reshape((4, 2))))
    raises(ValueError, lambda: lmb3(inp3b.reshape((2, 4)).T))
